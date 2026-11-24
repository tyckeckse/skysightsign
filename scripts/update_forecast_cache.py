from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from app_server import ROOT_DIR, UPPSALA_LAT, UPPSALA_LON, build_rows
from log_utils import kv, setup_logger


DEFAULT_HOURS = 24
CACHE_PATH = ROOT_DIR / "data" / "forecast_cache.json"
SMHI_FORECAST_URL = (
    "https://opendata-download-metfcst.smhi.se/api/category/snow1g/version/1/geotype/point/"
    "lon/{lon:.4f}/lat/{lat:.4f}/data.json"
)
LOG_PATH = ROOT_DIR / "data" / "forecast_log.jsonl"
DEFAULT_LOG_PATH = ROOT_DIR / "data" / "cache_refresh.log"
TRAIN_STATE_PATH = ROOT_DIR / "data" / "train_state.json"
RETRAIN_SCRIPT = ROOT_DIR / "ops" / "run_retrain_background.ps1"
MAX_TRAIN_AGE = timedelta(hours=25)  # 24h + 1h slack, so we don't race the 12:00 task
# One line per catch-up attempt; its modification time spaces attempts out when training keeps failing.
CATCH_UP_LOG_PATH = ROOT_DIR / "data" / "retrain_catch_up.log"
MIN_CATCH_UP_INTERVAL = timedelta(hours=6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uppdatera lokal prognoscache for weather-appen.")
    parser.add_argument("--hours", type=int, default=DEFAULT_HOURS, help="Antal timmar att cacha (default: 24).")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH, help=f"Path till cache-logg (default: {DEFAULT_LOG_PATH}).")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Loggniva (default: INFO).",
    )
    parser.add_argument("--no-catch-up", action="store_true", help="Starta aldrig träning härifrån.")
    return parser.parse_args()


# SMHI snow1g Wsymb2 (symbol_code 1-27) mapped to our 4 precipitation classes.
_SMHI_SYMBOL_RAIN = {8, 9, 10, 11, 18, 19, 20, 21}
_SMHI_SYMBOL_SLEET = {12, 13, 14, 22, 23, 24}
_SMHI_SYMBOL_SNOW = {15, 16, 17, 25, 26, 27}


def smhi_symbol_to_precip(symbol_code: object) -> str | None:
    """Map SMHI snow1g Wsymb2 symbol_code (1-27) to Inget/Regn/Sno/Snoblandat."""
    try:
        code = int(symbol_code)
    except (TypeError, ValueError):
        return None
    if code in _SMHI_SYMBOL_SLEET:
        return "Snoblandat"
    if code in _SMHI_SYMBOL_SNOW:
        return "Sno"
    if code in _SMHI_SYMBOL_RAIN:
        return "Regn"
    if 1 <= code <= 7:
        return "Inget"
    return None


def fetch_smhi_forecast() -> dict[str, dict[str, object]]:
    """SMHI's own forecast of cloud cover (%) and precipitation class, keyed by UTC hour."""
    try:
        response = requests.get(SMHI_FORECAST_URL.format(lat=UPPSALA_LAT, lon=UPPSALA_LON), timeout=30)
        response.raise_for_status()
        series = response.json().get("timeSeries", [])
    except (requests.RequestException, ValueError):
        return {}
    out: dict[str, dict[str, object]] = {}
    for item in series:
        data = item.get("data") or {}
        t = item.get("time")
        if not t:
            continue
        cloud_octas = data.get("cloud_area_fraction")
        out[t] = {
            "cloud": None if cloud_octas is None else cloud_octas / 8 * 100,
            "precip": smhi_symbol_to_precip(data.get("symbol_code")),
        }
    return out


def append_forecast_log(payload: dict) -> int:
    """Append one line per forecast hour (day + night), so cloud and precip can be scored against SMHI later."""
    issued = payload.get("generated_at_utc")
    smhi = fetch_smhi_forecast()
    lines = []
    for row in payload.get("rows", []):
        t = row["time_utc"]
        smhi_hour = smhi.get(t, {})
        is_night = bool(row.get("is_night"))
        lines.append(
            json.dumps(
                {
                    "issued_at_utc": issued,
                    "time_utc": t,
                    "is_night": is_night,
                    # Cloud fields are kept daytime-only (None at night) so cloud evaluation stays daytime-only.
                    "pred_cloud": None if is_night else row.get("pred_cloud"),
                    "main_cloud": None if is_night else row.get("main_cloud"),
                    "p_overcast": None if is_night else row.get("p_overcast"),
                    "om_cloud": None if is_night else row.get("cloud_cover"),
                    "smhi_cloud": None if is_night else smhi_hour.get("cloud"),
                    "pred_precip": row.get("precip_label"),
                    "p_precip": row.get("p_precip"),
                    "om_precip": row.get("om_precip"),
                    "smhi_precip": smhi_hour.get("precip"),
                }
            )
        )
    if lines:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return len(lines)


def training_overdue(state_path: Path = TRAIN_STATE_PATH, now: datetime | None = None) -> bool:
    """True if the last successful training is too old, e.g. because the nightly run was cut off."""
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        last = datetime.fromisoformat(state["last_train_utc"].replace("Z", "+00:00"))
    except (OSError, ValueError, KeyError):
        return True
    return (now or datetime.now(timezone.utc)) - last > MAX_TRAIN_AGE


def catch_up_allowed(log_path: Path = CATCH_UP_LOG_PATH, now: datetime | None = None) -> bool:
    """False if a catch-up training was already started within MIN_CATCH_UP_INTERVAL."""
    try:
        last = datetime.fromtimestamp(log_path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return True
    return (now or datetime.now(timezone.utc)) - last >= MIN_CATCH_UP_INTERVAL


def note_catch_up(log_path: Path = CATCH_UP_LOG_PATH) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") + "\n")


def main() -> None:
    args = parse_args()
    logger = setup_logger("sunshine.cache_refresh", args.log_file, level=args.log_level)

    try:
        if args.hours < 1:
            raise ValueError("--hours maste vara minst 1.")
        if args.hours > 48:
            raise ValueError("--hours maste vara <= 48.")

        logger.info(kv(svc="cache", event="cache_refresh_start", hours=args.hours))
        payload = build_rows(args.hours)
        payload["source"] = "cache-refresh"

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        append_forecast_log(payload)

        if args.no_catch_up:
            pass
        elif training_overdue() and not catch_up_allowed():
            logger.info(kv(svc="cache", event="retrain_catch_up_skipped", msg="Tried recently; waiting before retrying"))
        elif training_overdue():
            note_catch_up()
            logger.info(kv(svc="cache", event="retrain_catch_up_start"))
            # Same wrapper as the nightly task, so training settings live in one place.
            subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(RETRAIN_SCRIPT)],
                check=False,
            )

        logger.info(
            kv(
                svc="cache",
                event="cache_refresh_done",
                rows=len(payload.get("rows", [])),
                generated_at=payload.get("generated_at_utc"),
                cache_path=CACHE_PATH,
            )
        )
    except Exception as exc:
        logger.exception(
            kv(
                svc="cache",
                event="cache_refresh_failed",
                msg="Cache refresh failed",
                error=str(exc),
            )
        )
        raise


if __name__ == "__main__":
    main()
