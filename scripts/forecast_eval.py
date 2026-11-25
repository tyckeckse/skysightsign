from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from precip_model import PRECIP_CLASSES, smhi_weather_code_to_class

SMHI_LATEST_URL = (
    "https://opendata-download-metobs.smhi.se/api/version/latest/parameter/16/station/97530/period/latest-months/data.json"
)
SMHI_PRECIP_LATEST_URL = (
    "https://opendata-download-metobs.smhi.se/api/version/latest/parameter/13/station/97530/period/latest-months/data.json"
)
OBS_TTL_SECONDS = 1800
# A target hour's forecast is locked by the run made this many hours before it. 1 h is the main number.
WINDOW_HOURS = (1, 6, 12, 24)

_obs_cache: dict[str, Any] = {"at": 0.0, "data": {}}
_precip_obs_cache: dict[str, Any] = {"at": 0.0, "data": {}}


def fetch_recent_precip_obs() -> dict[str, str]:
    """SMHI observed precipitation class (Inget/Regn/Sno/Snoblandat) for the last months, cached 30 min."""
    if time.time() - _precip_obs_cache["at"] < OBS_TTL_SECONDS and _precip_obs_cache["data"]:
        return _precip_obs_cache["data"]
    response = requests.get(SMHI_PRECIP_LATEST_URL, timeout=30)
    response.raise_for_status()
    obs: dict[str, str] = {}
    for item in response.json().get("value", []):
        raw_val = item.get("value")
        if raw_val is None:
            continue
        try:
            cls = smhi_weather_code_to_class(int(raw_val))
        except (TypeError, ValueError):
            continue
        if cls is None:
            continue
        dt = datetime.fromtimestamp(item["date"] / 1000, tz=timezone.utc)
        obs[dt.strftime("%Y-%m-%dT%H:%M")] = PRECIP_CLASSES[cls]
    _precip_obs_cache.update(at=time.time(), data=obs)
    return obs


def fetch_recent_obs() -> dict[str, float]:
    """SMHI observed cloud cover (%) for the last months, keyed by "YYYY-MM-DDTHH:MM" (UTC). Cached 30 min."""
    if time.time() - _obs_cache["at"] < OBS_TTL_SECONDS and _obs_cache["data"]:
        return _obs_cache["data"]
    response = requests.get(SMHI_LATEST_URL, timeout=30)
    response.raise_for_status()
    obs: dict[str, float] = {}
    for item in response.json().get("value", []):
        if item.get("value") is None:
            continue
        dt = datetime.fromtimestamp(item["date"] / 1000, tz=timezone.utc)
        obs[dt.strftime("%Y-%m-%dT%H:%M")] = float(item["value"])
    _obs_cache.update(at=time.time(), data=obs)
    return obs


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


SERIES = ("model", "om", "smhi")


def _summary(pairs: list[dict]) -> dict:
    n = len(pairs)
    out: dict[str, Any] = {"n": n}
    for name in SERIES:
        errs = [p[name] for p in pairs]
        out[f"{name}_mae"] = round(sum(abs(e) for e in errs) / n, 1) if n else None
        out[f"{name}_bias"] = round(sum(errs) / n, 1) if n else None
    return out


def _hour_rows(day_pairs: list[dict]) -> list[dict]:
    """Forecast values and signed errors (forecast - observed) per target hour."""
    return [
        {
            "time_utc": p["time_utc"],
            "obs": p["obs"],
            "lead_h": round(p["lead"], 1),
            **{name: round(p["obs"] + p[name], 1) for name in SERIES},
            **{f"{name}_err": round(p[name], 1) for name in SERIES},
        }
        for p in sorted(day_pairs, key=lambda p: p["time_utc"])
    ]


def drop_duplicate_rows(log_rows: list[dict]) -> list[dict]:
    """Keep one row per issue minute and target hour; overlapping refresh jobs log the same forecast twice."""
    seen: set[tuple[str, str]] = set()
    unique = []
    for row in log_rows:
        key = (str(row.get("issued_at_utc"))[:16], str(row.get("time_utc")))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def lock_forecasts(
    log_rows: list[dict],
    obs: dict[str, Any],
    fields: dict[str, str],
    days: int,
    now: datetime,
    issued_from: datetime | None,
) -> dict[int, list[dict]]:
    """Per window, one forecast per target hour: the first run in the clock hour that starts WINDOW_HOURS before it
    and has all three forecasts. `fields` names the log field for each of SERIES.
    """
    cutoff = now.timestamp() - days * 86400
    locked: dict[tuple[int, datetime], dict] = {}
    for row in log_rows:
        try:
            target = _parse(row["time_utc"])
            issued = _parse(row["issued_at_utc"])
        except (KeyError, ValueError, TypeError):
            continue
        run_hour = issued.replace(minute=0, second=0, microsecond=0)
        window = (target - run_hour).total_seconds() / 3600
        if window not in WINDOW_HOURS or target.timestamp() < cutoff:
            continue
        if issued_from is not None and issued < issued_from:
            continue
        preds = {name: row.get(fields[name]) for name in SERIES}
        actual = obs.get(row["time_utc"][:16])
        if actual is None or None in preds.values():
            continue
        key = (int(window), target)
        if key in locked and locked[key]["issued"] <= issued:
            continue
        locked[key] = {
            "day": target.strftime("%Y-%m-%d"),
            "time_utc": target.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "issued": issued,
            "obs": actual,
            "lead": (target - issued).total_seconds() / 3600,
            **preds,
        }
    return {w: [p for (window, _), p in locked.items() if window == w] for w in WINDOW_HOURS}


def score(
    log_rows: list[dict],
    obs: dict[str, float],
    days: int,
    now: datetime | None = None,
    issued_from: datetime | None = None,
) -> dict:
    """Compare locked cloud forecasts (ours, Open-Meteo's, SMHI's) to SMHI observations. Errors are forecast - observed."""
    now = now or datetime.now(timezone.utc)
    fields = {"model": "pred_cloud", "om": "om_cloud", "smhi": "smhi_cloud"}
    by_window = {
        window: [{**p, **{name: p[name] - p["obs"] for name in SERIES}} for p in pairs]
        for window, pairs in lock_forecasts(log_rows, obs, fields, days, now, issued_from).items()
    }
    main = by_window[WINDOW_HOURS[0]]
    by_day = []
    for day in sorted({p["day"] for p in main}):
        day_pairs = [p for p in main if p["day"] == day]
        by_day.append({"day": day, **_summary(day_pairs), "hours": _hour_rows(day_pairs)})
    by_sky = {
        "clear": [p for p in main if p["obs"] <= 12],
        "mixed": [p for p in main if 12 < p["obs"] < 95],
        "overcast": [p for p in main if p["obs"] >= 95],
    }
    return {
        "days": days,
        "generated_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall": _summary(main),
        "by_day": by_day,
        "by_window": [{"window_h": w, **_summary(pairs)} for w, pairs in by_window.items()],
        "by_sky": [{"sky": sky, **_summary(pairs)} for sky, pairs in by_sky.items()],
        "issued_from_utc": issued_from.strftime("%Y-%m-%dT%H:%M:%SZ") if issued_from else None,
    }


def devil_live_from(log_rows: list[dict]) -> datetime | None:
    """When the first forecast with the overcast devil was logged (rows with p_overcast), or None if none yet."""
    issued = [row["issued_at_utc"] for row in log_rows if row.get("p_overcast") is not None and row.get("issued_at_utc")]
    return _parse(min(issued)) if issued else None


def precip_live_from(log_rows: list[dict]) -> datetime | None:
    """When the first forecast with precipitation predictions was logged, or None if none yet."""
    issued = [row["issued_at_utc"] for row in log_rows if row.get("pred_precip") is not None and row.get("issued_at_utc")]
    return _parse(min(issued)) if issued else None


def _binary_f1_pct(actuals: list[str], preds: list[str]) -> float | None:
    tp = sum(a != "Inget" and p != "Inget" for a, p in zip(actuals, preds))
    fp = sum(a == "Inget" and p != "Inget" for a, p in zip(actuals, preds))
    fn = sum(a != "Inget" and p == "Inget" for a, p in zip(actuals, preds))
    if tp == 0 and fp == 0 and fn == 0:
        return None
    denom = 2 * tp + fp + fn
    return round(100.0 * (2 * tp) / denom, 1) if denom else 0.0


def _precip_summary(pairs: list[dict]) -> dict[str, Any]:
    n = len(pairs)
    actuals = [p["obs"] for p in pairs]
    out: dict[str, Any] = {"n": n}
    for name in SERIES:
        ps = [p[name] for p in pairs]
        out[f"{name}_acc"] = round(100.0 * sum(a == p for a, p in zip(actuals, ps)) / n, 1) if n else None
        out[f"{name}_f1"] = _binary_f1_pct(actuals, ps) if n else None
    return out


def _precip_hour_rows(day_pairs: list[dict]) -> list[dict]:
    """Observed and predicted precipitation classes per target hour."""
    return [
        {"time_utc": p["time_utc"], "obs": p["obs"], "lead_h": round(p["lead"], 1), **{name: p[name] for name in SERIES}}
        for p in sorted(day_pairs, key=lambda p: p["time_utc"])
    ]


def score_precip(
    log_rows: list[dict],
    obs: dict[str, str],
    days: int,
    now: datetime | None = None,
    issued_from: datetime | None = None,
) -> dict:
    """Compare locked precipitation classes (ours, Open-Meteo's, SMHI's) to SMHI parameter 13 observations."""
    now = now or datetime.now(timezone.utc)
    fields = {"model": "pred_precip", "om": "om_precip", "smhi": "smhi_precip"}
    by_window = lock_forecasts(log_rows, obs, fields, days, now, issued_from)
    main = by_window[WINDOW_HOURS[0]]
    by_day = []
    for day in sorted({p["day"] for p in main}):
        day_pairs = [p for p in main if p["day"] == day]
        by_day.append({"day": day, **_precip_summary(day_pairs), "hours": _precip_hour_rows(day_pairs)})
    return {
        **_precip_summary(main),
        "issued_from_utc": issued_from.strftime("%Y-%m-%dT%H:%M:%SZ") if issued_from else None,
        "by_day": by_day,
        "by_window": [{"window_h": w, **_precip_summary(pairs)} for w, pairs in by_window.items()],
    }


def build_performance(log_path: Path, days: int) -> dict:
    # Clean slate: only forecasts made with the devil are scored. Older log lines are kept,
    # since they hold SMHI forecasts we may train on later.
    rows = drop_duplicate_rows(read_log(log_path))
    start = devil_live_from(rows) or datetime.now(timezone.utc)  # nothing to score until the devil is live
    result = score(rows, fetch_recent_obs(), days, issued_from=start)
    p_start = precip_live_from(rows)
    try:
        precip_obs = fetch_recent_precip_obs() if p_start is not None else {}
    except requests.RequestException:
        precip_obs = {}
    result["precip"] = score_precip(rows, precip_obs, days, issued_from=p_start)
    return result
