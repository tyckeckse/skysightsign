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
LEAD_BUCKETS = [(0, 6), (6, 12), (12, 24), (24, 48)]

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


def _summary(errs: dict[str, list[float]]) -> dict:
    n = len(errs["model"])
    out: dict[str, Any] = {"n": n}
    for name in SERIES:
        out[f"{name}_mae"] = round(sum(abs(e) for e in errs[name]) / n, 1) if n else None
        out[f"{name}_bias"] = round(sum(errs[name]) / n, 1) if n else None
    return out


def _errs(pairs: list[dict]) -> dict[str, list[float]]:
    return {name: [p[name] for p in pairs] for name in SERIES}


def _by_hour(day_pairs: list[dict]) -> list[dict]:
    """Latest forecast per target hour (shortest lead): forecast values and signed errors (forecast - observed)."""
    hours = []
    for t in sorted({p["time_utc"] for p in day_pairs}):
        latest = min((p for p in day_pairs if p["time_utc"] == t), key=lambda p: p["lead"])
        hours.append(
            {
                "time_utc": t,
                "obs": latest["obs"],
                "lead_h": round(latest["lead"], 1),
                **{name: round(latest["obs"] + latest[name], 1) for name in SERIES},
                **{f"{name}_err": round(latest[name], 1) for name in SERIES},
            }
        )
    return hours


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


def score(
    log_rows: list[dict],
    obs: dict[str, float],
    days: int,
    now: datetime | None = None,
    issued_from: datetime | None = None,
) -> dict:
    """Compare logged forecasts (ours, Open-Meteo's, SMHI's) to SMHI observations. Errors are forecast - observed.

    Only hours where all three forecasts exist are scored, so the comparison is like for like.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now.timestamp() - days * 86400
    pairs = []
    for row in log_rows:
        try:
            target = _parse(row["time_utc"])
            issued = _parse(row["issued_at_utc"])
        except (KeyError, ValueError, TypeError):
            continue
        preds = {"model": row.get("pred_cloud"), "om": row.get("om_cloud"), "smhi": row.get("smhi_cloud")}
        actual = obs.get(row["time_utc"][:16])
        if actual is None or None in preds.values() or target.timestamp() < cutoff:
            continue
        if issued_from is not None and issued < issued_from:
            continue
        if issued > target:  # made after the hour began, so not a real forecast
            continue
        pairs.append(
            {
                "day": target.strftime("%Y-%m-%d"),
                "time_utc": target.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "obs": actual,
                "lead": (target - issued).total_seconds() / 3600,
                **{name: value - actual for name, value in preds.items()},
            }
        )

    by_day = []
    for day in sorted({p["day"] for p in pairs}):
        day_pairs = [p for p in pairs if p["day"] == day]
        by_day.append({"day": day, **_summary(_errs(day_pairs)), "hours": _by_hour(day_pairs)})
    by_lead = [
        {"from_h": lo, "to_h": hi, **_summary(_errs([p for p in pairs if lo <= p["lead"] < hi]))}
        for lo, hi in LEAD_BUCKETS
    ]
    return {
        "days": days,
        "generated_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall": _summary(_errs(pairs)),
        "by_day": by_day,
        "by_lead": by_lead,
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


def _precip_by_hour(day_pairs: list[dict]) -> list[dict]:
    """Latest forecast per target hour (shortest lead): observed and predicted precipitation classes."""
    hours = []
    for t in sorted({p["time_utc"] for p in day_pairs}):
        latest = min((p for p in day_pairs if p["time_utc"] == t), key=lambda p: p["lead"])
        hours.append(
            {
                "time_utc": t,
                "obs": latest["obs"],
                "lead_h": round(latest["lead"], 1),
                **{name: latest[name] for name in SERIES},
            }
        )
    return hours


def score_precip(
    log_rows: list[dict],
    obs: dict[str, str],
    days: int,
    now: datetime | None = None,
    issued_from: datetime | None = None,
) -> dict:
    """Compare logged precipitation classes (ours, Open-Meteo's, SMHI's) to SMHI parameter 13 observations."""
    now = now or datetime.now(timezone.utc)
    cutoff = now.timestamp() - days * 86400
    pairs: list[dict[str, Any]] = []

    for row in log_rows:
        try:
            target = _parse(row["time_utc"])
            issued = _parse(row["issued_at_utc"])
        except (KeyError, ValueError, TypeError):
            continue
        preds = {"model": row.get("pred_precip"), "om": row.get("om_precip"), "smhi": row.get("smhi_precip")}
        actual = obs.get(row["time_utc"][:16])
        if actual is None or None in preds.values() or target.timestamp() < cutoff:
            continue
        if issued_from is not None and issued < issued_from:
            continue
        if issued > target:  # made after the hour began, so not a real forecast
            continue
        pairs.append(
            {
                "day": target.strftime("%Y-%m-%d"),
                "time_utc": target.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "obs": actual,
                "lead": (target - issued).total_seconds() / 3600,
                **{name: str(value) for name, value in preds.items()},
            }
        )

    by_day = []
    for day in sorted({p["day"] for p in pairs}):
        day_pairs = [p for p in pairs if p["day"] == day]
        by_day.append({"day": day, **_precip_summary(day_pairs), "hours": _precip_by_hour(day_pairs)})

    return {
        **_precip_summary(pairs),
        "unique_hours": len({p["time_utc"] for p in pairs}),
        "issued_from_utc": issued_from.strftime("%Y-%m-%dT%H:%M:%SZ") if issued_from else None,
        "by_day": by_day,
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
