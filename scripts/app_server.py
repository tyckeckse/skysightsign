from __future__ import annotations

import json
import math
import posixpath
import socket
import sys
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import requests
import xgboost as xgb
from forecast_eval import build_performance
from icon_features import ICON_FEATURES, add_icon_features
from log_utils import kv, setup_logger
from model_common import (
    DEVIL_MODEL_FILE,
    PRECIP_META_FILE,
    PRECIP_MODEL_FILE,
    ROOT_DIR,
    SOLAR_META_FILE,
    SOLAR_MODEL_FILE,
    UPPSALA_LAT,
    UPPSALA_LON,
    WEATHER_FEATURES,
    day_of_year,
    load_meta,
    resolve_meta_path,
    time_features,
)
from model_status import build_model_status
from overcast_devil import DEVIL_MAX_LEAD_HOURS, EXTRA_FEATURES, add_extra_cloud, apply_devil, overcast_probability
from precip_model import (
    PRECIP_ACTIVE_LABELS,
    PRECIP_EXTRA_HOURLY,
    aggregate_daily_precip,
    predict_precip_for_hour,
)
from sun_time_utils import parse_utc_ms


MODEL_PATH = ROOT_DIR / "model" / SOLAR_MODEL_FILE
META_PATH = ROOT_DIR / "model" / SOLAR_META_FILE
DEVIL_PATH = ROOT_DIR / "model" / DEVIL_MODEL_FILE
PRECIP_MODEL_PATH = ROOT_DIR / "model" / PRECIP_MODEL_FILE
PRECIP_META_PATH = ROOT_DIR / "model" / PRECIP_META_FILE
CACHE_PATH = ROOT_DIR / "data" / "forecast_cache.json"
FORECAST_LOG_PATH = ROOT_DIR / "data" / "forecast_log.jsonl"
TRAIN_STATE_PATH = ROOT_DIR / "data" / "train_state.json"
MLFLOW_DB_PATH = ROOT_DIR / "mlflow.db"
APP_LOG_PATH = ROOT_DIR / "data" / "app_server.log"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_HOURS = 24
DEFAULT_DAILY_DAYS = 7
MAX_DAILY_DAYS = 14
DEFAULT_PORT = 8000
CLASS_NAMES = ["Mulet", "Vaxlande", "Soligt"]


LOGGER = setup_logger("sunshine.app", APP_LOG_PATH, level="INFO")
booster_cache: xgb.Booster | None = None
meta_cache: dict | None = None
model_sig_cache: tuple[int, ...] | None = None
model_cache_lock = threading.Lock()


def ms_to_utc_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def current_full_hour_utc_ms(now_ms: int) -> int:
    dt = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
    floored = dt.replace(minute=0, second=0, microsecond=0)
    return int(floored.timestamp() * 1000)


def safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        out = float(value)
        if math.isfinite(out):
            return out
    return None


def wind_to_ms(value: float | None, unit: str | None) -> float | None:
    if value is None:
        return None
    normalized = (unit or "").strip().lower()
    if normalized in {"m/s", "ms"}:
        return value
    if normalized in {"km/h", "kmh", "kph"}:
        return value / 3.6
    # Unknown/missing unit: keep original to avoid silently dropping data.
    return value


def hourly_is_day(hourly: dict, i: int) -> bool:
    values = hourly.get("is_day") or []
    return i < len(values) and values[i] == 1


def hourly_weather(hourly: dict, i: int) -> dict[str, float | None]:
    weather = {}
    for name in list(dict.fromkeys(WEATHER_FEATURES + PRECIP_EXTRA_HOURLY + ICON_FEATURES + EXTRA_FEATURES)):
        values = hourly.get(name) or []
        weather[name] = safe_float(values[i]) if i < len(values) else None
    return weather


def display_label_for(sky_label: str | None, precip_label: str | None) -> str | None:
    if isinstance(precip_label, str) and precip_label in PRECIP_ACTIVE_LABELS:
        return precip_label
    return sky_label


def open_meteo_precip_label(weather: dict[str, float | None]) -> str | None:
    rain = weather.get("rain")
    showers = weather.get("showers")
    snowfall = weather.get("snowfall")
    if rain is None and showers is None and snowfall is None:
        return None
    liquid = (rain or 0.0) + (showers or 0.0)
    snow = snowfall or 0.0
    if liquid > 0.0 and snow > 0.0:
        return "Snoblandat"
    if snow > 0.0:
        return "Sno"
    if liquid > 0.0:
        return "Regn"
    return "Inget"


def predict_hourly_precip(meta: dict, utc_ms: int, weather: dict[str, float | None]) -> dict[str, Any]:
    precip_booster = meta.get("_precip_booster")
    precip_meta = meta.get("_precip_meta")
    if precip_booster is None or not isinstance(precip_meta, dict):
        return {
            "precip_label": None,
            "p_precip": None,
            "p_rain": None,
            "p_snow": None,
            "p_sleet": None,
            "precip_intensity": None,
        }
    return predict_precip_for_hour(precip_booster, precip_meta, utc_ms, weather)


def feature_vector(utc_ms: int, weather: dict[str, float | None], feature_order: list[str]) -> list[float | None]:
    # Built from the model's own feature_order, so old and new models both get matching inputs.
    dt = datetime.fromtimestamp(utc_ms / 1000, tz=timezone.utc)
    values = {**weather, **time_features(dt)}
    return [values.get(name) for name in feature_order]


def sun_index_from_cloud(cloud_pct: float) -> float:
    # The model predicts SMHI-observed cloud cover (%); clear sky = 1.0.
    return max(0.0, min(1.0, 1.0 - cloud_pct / 100.0))


def classify_from_sun_index(meta: dict, sun_index: float) -> str:
    th = meta.get("thresholds", {})
    sunny_idx = float(th.get("sunny_idx", 0.65))
    mixed_idx = float(th.get("mixed_idx", 0.35))
    if sun_index >= sunny_idx:
        return "Soligt"
    if sun_index >= mixed_idx:
        return "Vaxlande"
    return "Mulet"


def predict_hourly_sun(
    booster: xgb.Booster,
    meta: dict,
    utc_ms: int,
    weather: dict[str, float | None],
    is_day: bool,
    now_ms: int | None = None,
) -> dict[str, Any]:
    if not is_day:
        return {
            "is_night": True,
            "pred_cloud": None,
            "main_cloud": None,
            "p_overcast": None,
            "sun_index": None,
            "sun_label": "Natt",
        }

    feature_order = meta.get("feature_order", [])
    fv = feature_vector(utc_ms, weather, feature_order)
    # ICON gaps (beyond its ~7-day range, or ICON unreachable) go in as NaN, which XGBoost can handle.
    missing = [name for name, value in zip(feature_order, fv) if value is None and name not in ICON_FEATURES]
    if missing:
        raise RuntimeError("Forecast-data saknar värden: " + ", ".join(missing))

    x = np.array([fv], dtype=float)
    main_cloud = float(booster.predict(xgb.DMatrix(x, feature_names=feature_order))[0])
    pred_cloud = main_cloud
    p_overcast = None
    devil = meta.get("_devil_booster")
    threshold = (meta.get("devil") or {}).get("threshold")
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000) if now_ms is None else now_ms
    within_horizon = (utc_ms - now_ms) <= DEVIL_MAX_LEAD_HOURS * 3_600_000
    if devil is not None and within_horizon:
        # The devil gets a vote before the forecast is locked: pull to 100 if it is sure enough of overcast.
        extra = np.array([[safe_float(weather.get(name)) for name in EXTRA_FEATURES]], dtype=float)
        p_overcast = float(overcast_probability(devil, x, feature_order, extra, np.array([main_cloud]))[0])
        pred_cloud = float(apply_devil(np.array([main_cloud]), np.array([p_overcast]), threshold)[0])
    sun_index = sun_index_from_cloud(pred_cloud)
    return {
        "is_night": False,
        "pred_cloud": pred_cloud,
        "main_cloud": main_cloud,
        "p_overcast": p_overcast,
        "sun_index": sun_index,
        "sun_label": classify_from_sun_index(meta, sun_index),
    }


def load_model() -> tuple[xgb.Booster, dict]:
    global booster_cache, meta_cache, model_sig_cache
    with model_cache_lock:
        resolved_meta_path = resolve_meta_path(META_PATH)
        if not MODEL_PATH.exists() or resolved_meta_path is None:
            raise FileNotFoundError("Saknar modellfiler. Kör: python scripts\\train_model.py")

        resolved_precip_meta = resolve_meta_path(PRECIP_META_PATH)
        devil_mtime = DEVIL_PATH.stat().st_mtime_ns if DEVIL_PATH.exists() else 0
        precip_model_mtime = PRECIP_MODEL_PATH.stat().st_mtime_ns if PRECIP_MODEL_PATH.exists() else 0
        precip_meta_mtime = resolved_precip_meta.stat().st_mtime_ns if resolved_precip_meta is not None else 0
        current_sig = (
            MODEL_PATH.stat().st_mtime_ns,
            resolved_meta_path.stat().st_mtime_ns,
            devil_mtime,
            precip_model_mtime,
            precip_meta_mtime,
        )
        if (
            booster_cache is not None
            and meta_cache is not None
            and model_sig_cache is not None
            and model_sig_cache == current_sig
        ):
            return booster_cache, meta_cache

        try:
            booster = xgb.Booster()
            booster.load_model(MODEL_PATH)
            meta = load_meta(META_PATH)
            if meta is None:
                raise RuntimeError(f"Ogiltig metadatafil: {resolved_meta_path}")
            # Kept in memory next to the meta; only models trained with a devil have one.
            meta["_devil_booster"] = None
            if meta.get("devil") and DEVIL_PATH.exists():
                devil = xgb.Booster()
                devil.load_model(DEVIL_PATH)
                meta["_devil_booster"] = devil
            meta["_precip_booster"] = None
            meta["_precip_meta"] = None
            if PRECIP_MODEL_PATH.exists() and resolved_precip_meta is not None:
                precip_booster = xgb.Booster()
                precip_booster.load_model(PRECIP_MODEL_PATH)
                precip_meta = load_meta(PRECIP_META_PATH)
                if precip_meta is not None:
                    meta["_precip_booster"] = precip_booster
                    meta["_precip_meta"] = precip_meta
            booster_cache = booster
            meta_cache = meta
            model_sig_cache = current_sig
            LOGGER.info(
                kv(
                    svc="app",
                    event="model_reload",
                    model_path=MODEL_PATH,
                    meta_path=resolved_meta_path,
                )
            )
            return booster, meta
        except Exception as exc:
            if booster_cache is not None and meta_cache is not None:
                LOGGER.warning(
                    kv(
                        svc="app",
                        event="model_reload",
                        msg="Reload failed, using cached model",
                        error=str(exc),
                    )
                )
                return booster_cache, meta_cache
            raise


def fetch_forecast() -> dict:
    hourly_fields = list(dict.fromkeys(WEATHER_FEATURES + PRECIP_EXTRA_HOURLY + ["is_day", "temperature_2m", "wind_speed_10m"]))
    params = {
        "latitude": f"{UPPSALA_LAT:.4f}",
        "longitude": f"{UPPSALA_LON:.4f}",
        "hourly": ",".join(hourly_fields),
        "wind_speed_unit": "ms",
        "forecast_days": "2",
        "timezone": "UTC",
    }
    response = requests.get(FORECAST_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    add_icon_features(FORECAST_URL, params, payload.setdefault("hourly", {}))
    add_extra_cloud(FORECAST_URL, params, payload["hourly"])
    return payload


def fetch_daily_forecast(forecast_days: int) -> dict:
    hourly_fields = list(dict.fromkeys(WEATHER_FEATURES + PRECIP_EXTRA_HOURLY + ["is_day"]))
    params = {
        "latitude": f"{UPPSALA_LAT:.4f}",
        "longitude": f"{UPPSALA_LON:.4f}",
        "hourly": ",".join(hourly_fields),
        "daily": "temperature_2m_min,temperature_2m_max,wind_speed_10m_mean",
        "wind_speed_unit": "ms",
        "forecast_days": str(forecast_days),
        "timezone": "UTC",
    }
    response = requests.get(FORECAST_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    add_icon_features(FORECAST_URL, params, payload.setdefault("hourly", {}))
    add_extra_cloud(FORECAST_URL, params, payload["hourly"])
    return payload


def parse_bool_query(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("include_today must be one of 1/0/true/false")


def build_daily_sun_summary(payload: dict, booster: xgb.Booster, meta: dict) -> dict[str, dict[str, Any]]:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    clouds = hourly.get("cloud_cover", [])
    precip = hourly.get("precipitation", [])
    if not (len(times) == len(clouds) == len(precip) and len(times) > 0):
        raise RuntimeError("Daily forecast-data saknar hourly-serier.")

    grouped: dict[str, dict[str, list[Any]]] = {}

    for i, time_str in enumerate(times):
        if not isinstance(time_str, str):
            continue
        utc_ms = parse_utc_ms(time_str)
        date_str = datetime.fromtimestamp(utc_ms / 1000, tz=timezone.utc).date().isoformat()
        cloud_cover = safe_float(clouds[i])
        weather_i = hourly_weather(hourly, i)
        prediction = predict_hourly_sun(booster, meta, utc_ms, weather_i, hourly_is_day(hourly, i))
        if bool(prediction.get("is_night")):
            continue

        day_bucket = grouped.setdefault(date_str, {"sun_index": [], "cloud_cover": [], "precip": []})
        sun_index = safe_float(prediction.get("sun_index"))
        if sun_index is not None:
            day_bucket["sun_index"].append(sun_index)
        if cloud_cover is not None:
            day_bucket["cloud_cover"].append(cloud_cover)
        precip_pred = predict_hourly_precip(meta, utc_ms, weather_i)
        if precip_pred.get("precip_label") is not None:
            day_bucket["precip"].append(precip_pred)

    precip_meta = meta.get("_precip_meta") or {}
    precip_threshold = float(precip_meta.get("precip_threshold", 0.40))
    sleet_mix_ratio = float(precip_meta.get("sleet_mix_ratio", 0.30))

    summary: dict[str, dict[str, Any]] = {}
    for date_str, values in grouped.items():
        sun_values = values["sun_index"]
        cloud_values = values["cloud_cover"]
        sun_index_mean = None
        if sun_values:
            sun_index_mean = round(sum(sun_values) / len(sun_values), 2)
        cloud_cover_mean = None
        if cloud_values:
            cloud_cover_mean = round(sum(cloud_values) / len(cloud_values), 1)
        sun_label = None if sun_index_mean is None else classify_from_sun_index(meta, sun_index_mean)
        precip_summary = (
            aggregate_daily_precip(values["precip"], precip_threshold=precip_threshold, sleet_mix_ratio=sleet_mix_ratio)
            if values["precip"]
            else {"precip_label": None, "precip_intensity": None, "p_precip_max": None}
        )
        precip_label = precip_summary.get("precip_label")
        summary[date_str] = {
            "sun_index_mean": sun_index_mean,
            "sun_label": sun_label,
            "precip_label": precip_label,
            "precip_intensity": precip_summary.get("precip_intensity"),
            "p_precip_max": precip_summary.get("p_precip_max"),
            "display_label": display_label_for(sun_label, precip_label),
            "cloud_cover_mean_pct": cloud_cover_mean,
        }
    return summary


def build_daily_temperature_rows(days: int, include_today: bool) -> dict:
    fetch_days = days if include_today else min(MAX_DAILY_DAYS, days + 1)
    payload = fetch_daily_forecast(fetch_days)
    booster, meta = load_model()
    daily = payload.get("daily", {})
    dates = daily.get("time", [])
    mins = daily.get("temperature_2m_min", [])
    maxs = daily.get("temperature_2m_max", [])
    winds = daily.get("wind_speed_10m_mean", [])
    wind_unit = str(payload.get("daily_units", {}).get("wind_speed_10m_mean", "m/s"))
    if not (len(dates) == len(mins) == len(maxs) and len(dates) > 0):
        raise RuntimeError("Daily forecast-data saknar serier.")
    daily_sun_summary = build_daily_sun_summary(payload, booster, meta)

    today_utc = datetime.now(timezone.utc).date().isoformat()
    rows: list[dict] = []
    for i, date_str in enumerate(dates):
        if not isinstance(date_str, str):
            continue
        if not include_today and date_str == today_utc:
            continue
        tmin = safe_float(mins[i])
        tmax = safe_float(maxs[i])
        wind_mean = wind_to_ms(safe_float(winds[i]), wind_unit) if i < len(winds) else None
        sun_summary = daily_sun_summary.get(date_str, {})
        rows.append(
            {
                "date": date_str,
                "temp_min_c": None if tmin is None else round(tmin, 1),
                "temp_max_c": None if tmax is None else round(tmax, 1),
                "wind_mean_ms": None if wind_mean is None else round(wind_mean, 1),
                "sun_index_mean": sun_summary.get("sun_index_mean"),
                "sun_label": sun_summary.get("sun_label"),
                "precip_label": sun_summary.get("precip_label"),
                "precip_intensity": sun_summary.get("precip_intensity"),
                "p_precip_max": sun_summary.get("p_precip_max"),
                "display_label": sun_summary.get("display_label") or sun_summary.get("sun_label"),
            }
        )
        if len(rows) >= days:
            break

    if not rows:
        raise RuntimeError("Ingen daglig prognosrad hittades.")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "generated_at_utc": generated_at,
        "location": {"name": "Uppsala", "lat": UPPSALA_LAT, "lon": UPPSALA_LON},
        "days": rows,
    }


def build_rows(hours: int) -> dict:
    booster, meta = load_model()
    payload = fetch_forecast()
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    clouds = hourly.get("cloud_cover", [])
    precip = hourly.get("precipitation", [])
    temps = hourly.get("temperature_2m", [])
    winds = hourly.get("wind_speed_10m", [])
    wind_unit = str(payload.get("hourly_units", {}).get("wind_speed_10m", ""))

    if not (len(times) == len(clouds) == len(precip) == len(temps) == len(winds) and len(times) > 0):
        raise RuntimeError("Forecast-data saknar serier.")

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    anchor_ms = current_full_hour_utc_ms(now_ms)
    rows = []

    for i, t in enumerate(times):
        utc_ms = parse_utc_ms(t)
        if utc_ms < anchor_ms:
            continue
        cloud_cover = safe_float(clouds[i])
        precipitation = safe_float(precip[i])
        temp_c = safe_float(temps[i])
        wind_ms = wind_to_ms(safe_float(winds[i]), wind_unit)
        weather_i = hourly_weather(hourly, i)
        sun_prediction = predict_hourly_sun(booster, meta, utc_ms, weather_i, hourly_is_day(hourly, i))
        precip_prediction = predict_hourly_precip(meta, utc_ms, weather_i)
        night = bool(sun_prediction["is_night"])
        pred_cloud = safe_float(sun_prediction["pred_cloud"])
        main_cloud = safe_float(sun_prediction["main_cloud"])
        p_overcast = safe_float(sun_prediction["p_overcast"])
        sun_index = safe_float(sun_prediction["sun_index"])
        label = str(sun_prediction["sun_label"])
        precip_label = precip_prediction.get("precip_label")
        rows.append(
            {
                "time_utc": ms_to_utc_iso(utc_ms),
                "label": label,
                "precip_label": precip_label,
                "om_precip": open_meteo_precip_label(weather_i),
                "display_label": display_label_for(label, precip_label),
                "precip_intensity": precip_prediction.get("precip_intensity"),
                "p_precip": precip_prediction.get("p_precip"),
                "p_rain": precip_prediction.get("p_rain"),
                "p_snow": precip_prediction.get("p_snow"),
                "p_sleet": precip_prediction.get("p_sleet"),
                "cloud_cover": None if cloud_cover is None else round(cloud_cover, 1),
                "precipitation": None if precipitation is None else round(precipitation, 2),
                "temp_c": None if temp_c is None else round(temp_c, 1),
                "wind_ms": None if wind_ms is None else round(wind_ms, 1),
                "pred_cloud": None if pred_cloud is None else round(pred_cloud, 1),
                "main_cloud": None if main_cloud is None else round(main_cloud, 1),
                "p_overcast": None if p_overcast is None else round(p_overcast, 3),
                "sun_index": None if sun_index is None else round(sun_index, 2),
                "is_night": night,
            }
        )
        if len(rows) >= hours:
            break

    if not rows:
        raise RuntimeError("Ingen framtida prognosrad hittades.")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "generated_at_utc": generated_at,
        "latest_prediction_at_utc": generated_at,
        "anchor_utc": ms_to_utc_iso(anchor_ms),
        "wind_speed_unit": "m/s",
        "source_wind_speed_unit": wind_unit,
        "location": {"name": "Uppsala", "lat": UPPSALA_LAT, "lon": UPPSALA_LON},
        "rows": rows,
    }


def normalize_row(row: dict) -> dict:
    out = dict(row)
    cloud_cover = safe_float(out.get("cloud_cover"))
    precipitation = safe_float(out.get("precipitation"))
    temp_c = safe_float(out.get("temp_c"))
    wind_ms = safe_float(out.get("wind_ms"))
    pred_cloud = safe_float(out.get("pred_cloud"))
    sun_index = safe_float(out.get("sun_index"))
    is_night = bool(out.get("is_night")) or out.get("label") == "Natt"

    if sun_index is None and pred_cloud is not None:
        sun_index = sun_index_from_cloud(pred_cloud)
    if pred_cloud is not None:
        pred_cloud = round(pred_cloud, 1)
    if sun_index is not None:
        sun_index = round(sun_index, 2)

    out["cloud_cover"] = None if cloud_cover is None else round(cloud_cover, 1)
    out["precipitation"] = None if precipitation is None else round(precipitation, 2)
    out["temp_c"] = None if temp_c is None else round(temp_c, 1)
    out["wind_ms"] = None if wind_ms is None else round(wind_ms, 1)
    out.pop("pred_sw", None)
    out["pred_cloud"] = pred_cloud
    out["sun_index"] = sun_index
    out["is_night"] = is_night
    precip_label = out.get("precip_label")
    out["precip_label"] = precip_label if isinstance(precip_label, str) else None
    intensity = out.get("precip_intensity")
    out["precip_intensity"] = intensity if isinstance(intensity, str) else None
    for key in ("p_precip", "p_rain", "p_snow", "p_sleet"):
        val = safe_float(out.get(key))
        out[key] = None if val is None else round(val, 3)
    return out


def threshold_label_for_row(meta: dict, row: dict) -> str:
    if bool(row.get("is_night")):
        return "Natt"
    sun_index = safe_float(row.get("sun_index"))
    if sun_index is not None:
        return classify_from_sun_index(meta, sun_index)
    maybe_label = row.get("label")
    if isinstance(maybe_label, str) and maybe_label in CLASS_NAMES:
        return maybe_label
    return "Mulet"


def enrich_rows(rows: list[dict]) -> list[dict]:
    _, meta = load_model()
    out_rows: list[dict] = []
    for row in rows:
        time_utc = row.get("time_utc")
        if not isinstance(time_utc, str):
            continue
        row_norm = normalize_row(row)
        row_norm["label"] = threshold_label_for_row(meta, row_norm)
        row_norm["display_label"] = display_label_for(row_norm["label"], row_norm.get("precip_label"))
        out_rows.append(row_norm)
    return out_rows


def trim_rows(payload: dict, hours: int) -> dict:
    if payload.get("wind_speed_unit") != "m/s":
        raise RuntimeError("Cache-format fel: wind_speed_unit måste vara m/s.")
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise RuntimeError("Cache-format fel: rows ar inte en lista.")

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    anchor_ms = current_full_hour_utc_ms(now_ms)
    kept = []
    for row in rows:
        time_utc = row.get("time_utc")
        if not isinstance(time_utc, str):
            continue
        row_ms = parse_utc_ms(time_utc)
        if row_ms < anchor_ms:
            continue
        kept.append(normalize_row(row))
        if len(kept) >= hours:
            break

    if not kept:
        raise RuntimeError("Cache saknar framtida rader.")

    out = dict(payload)
    out["rows"] = kept
    out["anchor_utc"] = kept[0].get("time_utc", ms_to_utc_iso(anchor_ms))
    generated_at = out.get("generated_at_utc")
    out["latest_prediction_at_utc"] = generated_at if isinstance(generated_at, str) else None
    return out


def load_cached_rows(hours: int) -> dict:
    if not CACHE_PATH.exists():
        raise FileNotFoundError("Ingen cachefil hittades.")
    with CACHE_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return trim_rows(payload, hours)


def is_allowed_static_path(raw_path: str) -> bool:
    """Allow only the page itself and weather icons; block repo files like mlflow.db, logs, and scripts."""
    decoded = unquote(raw_path).replace("\\", "/")
    normalized = posixpath.normpath(decoded)
    if normalized in ("/", "/index.html"):
        return True
    return normalized.startswith("/icons/") and ".." not in decoded.split("/")


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def end_headers(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(HTTPStatus.NOT_FOUND)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/forecast-classes":
            self.handle_forecast_api(parsed.query)
            return
        if parsed.path == "/api/forecast-daily":
            self.handle_forecast_daily_api(parsed.query)
            return
        if parsed.path == "/api/performance":
            self.handle_performance_api(parsed.query)
            return
        if parsed.path == "/api/model-status":
            self.handle_model_status_api(parsed.query)
            return
        if parsed.path == "/api/health":
            self.send_json({"ok": True})
            return
        if not is_allowed_static_path(parsed.path):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        super().do_GET()

    def log_message(self, format: str, *args) -> None:
        return

    def handle_forecast_api(self, query: str) -> None:
        try:
            params = parse_qs(query)
            hours = int(params.get("hours", [str(DEFAULT_HOURS)])[0])
            if hours < 1:
                raise ValueError("hours must be >= 1")
            if hours > 48:
                hours = 48
        except ValueError as exc:
            self.send_api_error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        try:
            payload = load_cached_rows(hours)
            payload["source"] = "cache"
            payload["rows"] = enrich_rows(payload.get("rows", []))
            self.send_json(payload)
            LOGGER.info(
                kv(
                    svc="app",
                    event="forecast_cache_hit",
                    path="/api/forecast-classes",
                    method="GET",
                    rows=len(payload.get("rows", [])),
                    hours=hours,
                )
            )
        except (FileNotFoundError, RuntimeError, json.JSONDecodeError) as cache_exc:
            LOGGER.warning(
                kv(
                    svc="app",
                    event="forecast_live_fallback",
                    path="/api/forecast-classes",
                    method="GET",
                    reason=type(cache_exc).__name__,
                    error=str(cache_exc),
                )
            )
            try:
                payload = build_rows(hours)
                payload["source"] = "live"
                payload["rows"] = enrich_rows(payload.get("rows", []))
                self.send_json(payload)
            except requests.RequestException as exc:
                self.send_api_error(HTTPStatus.BAD_GATEWAY, f"Forecast fetch failed: {exc}")
            except Exception as exc:
                self.send_api_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
        except requests.RequestException as exc:
            self.send_api_error(HTTPStatus.BAD_GATEWAY, f"Forecast fetch failed: {exc}")
        except Exception as exc:
            self.send_api_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def handle_performance_api(self, query: str) -> None:
        try:
            params = parse_qs(query)
            days = max(1, min(int(params.get("days", ["14"])[0]), 60))
            self.send_json(build_performance(FORECAST_LOG_PATH, days))
        except ValueError as exc:
            self.send_api_error(HTTPStatus.BAD_REQUEST, str(exc))
        except requests.RequestException as exc:
            self.send_api_error(HTTPStatus.BAD_GATEWAY, f"SMHI fetch failed: {exc}")
        except Exception as exc:
            self.send_api_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def handle_model_status_api(self, query: str) -> None:
        try:
            params = parse_qs(query)
            limit = max(1, min(int(params.get("limit", ["20"])[0]), 50))
            payload = build_model_status(
                model_dir=ROOT_DIR / "model",
                mlflow_db_path=MLFLOW_DB_PATH,
                train_state_path=TRAIN_STATE_PATH,
                limit=limit,
            )
            self.send_json(payload)
        except ValueError as exc:
            self.send_api_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self.send_api_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def handle_forecast_daily_api(self, query: str) -> None:
        try:
            params = parse_qs(query)
            days = int(params.get("days", [str(DEFAULT_DAILY_DAYS)])[0])
            if days < 1 or days > MAX_DAILY_DAYS:
                raise ValueError(f"days must be between 1 and {MAX_DAILY_DAYS}")
            include_today = parse_bool_query(params.get("include_today", ["1"])[0], default=True)

            payload = build_daily_temperature_rows(days=days, include_today=include_today)
            payload["source"] = "live"
            self.send_json(payload)
            LOGGER.info(
                kv(
                    svc="app",
                    event="forecast_daily_hit",
                    path="/api/forecast-daily",
                    method="GET",
                    days=days,
                    include_today=include_today,
                    rows=len(payload.get("days", [])),
                )
            )
        except ValueError as exc:
            self.send_api_error(HTTPStatus.BAD_REQUEST, str(exc))
        except requests.RequestException as exc:
            self.send_api_error(HTTPStatus.BAD_GATEWAY, f"Forecast fetch failed: {exc}")
        except Exception as exc:
            self.send_api_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_api_error(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

        log_payload = {
            "svc": "app",
            "event": "api_error",
            "path": urlparse(self.path).path,
            "method": self.command,
            "status": int(status),
            "msg": message,
        }
        if int(status) >= 500:
            LOGGER.error(kv(**log_payload))
        else:
            LOGGER.warning(kv(**log_payload))


class ExclusiveHTTPServer(ThreadingHTTPServer):
    """Refuses to start if the port is taken.

    The standard server sets SO_REUSEADDR, which on Windows lets a second server share the port silently
    (two logon-task runs gave two servers on 8000). SO_EXCLUSIVEADDRUSE makes the second bind fail instead.
    """

    allow_reuse_address = False

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):  # Windows only
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def main() -> None:
    try:
        server = ExclusiveHTTPServer(("127.0.0.1", DEFAULT_PORT), AppHandler)
    except OSError as exc:
        LOGGER.warning(kv(svc="app", event="server_start_skipped", msg="Port already in use", port=DEFAULT_PORT, error=str(exc)))
        print(f"Port {DEFAULT_PORT} is already in use; not starting a second server.")
        sys.exit(1)
    LOGGER.info(
        kv(
            svc="app",
            event="server_start",
            host="127.0.0.1",
            port=DEFAULT_PORT,
        )
    )
    print(f"App server running on http://127.0.0.1:{DEFAULT_PORT}/")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
