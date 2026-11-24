"""Shared constants, cyclical time features, and JSON/legacy-pickle metadata helpers."""
from __future__ import annotations

import json
import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
UPPSALA_LAT = 59.8586
UPPSALA_LON = 17.6389

# Hourly Open-Meteo fields shared by the solar and precipitation models.
WEATHER_FEATURES = [
    "cloud_cover",
    "precipitation",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "relative_humidity_2m",
]

SOLAR_MODEL_FILE = "uppsala_solar_model.ubj"
SOLAR_META_FILE = "uppsala_solar_meta.json"
DEVIL_MODEL_FILE = "uppsala_overcast_devil.ubj"
PRECIP_MODEL_FILE = "uppsala_precip_model.ubj"
PRECIP_META_FILE = "uppsala_precip_meta.json"


def day_of_year(dt: datetime) -> int:
    return int(dt.strftime("%j"))


def time_features(dt: datetime) -> dict[str, float]:
    """Cyclical hour-of-day and day-of-year features in UTC."""
    two_pi = 2.0 * math.pi
    hour = dt.hour
    doy = day_of_year(dt)
    return {
        "hour_sin": math.sin(two_pi * hour / 24.0),
        "hour_cos": math.cos(two_pi * hour / 24.0),
        "doy_sin": math.sin(two_pi * doy / 365.25),
        "doy_cos": math.cos(two_pi * doy / 365.25),
    }


def resolve_meta_path(json_path: Path) -> Path | None:
    """Return the JSON metadata path if present, falling back to a legacy .pkl file."""
    if json_path.exists():
        return json_path
    legacy_path = json_path.with_suffix(".pkl")
    if legacy_path.exists():
        return legacy_path
    return None


def load_meta(json_path: Path) -> dict[str, Any] | None:
    """Load model metadata from JSON, with read fallback to legacy .pkl state archives."""
    actual = resolve_meta_path(json_path)
    if actual is None:
        return None
    if actual.suffix == ".pkl":
        with actual.open("rb") as f:
            data = pickle.load(f)
    else:
        data = json.loads(actual.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def save_meta(json_path: Path, payload: dict[str, Any]) -> None:
    """Write model metadata as UTF-8 JSON and clean up any legacy .pkl sibling."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    json_path.write_text(text, encoding="utf-8")
    legacy_path = json_path.with_suffix(".pkl")
    if legacy_path != json_path:
        legacy_path.unlink(missing_ok=True)
