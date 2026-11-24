"""Cloud layers from DWD's ICON model, used as extra model inputs next to Open-Meteo's default forecast."""
from __future__ import annotations

import requests

ICON_FIELDS = ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"]
ICON_FEATURES = [f"icon_{field}" for field in ICON_FIELDS]


def add_icon_features(url: str, base_params: dict, hourly: dict, timeout: int = 120) -> None:
    """Fetch ICON for the same request and add icon_* series to `hourly`, aligned on its "time" list.

    Hours ICON does not cover (or a failed request) become None, so the caller decides how to handle gaps.
    """
    params = {key: value for key, value in base_params.items() if key not in ("hourly", "daily")}
    params.update({"hourly": ",".join(ICON_FIELDS), "models": "icon_seamless"})
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        icon = response.json().get("hourly", {})
    except (requests.RequestException, ValueError):
        icon = {}

    index = {t: i for i, t in enumerate(icon.get("time", []))}
    for field, name in zip(ICON_FIELDS, ICON_FEATURES):
        # Single-model requests usually return plain names; accept the suffixed form too.
        series = icon.get(field) or icon.get(f"{field}_icon_seamless") or []
        hourly[name] = [
            series[index[t]] if t in index and index[t] < len(series) else None for t in hourly.get("time", [])
        ]
