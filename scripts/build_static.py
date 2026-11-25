"""Build a static copy of the app: the page plus one JSON file per API response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app_server import (
    FORECAST_LOG_PATH,
    MLFLOW_DB_PATH,
    ROOT_DIR,
    TRAIN_STATE_PATH,
    build_daily_temperature_rows,
    build_rows,
    enrich_rows,
)
from forecast_eval import build_performance
from model_status import build_model_status
from static_page import API_DIR, static_page, write_page  # noqa: F401 (static_page is re-exported)


# More than the page shows (24 h, 7 days), so it can drop passed hours and days between builds.
FORECAST_HOURS = 48
DAILY_DAYS = 8
PERFORMANCE_DAYS = 30
MODEL_RUNS = 20

def build_payloads() -> dict[str, dict]:
    forecast = build_rows(FORECAST_HOURS)
    forecast["source"] = "static"
    forecast["rows"] = enrich_rows(forecast["rows"])
    daily = build_daily_temperature_rows(days=DAILY_DAYS, include_today=True)
    daily["source"] = "static"
    return {
        "forecast-classes": forecast,
        "forecast-daily": daily,
        "performance": build_performance(FORECAST_LOG_PATH, PERFORMANCE_DAYS),
        "model-status": build_model_status(
            model_dir=ROOT_DIR / "model",
            mlflow_db_path=MLFLOW_DB_PATH,
            train_state_path=TRAIN_STATE_PATH,
            limit=MODEL_RUNS,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT_DIR / "site", help="Output folder")
    args = parser.parse_args()

    # Fetch everything first, so a failed fetch never leaves a half-built site.
    payloads = build_payloads()

    write_page(args.out)
    api_dir = args.out / API_DIR
    api_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        # allow_nan=False: browsers cannot parse NaN.
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        (api_dir / f"{name}.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
