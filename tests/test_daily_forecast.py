from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import app_server


def iso_hour(date_str: str, hour: int) -> str:
    return f"{date_str}T{hour:02d}:00"


def build_payload(
    daily_dates: list[str],
    mins: list[float],
    maxs: list[float],
    hourly_times: list[str],
    clouds: list[float],
    precip: list[float],
) -> dict:
    return {
        "daily": {
            "time": daily_dates,
            "temperature_2m_min": mins,
            "temperature_2m_max": maxs,
        },
        "hourly": {
            "time": hourly_times,
            "cloud_cover": clouds,
            "precipitation": precip,
        },
    }


class DailyForecastTests(unittest.TestCase):
    def test_build_daily_sun_summary_averages_daylight_hours(self) -> None:
        date_str = "2026-03-03"
        payload = build_payload(
            [date_str],
            [1.0],
            [8.0],
            [iso_hour(date_str, 10), iso_hour(date_str, 12)],
            [20.0, 60.0],
            [0.0, 0.0],
        )

        def fake_predict(_booster, _meta, utc_ms, _weather, _is_day):
            hour = datetime.fromtimestamp(utc_ms / 1000, tz=timezone.utc).hour
            return {
                "is_night": False,
                "pred_cloud": 100.0,
                "sun_index": 0.4 if hour == 10 else 0.8,
                "sun_label": "Vaxlande",
            }

        with mock.patch.object(app_server, "predict_hourly_sun", side_effect=fake_predict):
            summary = app_server.build_daily_sun_summary(payload, booster=object(), meta={})

        self.assertEqual(
            summary[date_str],
            {
                "sun_index_mean": 0.6,
                "sun_label": "Vaxlande",
                "precip_label": None,
                "precip_intensity": None,
                "p_precip_max": None,
                "display_label": "Vaxlande",
                "cloud_cover_mean_pct": 40.0,
            },
        )

    def test_build_daily_sun_summary_skips_night_hours(self) -> None:
        date_str = "2026-03-03"
        payload = build_payload(
            [date_str],
            [1.0],
            [8.0],
            [iso_hour(date_str, 2), iso_hour(date_str, 12), iso_hour(date_str, 13), iso_hour(date_str, 23)],
            [99.0, 20.0, 40.0, 88.0],
            [0.0, 0.0, 0.0, 0.0],
        )

        def fake_predict(_booster, _meta, utc_ms, _weather, _is_day):
            hour = datetime.fromtimestamp(utc_ms / 1000, tz=timezone.utc).hour
            if hour in {2, 23}:
                return {"is_night": True, "pred_cloud": None, "sun_index": None, "sun_label": "Natt"}
            return {
                "is_night": False,
                "pred_cloud": 100.0,
                "sun_index": 0.5 if hour == 12 else 0.7,
                "sun_label": "Vaxlande",
            }

        with mock.patch.object(app_server, "predict_hourly_sun", side_effect=fake_predict):
            summary = app_server.build_daily_sun_summary(payload, booster=object(), meta={})

        self.assertEqual(summary[date_str]["sun_index_mean"], 0.6)
        self.assertEqual(summary[date_str]["cloud_cover_mean_pct"], 30.0)

    def test_build_daily_sun_summary_raises_on_misaligned_hourly_series(self) -> None:
        payload = {
            "daily": {
                "time": ["2026-03-03"],
            },
            "hourly": {
                "time": ["2026-03-03T10:00", "2026-03-03T11:00"],
                "cloud_cover": [20.0],
                "precipitation": [0.0, 0.0],
            },
        }

        with self.assertRaisesRegex(RuntimeError, "hourly-serier"):
            app_server.build_daily_sun_summary(payload, booster=object(), meta={})

    def test_build_daily_temperature_rows_excludes_today_when_requested(self) -> None:
        today = datetime.now(timezone.utc).date()
        tomorrow = today + timedelta(days=1)
        today_str = today.isoformat()
        tomorrow_str = tomorrow.isoformat()
        payload = build_payload(
            [today_str, tomorrow_str],
            [1.0, 2.0],
            [5.0, 6.0],
            [iso_hour(today_str, 12), iso_hour(tomorrow_str, 12)],
            [20.0, 30.0],
            [0.0, 0.0],
        )

        with (
            mock.patch.object(app_server, "fetch_daily_forecast", return_value=payload) as fetch_mock,
            mock.patch.object(app_server, "load_model", return_value=(object(), {})),
            mock.patch.object(
                app_server,
                "build_daily_sun_summary",
                return_value={
                    today_str: {"sun_index_mean": 0.4, "sun_label": "Vaxlande", "cloud_cover_mean_pct": 20.0},
                    tomorrow_str: {"sun_index_mean": 0.7, "sun_label": "Soligt", "cloud_cover_mean_pct": 30.0},
                },
            ),
        ):
            result = app_server.build_daily_temperature_rows(days=1, include_today=False)

        fetch_mock.assert_called_once_with(2)
        self.assertEqual(
            result["days"],
            [
                {
                    "date": tomorrow_str,
                    "temp_min_c": 2.0,
                    "temp_max_c": 6.0,
                    "wind_mean_ms": None,
                    "sun_index_mean": 0.7,
                    "sun_label": "Soligt",
                    "precip_label": None,
                    "precip_intensity": None,
                    "p_precip_max": None,
                    "display_label": "Soligt",
                }
            ],
        )

    def test_build_daily_temperature_rows_raises_on_misaligned_daily_series(self) -> None:
        payload = {
            "daily": {
                "time": ["2026-03-03", "2026-03-04"],
                "temperature_2m_min": [1.0],
                "temperature_2m_max": [5.0, 6.0],
            },
            "hourly": {
                "time": ["2026-03-03T12:00"],
                "cloud_cover": [20.0],
                "precipitation": [0.0],
            },
        }

        with (
            mock.patch.object(app_server, "fetch_daily_forecast", return_value=payload),
            mock.patch.object(app_server, "load_model", return_value=(object(), {})),
        ):
            with self.assertRaisesRegex(RuntimeError, "saknar serier"):
                app_server.build_daily_temperature_rows(days=2, include_today=True)

    def test_build_daily_temperature_rows_returns_null_sun_index_without_daylight_rows(self) -> None:
        date_str = "2026-03-03"
        payload = build_payload(
            [date_str],
            [1.0],
            [8.0],
            [iso_hour(date_str, 1), iso_hour(date_str, 2)],
            [95.0, 90.0],
            [0.0, 0.0],
        )

        with (
            mock.patch.object(app_server, "fetch_daily_forecast", return_value=payload),
            mock.patch.object(app_server, "load_model", return_value=(object(), {})),
            mock.patch.object(
                app_server,
                "predict_hourly_sun",
                return_value={"is_night": True, "pred_cloud": None, "sun_index": None, "sun_label": "Natt"},
            ),
        ):
            result = app_server.build_daily_temperature_rows(days=1, include_today=True)

        self.assertEqual(
            result["days"][0],
            {
                "date": date_str,
                "temp_min_c": 1.0,
                "temp_max_c": 8.0,
                "wind_mean_ms": None,
                "sun_index_mean": None,
                "sun_label": None,
                "precip_label": None,
                "precip_intensity": None,
                "p_precip_max": None,
                "display_label": None,
            },
        )


class FeatureVectorTests(unittest.TestCase):
    def test_feature_vector_follows_model_feature_order(self) -> None:
        utc_ms = int(datetime(2026, 6, 1, 12, tzinfo=timezone.utc).timestamp() * 1000)
        weather = {"cloud_cover": 50.0, "precipitation": 0.0, "cloud_cover_low": 10.0}

        fv = app_server.feature_vector(utc_ms, weather, ["cloud_cover_low", "cloud_cover", "hour_cos"])

        self.assertEqual(fv, [10.0, 50.0, -1.0])

    def test_predict_hourly_sun_raises_when_model_feature_is_missing(self) -> None:
        utc_ms = int(datetime(2026, 6, 1, 12, tzinfo=timezone.utc).timestamp() * 1000)
        meta = {"feature_order": ["cloud_cover", "cloud_cover_low"]}

        with self.assertRaisesRegex(RuntimeError, "cloud_cover_low"):
            app_server.predict_hourly_sun(object(), meta, utc_ms, {"cloud_cover": 50.0}, True)

    def test_hourly_is_day_reads_open_meteo_flag(self) -> None:
        hourly = {"is_day": [0, 1]}

        self.assertEqual(
            [app_server.hourly_is_day(hourly, 0), app_server.hourly_is_day(hourly, 1), app_server.hourly_is_day(hourly, 2)],
            [False, True, False],
        )


if __name__ == "__main__":
    unittest.main()
