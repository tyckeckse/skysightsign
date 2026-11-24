from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import precip_model as pm


class SmhiCodeMappingTest(unittest.TestCase):
    def test_maps_core_weather_codes(self):
        self.assertEqual(pm.smhi_weather_code_to_class(100), pm.CLASS_TO_ID["Inget"])
        self.assertEqual(pm.smhi_weather_code_to_class(120), pm.CLASS_TO_ID["Inget"])  # fog
        self.assertEqual(pm.smhi_weather_code_to_class(508), pm.CLASS_TO_ID["Inget"])
        self.assertEqual(pm.smhi_weather_code_to_class(151), pm.CLASS_TO_ID["Regn"])
        self.assertEqual(pm.smhi_weather_code_to_class(161), pm.CLASS_TO_ID["Regn"])
        self.assertEqual(pm.smhi_weather_code_to_class(123), pm.CLASS_TO_ID["Regn"])
        self.assertEqual(pm.smhi_weather_code_to_class(171), pm.CLASS_TO_ID["Sno"])
        self.assertEqual(pm.smhi_weather_code_to_class(124), pm.CLASS_TO_ID["Sno"])
        self.assertEqual(pm.smhi_weather_code_to_class(185), pm.CLASS_TO_ID["Sno"])
        self.assertEqual(pm.smhi_weather_code_to_class(167), pm.CLASS_TO_ID["Snoblandat"])
        self.assertEqual(pm.smhi_weather_code_to_class(68), pm.CLASS_TO_ID["Snoblandat"])

    def test_skips_missing_and_ambiguous_codes(self):
        self.assertIsNone(pm.smhi_weather_code_to_class(509))
        self.assertIsNone(pm.smhi_weather_code_to_class(510))
        self.assertIsNone(pm.smhi_weather_code_to_class(122))
        self.assertIsNone(pm.smhi_weather_code_to_class(140))


class ClassifyPrecipProbsTest(unittest.TestCase):
    def test_dry_hour_below_threshold_returns_inget(self):
        out = pm.classify_precip_probs([0.80, 0.15, 0.04, 0.01], precip_threshold=0.40)
        self.assertEqual(out["precip_label"], "Inget")
        self.assertAlmostEqual(out["p_precip"], 0.2, places=3)

    def test_clear_rain_and_snow_winners(self):
        rain = pm.classify_precip_probs([0.20, 0.72, 0.06, 0.02], precip_threshold=0.40)
        snow = pm.classify_precip_probs([0.15, 0.05, 0.78, 0.02], precip_threshold=0.40)
        self.assertEqual(rain["precip_label"], "Regn")
        self.assertEqual(snow["precip_label"], "Sno")

    def test_sleet_via_direct_probability_or_rain_snow_mix(self):
        direct = pm.classify_precip_probs([0.25, 0.35, 0.20, 0.20], precip_threshold=0.40, sleet_threshold=0.15)
        mixed = pm.classify_precip_probs([0.20, 0.42, 0.36, 0.02], precip_threshold=0.40, sleet_mix_ratio=0.30)
        self.assertEqual(direct["precip_label"], "Snoblandat")
        self.assertEqual(mixed["precip_label"], "Snoblandat")


class ThresholdAndDailyAggregationTest(unittest.TestCase):
    def test_choose_precip_threshold_separates_wet_and_dry(self):
        probs = np.array([
            [0.90, 0.08, 0.01, 0.01],
            [0.85, 0.10, 0.04, 0.01],
            [0.25, 0.70, 0.04, 0.01],
            [0.20, 0.05, 0.74, 0.01],
        ])
        y = np.array([0, 0, 1, 2])
        threshold, f1 = pm.choose_precip_threshold(probs, y)
        self.assertGreaterEqual(threshold, 0.20)
        self.assertLessEqual(threshold, 0.75)
        self.assertEqual(f1, 1.0)

    def test_daily_ignores_single_low_confidence_hour(self):
        hours = [
            {"precip_label": "Inget", "p_precip": 0.1, "p_rain": 0.05, "p_snow": 0.05, "p_sleet": 0.0},
            {"precip_label": "Regn", "p_precip": 0.42, "p_rain": 0.38, "p_snow": 0.03, "p_sleet": 0.01},
        ]
        out = pm.aggregate_daily_precip(hours, precip_threshold=0.40)
        self.assertEqual(out["precip_label"], "Inget")

    def test_daily_needs_a_quarter_of_daytime_hours_wet(self):
        dry = {"precip_label": "Inget", "p_precip": 0.1, "p_rain": 0.05, "p_snow": 0.05, "p_sleet": 0.0}
        wet = {"precip_label": "Regn", "p_precip": 0.6, "p_rain": 0.55, "p_snow": 0.03, "p_sleet": 0.02}
        two_of_twelve = [wet] * 2 + [dry] * 10
        three_of_twelve = [wet] * 3 + [dry] * 9
        self.assertEqual(pm.aggregate_daily_precip(two_of_twelve)["precip_label"], "Inget")
        self.assertEqual(pm.aggregate_daily_precip(three_of_twelve)["precip_label"], "Regn")

    def test_daily_ignores_single_strong_hour(self):
        strong = {"precip_label": "Regn", "p_precip": 0.95, "p_rain": 0.9, "p_snow": 0.03, "p_sleet": 0.02}
        dry = {"precip_label": "Inget", "p_precip": 0.1, "p_rain": 0.05, "p_snow": 0.05, "p_sleet": 0.0}
        out = pm.aggregate_daily_precip([strong] + [dry] * 3)
        self.assertEqual(out["precip_label"], "Inget")
        self.assertEqual(out["p_precip_max"], 0.95)

    def test_daily_triggers_on_wet_days(self):
        snow_day = [
            {"precip_label": "Sno", "p_precip": 0.48, "p_rain": 0.05, "p_snow": 0.42, "p_sleet": 0.01},
            {"precip_label": "Sno", "p_precip": 0.50, "p_rain": 0.04, "p_snow": 0.45, "p_sleet": 0.01},
        ]
        self.assertEqual(pm.aggregate_daily_precip(snow_day, precip_threshold=0.40)["precip_label"], "Sno")

        mixed_day = [
            {"precip_label": "Regn", "p_precip": 0.70, "p_rain": 0.65, "p_snow": 0.04, "p_sleet": 0.01},
            {"precip_label": "Sno", "p_precip": 0.65, "p_rain": 0.05, "p_snow": 0.58, "p_sleet": 0.02},
        ]
        self.assertEqual(pm.aggregate_daily_precip(mixed_day, precip_threshold=0.40)["precip_label"], "Snoblandat")


class PrecipDuelTest(unittest.TestCase):
    def setUp(self):
        from datetime import datetime, timedelta, timezone

        rng = np.random.default_rng(0)
        n = 120
        self.x = rng.normal(size=(n, len(pm.PRECIP_FEATURE_ORDER)))
        self.y = (self.x[:, 0] > 0).astype(int)  # class 0/1 driven by the first column
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.times = [start + timedelta(hours=i) for i in range(n)]
        self.meta = {"feature_order": list(pm.PRECIP_FEATURE_ORDER), "history_end": start.isoformat()}

    def _model(self, labels, rounds):
        import xgboost as xgb

        dm = xgb.DMatrix(self.x, label=labels, feature_names=pm.PRECIP_FEATURE_ORDER)
        return xgb.train(pm.PRECIP_XGB_PARAMS, dm, num_boost_round=rounds)

    def test_better_challenger_wins_on_unseen_hours(self):
        good = self._model(self.y, 30)
        bad = self._model(np.zeros_like(self.y), 5)
        out = pm.precip_duel(good, bad, self.meta, self.x, self.y, self.times, min_unseen_rows=48)
        self.assertTrue(out["save"])
        self.assertLess(out["challenger_logloss"], out["champion_logloss"])

        out = pm.precip_duel(bad, good, self.meta, self.x, self.y, self.times, min_unseen_rows=48)
        self.assertFalse(out["save"])

    def test_too_few_unseen_hours_keeps_live_model(self):
        meta = {**self.meta, "history_end": self.times[-10].isoformat()}
        model = self._model(self.y, 5)
        out = pm.precip_duel(model, model, meta, self.x, self.y, self.times, min_unseen_rows=48)
        self.assertFalse(out["save"])
        self.assertEqual(out["unseen_rows"], 9)

    def test_saves_without_comparable_live_model(self):
        model = self._model(self.y, 5)
        self.assertTrue(pm.precip_duel(model, None, None, self.x, self.y, self.times, 48)["save"])
        other_inputs = {**self.meta, "feature_order": ["cloud_cover"]}
        self.assertTrue(pm.precip_duel(model, model, other_inputs, self.x, self.y, self.times, 48)["save"])


class PrecipIntensityTest(unittest.TestCase):
    def test_snow_uses_snowfall_and_rain_uses_precipitation(self):
        self.assertEqual(pm.precip_intensity("Sno", {"snowfall": 0.1}), "Latt")
        self.assertEqual(pm.precip_intensity("Sno", {"snowfall": 0.5}), "Mattlig")
        self.assertEqual(pm.precip_intensity("Sno", {"snowfall": 1.5}), "Kraftig")
        self.assertEqual(pm.precip_intensity("Regn", {"precipitation": 0.2}), "Latt")
        self.assertEqual(pm.precip_intensity("Regn", {"precipitation": 2.0}), "Mattlig")
        self.assertEqual(pm.precip_intensity("Snoblandat", {"precipitation": 5.0}), "Kraftig")

    def test_dry_hour_or_missing_amount(self):
        self.assertIsNone(pm.precip_intensity("Inget", {"precipitation": 3.0}))
        self.assertEqual(pm.precip_intensity("Sno", {"snowfall": None}), "Latt")

    def test_daily_intensity_needs_two_hours_at_that_level(self):
        def hour(intensity):
            return {"precip_label": "Sno", "precip_intensity": intensity, "p_precip": 0.8,
                    "p_rain": 0.05, "p_snow": 0.7, "p_sleet": 0.05}

        one_heavy = [hour("Kraftig"), hour("Latt"), hour("Latt")]
        two_heavy = [hour("Kraftig"), hour("Kraftig"), hour("Latt")]
        mixed = [hour("Kraftig"), hour("Mattlig"), hour("Latt")]
        self.assertEqual(pm.aggregate_daily_precip(one_heavy)["precip_intensity"], "Latt")
        self.assertEqual(pm.aggregate_daily_precip(two_heavy)["precip_intensity"], "Kraftig")
        self.assertEqual(pm.aggregate_daily_precip(mixed)["precip_intensity"], "Mattlig")


class TrainPrecipModelSplitTest(unittest.TestCase):
    def test_duel_rows_are_the_newest_tenth_and_untouched_by_tuning(self):
        n = 1000
        rng = np.random.default_rng(1)
        from datetime import datetime, timedelta

        start = datetime(2026, 1, 1)
        times = [(start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(n)]
        names = pm.BASE_WEATHER_FEATURES + pm.PRECIP_EXTRA_HOURLY
        weather = {name: rng.uniform(0, 100, n) for name in names}
        obs = {t: int(weather["precipitation"][i] > 70) for i, t in enumerate(times)}

        _, meta, (x_hold, y_hold, hold_times) = pm.train_precip_model(times, weather, obs, rounds=10)

        self.assertEqual(len(x_hold), 100)
        self.assertEqual(len(y_hold), len(hold_times))
        self.assertEqual(hold_times[0].strftime("%Y-%m-%dT%H:%M"), times[900])
        self.assertEqual(meta["metrics"]["tune_rows"], 100)
        self.assertEqual(meta["metrics"]["val_rows"], 100)


class AppServerPrecipIntegrationTest(unittest.TestCase):
    def test_enrich_rows_sets_display_label_without_overwriting_sky_label(self):
        import app_server

        row = {
            "time_utc": "2026-11-20T12:00:00Z",
            "label": "Mulet",
            "cloud_cover": 100.0,
            "precipitation": 0.4,
            "temp_c": -1.2,
            "wind_ms": 4.0,
            "pred_cloud": 98.0,
            "sun_index": 0.02,
            "is_night": False,
            "precip_label": "Sno",
            "p_precip": 0.81,
        }
        meta = {"thresholds": {"sunny_idx": 0.65, "mixed_idx": 0.35}}
        with mock.patch.object(app_server, "load_model", return_value=(None, meta)):
            enriched = app_server.enrich_rows([row])

        self.assertEqual(enriched[0]["label"], "Mulet")
        self.assertEqual(enriched[0]["precip_label"], "Sno")
        self.assertEqual(enriched[0]["display_label"], "Sno")

    def test_stale_cache_row_without_precip_falls_back_to_sky_label(self):
        import app_server

        row = {
            "time_utc": "2026-11-20T12:00:00Z",
            "label": "Soligt",
            "cloud_cover": 5.0,
            "precipitation": 0.0,
            "temp_c": 2.0,
            "wind_ms": 2.0,
            "pred_cloud": 10.0,
            "sun_index": 0.90,
            "is_night": False,
        }
        meta = {"thresholds": {"sunny_idx": 0.65, "mixed_idx": 0.35}}
        with mock.patch.object(app_server, "load_model", return_value=(None, meta)):
            enriched = app_server.enrich_rows([row])

        self.assertEqual(enriched[0]["label"], "Soligt")
        self.assertIsNone(enriched[0]["precip_label"])
        self.assertEqual(enriched[0]["display_label"], "Soligt")


if __name__ == "__main__":
    unittest.main()
