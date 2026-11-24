from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import overcast_devil as devil


def _response(hourly: dict) -> mock.Mock:
    response = mock.Mock()
    response.json.return_value = {"hourly": hourly}
    response.raise_for_status.return_value = None
    return response


class AddExtraCloudTest(unittest.TestCase):
    def test_aligns_on_time_and_fills_gaps_with_none(self):
        hourly = {"time": ["T1", "T2"]}
        extra = {"time": ["T2"], **{f"cloud_cover_{m}": [80] for m in devil.EXTRA_MODELS}}
        with mock.patch.object(devil.requests, "get", return_value=_response(extra)) as get:
            devil.add_extra_cloud("url", {"hourly": "x", "daily": "y", "latitude": "1"}, hourly)
        self.assertEqual(hourly["extra_cloud_dmi_seamless"], [None, 80])
        sent = get.call_args.kwargs["params"]
        self.assertEqual(sent["models"], ",".join(devil.EXTRA_MODELS))
        self.assertNotIn("daily", sent)

    def test_failed_request_gives_none(self):
        hourly = {"time": ["T1"]}
        with mock.patch.object(devil.requests, "get", side_effect=requests.ConnectionError):
            devil.add_extra_cloud("url", {}, hourly)
        self.assertEqual(hourly["extra_cloud_gfs_seamless"], [None])


class DevilMatrixTest(unittest.TestCase):
    def test_counts_sources_that_say_overcast_and_ignores_missing(self):
        order = ["cloud_cover", "icon_cloud_cover", "hour_sin"]
        x = np.array([[100.0, 96.0, 0.5]])
        extra = np.array([[99.0, 20.0, np.nan, 95.0, np.nan]])
        features, names = devil.devil_matrix(x, order, extra, np.array([70.0]))
        row = dict(zip(names, features[0]))
        self.assertEqual(row["n_sources_overcast"], 4)  # 100, 96, 99, 95
        self.assertEqual(row["n_sources"], 5)  # two extras missing
        self.assertEqual(row["main_pred"], 70.0)


class ThresholdTest(unittest.TestCase):
    def test_pulls_only_at_or_above_threshold(self):
        out = devil.apply_devil(np.array([70.0, 70.0]), np.array([0.6, 0.2]), 0.35)
        self.assertEqual(out.tolist(), [100.0, 70.0])

    def test_none_threshold_leaves_main_model_alone(self):
        out = devil.apply_devil(np.array([70.0]), np.array([0.99]), None)
        self.assertEqual(out.tolist(), [70.0])

    def test_chooses_threshold_that_lowers_error(self):
        # Overcast hours get p=0.6, a clear hour p=0.1: any threshold in (0.1, 0.6] fixes the overcast hours.
        base = np.array([70.0, 70.0, 10.0])
        p = np.array([0.6, 0.6, 0.1])
        y = np.array([100.0, 100.0, 0.0])
        threshold, without, with_devil = devil.choose_threshold(base, p, y)
        self.assertIsNotNone(threshold)
        self.assertTrue(0.1 < threshold <= 0.6)
        self.assertLess(with_devil, without)

    def test_stays_quiet_when_pulling_never_helps(self):
        base = np.array([70.0, 10.0])
        p = np.array([0.9, 0.9])
        y = np.array([70.0, 10.0])
        threshold, without, with_devil = devil.choose_threshold(base, p, y)
        self.assertIsNone(threshold)
        self.assertEqual(without, with_devil)


class AppPredictionTest(unittest.TestCase):
    def test_devil_pulls_live_forecast_and_keeps_main_number(self):
        import app_server

        booster = mock.Mock()
        booster.predict.return_value = np.array([70.0])
        meta = {"feature_order": ["cloud_cover"], "devil": {"threshold": 0.4}, "_devil_booster": object(),
                "thresholds": {"sunny_idx": 0.83, "mixed_idx": 0.2}}
        with mock.patch.object(app_server, "overcast_probability", return_value=np.array([0.7])):
            out = app_server.predict_hourly_sun(booster, meta, 0, {"cloud_cover": 100.0}, True)
        self.assertEqual(out["pred_cloud"], 100.0)
        self.assertEqual(out["main_cloud"], 70.0)
        self.assertEqual(out["sun_label"], "Mulet")

    def test_devil_stays_quiet_beyond_its_horizon(self):
        import app_server

        booster = mock.Mock()
        booster.predict.return_value = np.array([70.0])
        meta = {"feature_order": ["cloud_cover"], "devil": {"threshold": 0.4}, "_devil_booster": object(),
                "thresholds": {"sunny_idx": 0.83, "mixed_idx": 0.2}}
        hour_ms = 3_600_000
        with mock.patch.object(app_server, "overcast_probability", return_value=np.array([0.7])):
            near = app_server.predict_hourly_sun(booster, meta, 48 * hour_ms, {"cloud_cover": 100.0}, True, now_ms=0)
            far = app_server.predict_hourly_sun(booster, meta, 49 * hour_ms, {"cloud_cover": 100.0}, True, now_ms=0)
        self.assertEqual(near["pred_cloud"], 100.0)
        self.assertEqual(far["pred_cloud"], 70.0)
        self.assertIsNone(far["p_overcast"])

    def test_without_devil_main_number_is_shown(self):
        import app_server

        booster = mock.Mock()
        booster.predict.return_value = np.array([70.0])
        out = app_server.predict_hourly_sun(booster, {"feature_order": ["cloud_cover"]}, 0, {"cloud_cover": 100.0}, True)
        self.assertEqual(out["pred_cloud"], 70.0)
        self.assertIsNone(out["p_overcast"])


if __name__ == "__main__":
    unittest.main()
