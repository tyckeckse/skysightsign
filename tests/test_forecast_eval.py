from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import forecast_eval


class ScoreTest(unittest.TestCase):
    NOW = datetime(2026, 10, 25, 18, 0, tzinfo=timezone.utc)
    OBS = {"2026-10-25T06:00": 70.0, "2026-10-25T08:00": 80.0}

    def row(self, issued, **values):
        base = {"time_utc": "2026-10-25T08:00:00Z", "pred_cloud": 60.0, "om_cloud": 100.0, "smhi_cloud": 75.0}
        return {"issued_at_utc": issued, **base, **values}

    def score(self, log, obs=None, **kwargs):
        return forecast_eval.score(log, obs or self.OBS, days=30, now=self.NOW, **kwargs)

    def test_scores_the_locked_hour(self):
        result = self.score([self.row("2026-10-25T07:01:00Z")])
        overall = result["overall"]
        self.assertEqual(overall["n"], 1)
        self.assertEqual((overall["model_mae"], overall["om_mae"], overall["smhi_mae"]), (20.0, 20.0, 5.0))
        self.assertEqual((overall["model_bias"], overall["om_bias"]), (-20.0, 20.0))
        self.assertEqual(result["by_day"][0]["hours"][0]["model"], 60.0)

    def test_same_sky_uses_the_hour_before_the_run_hour(self):
        result = self.score([self.row("2026-10-25T07:01:00Z")], {**self.OBS, "2026-10-25T07:00": 0.0})
        self.assertEqual(result["by_day"][0]["hours"][0]["same"], 70.0)

    def test_first_run_in_the_hour_locks(self):
        log = [self.row("2026-10-25T07:40:00Z", pred_cloud=10.0), self.row("2026-10-25T07:01:00Z")]
        result = self.score(log)
        self.assertEqual(result["overall"]["n"], 1)
        self.assertEqual(result["overall"]["model_mae"], 20.0)

    def test_incomplete_first_run_is_skipped(self):
        log = [self.row("2026-10-25T07:01:00Z", smhi_cloud=None), self.row("2026-10-25T07:40:00Z", pred_cloud=70.0)]
        self.assertEqual(self.score(log)["overall"]["model_mae"], 10.0)

    def test_older_and_too_late_runs_do_not_fill_in(self):
        log = [self.row("2026-10-25T06:59:00Z"), self.row("2026-10-25T08:01:00Z")]
        self.assertEqual(self.score(log)["overall"]["n"], 0)

    def test_windows_are_kept_apart(self):
        obs = {**self.OBS, "2026-10-25T01:00": 50.0, "2026-10-24T07:00": 40.0}
        log = [self.row("2026-10-25T07:01:00Z"), self.row("2026-10-25T02:01:00Z"), self.row("2026-10-24T08:01:00Z")]
        windows = {w["window_h"]: w for w in self.score(log, obs)["by_window"]}
        self.assertEqual({h: w["n"] for h, w in windows.items()}, {1: 1, 6: 1, 12: 0, 24: 1})
        self.assertEqual(windows[6]["same_mae"], 30.0)
        self.assertIsNone(windows[12]["model_mae"])

    def test_skips_forecasts_issued_before_issued_from(self):
        start = datetime(2026, 10, 25, 7, 30, tzinfo=timezone.utc)
        self.assertEqual(self.score([self.row("2026-10-25T07:01:00Z")], issued_from=start)["overall"]["n"], 0)

    def test_splits_by_observed_sky(self):
        sky = {s["sky"]: s["n"] for s in self.score([self.row("2026-10-25T07:01:00Z")])["by_sky"]}
        self.assertEqual(sky, {"clear": 0, "mixed": 1, "overcast": 0})


class DropDuplicateRowsTest(unittest.TestCase):
    def test_keeps_one_row_per_issue_minute_and_target_hour(self):
        rows = [
            {"issued_at_utc": "2026-11-24T10:00:06Z", "time_utc": "2026-11-24T11:00:00Z"},
            {"issued_at_utc": "2026-11-24T10:00:07Z", "time_utc": "2026-11-24T11:00:00Z"},
            {"issued_at_utc": "2026-11-24T10:00:06Z", "time_utc": "2026-11-24T12:00:00Z"},
            {"issued_at_utc": "2026-11-24T11:00:05Z", "time_utc": "2026-11-24T12:00:00Z"},
        ]
        self.assertEqual(forecast_eval.drop_duplicate_rows(rows), [rows[0], rows[2], rows[3]])


class DevilLiveFromTest(unittest.TestCase):
    def test_first_row_with_devil_probability(self):
        log = [
            {"issued_at_utc": "2026-11-20T06:00:00Z", "p_overcast": None},
            {"issued_at_utc": "2026-11-21T07:00:00Z", "p_overcast": 0.4},
            {"issued_at_utc": "2026-11-21T06:00:00Z", "p_overcast": 0.1},
            {"issued_at_utc": "2026-11-19T06:00:00Z"},
        ]
        self.assertEqual(forecast_eval.devil_live_from(log), datetime(2026, 11, 21, 6, 0, tzinfo=timezone.utc))

    def test_none_before_devil_is_live(self):
        self.assertIsNone(forecast_eval.devil_live_from([{"issued_at_utc": "2026-11-20T06:00:00Z"}]))


class PrecipScoreTest(unittest.TestCase):
    def test_scores_precip_accuracy_and_f1(self):
        now = datetime(2026, 11, 20, 18, 0, tzinfo=timezone.utc)
        log = [
            {"issued_at_utc": "2026-11-20T12:00:00Z", "time_utc": "2026-11-20T14:00:00Z", "pred_precip": "Sno", "om_precip": "Inget", "smhi_precip": "Sno"},
            {"issued_at_utc": "2026-11-20T12:00:00Z", "time_utc": "2026-11-20T15:00:00Z", "pred_precip": "Inget", "om_precip": "Inget", "smhi_precip": "Inget"},
            {"issued_at_utc": "2026-11-20T12:00:00Z", "time_utc": "2026-11-20T16:00:00Z", "pred_precip": "Regn", "om_precip": "Regn", "smhi_precip": "Inget"},
        ]
        obs = {
            "2026-11-20T14:00": "Sno",
            "2026-11-20T15:00": "Inget",
            "2026-11-20T16:00": "Regn",
        }
        result = forecast_eval.score_precip(log, obs, days=7, now=now)
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["unique_hours"], 3)
        self.assertEqual(result["model_acc"], 100.0)
        self.assertEqual(result["model_f1"], 100.0)
        self.assertEqual(result["om_acc"], 66.7)
        self.assertEqual(result["om_f1"], 66.7)
        self.assertEqual(result["smhi_acc"], 66.7)
        self.assertEqual(result["smhi_f1"], 66.7)
        self.assertEqual(len(result["by_day"]), 1)
        self.assertEqual(len(result["by_day"][0]["hours"]), 3)
        self.assertEqual(result["by_day"][0]["hours"][0]["obs"], "Sno")
        self.assertEqual(result["by_day"][0]["hours"][0]["om"], "Inget")

    def test_skips_rows_missing_any_precip_source(self):
        now = datetime(2026, 11, 20, 18, 0, tzinfo=timezone.utc)
        log = [{"issued_at_utc": "2026-11-20T12:00:00Z", "time_utc": "2026-11-20T14:00:00Z", "pred_precip": "Sno", "om_precip": None, "smhi_precip": "Sno"}]
        result = forecast_eval.score_precip(log, {"2026-11-20T14:00": "Sno"}, days=7, now=now)
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["model_acc"])

    def test_smhi_symbol_to_precip_mapping(self):
        import update_forecast_cache as ufc

        self.assertEqual(ufc.smhi_symbol_to_precip(1), "Inget")
        self.assertEqual(ufc.smhi_symbol_to_precip(6), "Inget")
        self.assertEqual(ufc.smhi_symbol_to_precip(18), "Regn")
        self.assertEqual(ufc.smhi_symbol_to_precip(13), "Snoblandat")
        self.assertEqual(ufc.smhi_symbol_to_precip(26), "Sno")
        self.assertIsNone(ufc.smhi_symbol_to_precip(None))


if __name__ == "__main__":
    unittest.main()
