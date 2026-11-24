from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import forecast_eval


class ScoreTest(unittest.TestCase):
    def test_skips_hours_without_smhi_forecast(self):
        now = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
        log = [{"issued_at_utc": "2026-10-25T06:00:00Z", "time_utc": "2026-10-25T08:00:00Z", "pred_cloud": 60.0, "om_cloud": 100.0, "smhi_cloud": None}]
        result = forecast_eval.score(log, {"2026-10-25T08:00": 80.0}, days=14, now=now)
        self.assertEqual(result["overall"]["n"], 0)

    def test_scores_model_and_open_meteo_against_obs(self):
        now = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
        log = [
            {"issued_at_utc": "2026-10-25T06:00:00Z", "time_utc": "2026-10-25T08:00:00Z", "pred_cloud": 60.0, "om_cloud": 100.0, "smhi_cloud": 75.0},
            {"issued_at_utc": "2026-10-25T06:00:00Z", "time_utc": "2026-10-25T09:00:00Z", "pred_cloud": 50.0, "om_cloud": 50.0, "smhi_cloud": 50.0},
            {"issued_at_utc": "2026-10-25T06:00:00Z", "time_utc": "2026-10-25T20:00:00Z", "pred_cloud": 0.0, "om_cloud": 0.0, "smhi_cloud": 0.0},
        ]
        obs = {"2026-10-25T08:00": 80.0, "2026-10-25T09:00": 50.0}
        result = forecast_eval.score(log, obs, days=14, now=now)
        self.assertEqual(result["overall"]["n"], 2)
        self.assertEqual(result["overall"]["model_mae"], 10.0)
        self.assertEqual(result["overall"]["om_mae"], 10.0)
        self.assertEqual(result["overall"]["model_bias"], -10.0)
        self.assertEqual(result["overall"]["om_bias"], 10.0)
        self.assertEqual(result["overall"]["smhi_mae"], 2.5)
        self.assertEqual(result["by_lead"][0]["n"], 2)
        self.assertEqual(len(result["by_day"]), 1)

    def test_skips_forecasts_issued_before_issued_from(self):
        now = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
        row = {"time_utc": "2026-10-25T09:00:00Z", "pred_cloud": 50.0, "om_cloud": 50.0, "smhi_cloud": 50.0}
        log = [{**row, "issued_at_utc": "2026-10-25T05:00:00Z"}, {**row, "issued_at_utc": "2026-10-25T07:00:00Z"}]
        issued_from = datetime(2026, 10, 25, 6, 0, tzinfo=timezone.utc)
        result = forecast_eval.score(log, {"2026-10-25T09:00": 50.0}, days=14, now=now, issued_from=issued_from)
        self.assertEqual(result["overall"]["n"], 1)

    def test_skips_forecast_made_after_the_hour_began(self):
        now = datetime(2026, 11, 20, 18, 0, tzinfo=timezone.utc)
        row = {"time_utc": "2026-11-20T13:00:00Z", "pred_cloud": 100.0, "om_cloud": 100.0, "smhi_cloud": 100.0}
        log = [{**row, "issued_at_utc": "2026-11-20T13:10:01Z"}, {**row, "issued_at_utc": "2026-11-20T12:00:05Z"}]
        result = forecast_eval.score(log, {"2026-11-20T13:00": 100.0}, days=14, now=now)
        self.assertEqual(result["overall"]["n"], 1)
        self.assertEqual(result["by_day"][0]["hours"][0]["lead_h"], 1.0)

    def test_by_day_includes_latest_forecast_per_hour(self):
        now = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
        row = {"time_utc": "2026-10-25T08:00:00Z", "om_cloud": 80.0, "smhi_cloud": 80.0}
        log = [
            {**row, "issued_at_utc": "2026-10-25T05:00:00Z", "pred_cloud": 60.0},
            {**row, "issued_at_utc": "2026-10-25T07:00:00Z", "pred_cloud": 70.0},
        ]
        result = forecast_eval.score(log, {"2026-10-25T08:00": 80.0}, days=14, now=now)
        hours = result["by_day"][0]["hours"]
        self.assertEqual(len(hours), 1)
        self.assertEqual(hours[0]["time_utc"], "2026-10-25T08:00:00Z")
        self.assertEqual(hours[0]["obs"], 80.0)
        self.assertEqual(hours[0]["lead_h"], 1.0)
        self.assertEqual(hours[0]["model"], 70.0)
        self.assertEqual(hours[0]["model_err"], -10.0)
        self.assertEqual(hours[0]["om_err"], 0.0)


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
