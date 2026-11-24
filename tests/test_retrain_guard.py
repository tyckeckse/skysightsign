from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import xgboost as xgb

import train_model
import update_forecast_cache


class CloudEvaluationSplitTest(unittest.TestCase):
    def test_promotion_holdout_contains_only_rows_after_tuning(self):
        rows = list(range(501))
        tune, holdout = train_model.cloud_evaluation_slices(400, len(rows) - 400)

        self.assertEqual(rows[tune], list(range(400, 450)))
        self.assertEqual(rows[holdout], list(range(450, 501)))


class LoadChampionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "model.ubj"

    def tearDown(self):
        self.tmp.cleanup()

    def _save_model(self, names: list[str]):
        x = np.arange(20.0 * len(names)).reshape(20, len(names))
        data = xgb.DMatrix(x, label=x[:, 0], feature_names=names)
        xgb.train({"max_depth": 2}, data, num_boost_round=2).save_model(self.path)

    def test_missing_file_means_no_champion(self):
        self.assertIsNone(train_model.load_champion(self.path))

    def test_loads_model_with_known_features(self):
        names = train_model.FEATURE_ORDER[:2]
        self._save_model(names)
        self.assertEqual(train_model.load_champion(self.path).feature_names, names)

    def test_model_with_unknown_features_is_not_a_champion(self):
        self._save_model(["unknown_a", "unknown_b"])
        self.assertIsNone(train_model.load_champion(self.path))

    def test_unreadable_file_raises_instead_of_forcing_promotion(self):
        self.path.write_bytes(b"broken")
        with self.assertRaises(xgb.core.XGBoostError):
            train_model.load_champion(self.path)


class TrainingOverdueTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "train_state.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, last_train_utc: str):
        self.state.write_text(json.dumps({"last_train_utc": last_train_utc}), encoding="utf-8")

    def test_recent_training_is_not_overdue(self):
        self._write("2026-11-14T11:00:00Z")
        now = datetime(2026, 11, 15, 11, 30, tzinfo=timezone.utc)
        self.assertFalse(update_forecast_cache.training_overdue(self.state, now))

    def test_old_training_is_overdue(self):
        self._write("2026-11-13T15:56:36Z")
        now = datetime(2026, 11, 15, 7, 0, tzinfo=timezone.utc)
        self.assertTrue(update_forecast_cache.training_overdue(self.state, now))

    def test_missing_state_is_overdue(self):
        self.assertTrue(update_forecast_cache.training_overdue(self.state))


class CatchUpIntervalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "retrain_catch_up.log"

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_attempt_is_allowed(self):
        self.assertTrue(update_forecast_cache.catch_up_allowed(self.log))

    def test_waits_six_hours_after_an_attempt(self):
        update_forecast_cache.note_catch_up(self.log)
        attempted = datetime.fromtimestamp(self.log.stat().st_mtime, tz=timezone.utc)
        hour = update_forecast_cache.timedelta(hours=1)
        self.assertFalse(update_forecast_cache.catch_up_allowed(self.log, attempted + 5 * hour))
        self.assertTrue(update_forecast_cache.catch_up_allowed(self.log, attempted + 6 * hour))


class TrainLockTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lock = Path(self.tmp.name) / "train.lock"

    def tearDown(self):
        self.tmp.cleanup()

    def test_second_run_is_blocked_and_lock_is_released(self):
        with train_model.train_lock(self.lock) as first:
            with train_model.train_lock(self.lock) as second:
                self.assertTrue(first)
                self.assertFalse(second)
        self.assertFalse(self.lock.exists())

    def test_stale_lock_is_taken_over(self):
        self.lock.touch()
        old = time.time() - train_model.STALE_LOCK_SECONDS - 60
        os.utime(self.lock, (old, old))
        with train_model.train_lock(self.lock) as acquired:
            self.assertTrue(acquired)


if __name__ == "__main__":
    unittest.main()
