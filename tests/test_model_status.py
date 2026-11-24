from __future__ import annotations

import json
import pickle
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from model_status import build_model_status


def _create_mlflow_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE experiments (experiment_id INTEGER PRIMARY KEY, name TEXT)")
    cur.execute(
        """
        CREATE TABLE runs (
            run_uuid TEXT PRIMARY KEY,
            name TEXT,
            status TEXT,
            start_time INTEGER,
            end_time INTEGER,
            experiment_id INTEGER,
            lifecycle_stage TEXT
        )
        """
    )
    cur.execute("CREATE TABLE metrics (run_uuid TEXT, key TEXT, value REAL)")
    cur.execute("CREATE TABLE tags (run_uuid TEXT, key TEXT, value TEXT)")

    cur.execute("INSERT INTO experiments VALUES (1, 'uppsala-solar')")
    cur.execute("INSERT INTO experiments VALUES (2, 'other-exp')")

    cur.execute(
        "INSERT INTO runs VALUES ('r1', 'first-run', 'FINISHED', 1700000000000, 1700000020500, 1, 'active')"
    )
    cur.execute(
        "INSERT INTO runs VALUES ('r2', 'second-run', 'FINISHED', 1700003600000, 1700003615000, 1, 'active')"
    )
    cur.execute(
        "INSERT INTO runs VALUES ('r3', 'deleted-run', 'FINISHED', 1700007200000, 1700007210000, 1, 'deleted')"
    )
    cur.execute(
        "INSERT INTO runs VALUES ('r4', 'other-exp-run', 'FINISHED', 1700007200000, 1700007210000, 2, 'active')"
    )

    cur.executemany(
        "INSERT INTO metrics VALUES (?, ?, ?)",
        [
            ("r1", "val_mae", 17.4),
            ("r1", "val_mae_with_devil", 16.85),
            ("r1", "precip_val_accuracy", 0.901),
            ("r1", "precip_val_f1", 0.682),
            ("r2", "val_mae", 17.6),
            ("r2", "val_mae_with_devil", 17.05),
            ("r2", "unseen_rows", 15.0),
        ],
    )
    cur.executemany(
        "INSERT INTO tags VALUES (?, ?, ?)",
        [
            ("r1", "promoted", "true"),
            ("r1", "precip_saved", "true"),
            ("r1", "objective", "reg:absoluteerror"),
            ("r1", "mlflow.user", "ignored"),
            ("r2", "promoted", "false"),
            ("r2", "precip_saved", "false"),
        ],
    )
    conn.commit()
    conn.close()


class ModelStatusTests(unittest.TestCase):
    def test_build_model_status_reads_meta_state_and_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "uppsala_overcast_devil.ubj").write_bytes(b"stub")

            with (model_dir / "uppsala_solar_meta.pkl").open("wb") as f:
                pickle.dump(
                    {
                        "version": 2,
                        "trained_at_utc": "2026-11-20T13:09:21+00:00",
                        "history_start": "2023-01-19T09:00:00+00:00",
                        "history_end": "2026-11-19T14:00:00+00:00",
                        "feature_order": ["cloud_cover", "hour_sin"],
                        "thresholds": {"sunny_idx": 0.83, "mixed_idx": 0.41, "source": "feedback_calibrated"},
                        "devil": {"threshold": 0.4, "val_mae_with": 16.858},
                        "metrics": {
                            "objective": "reg:absoluteerror",
                            "train_rows": 13400,
                            "val_rows": 3351,
                            "best_iteration": 99,
                            "mae": 17.396,
                            "rmse": 26.86,
                            "r2": 0.5969,
                        },
                    },
                    f,
                )

            with (model_dir / "uppsala_precip_meta.pkl").open("wb") as f:
                pickle.dump(
                    {
                        "version": 1,
                        "trained_at_utc": "2026-11-20T13:09:29+00:00",
                        "feature_order": ["cloud_cover", "rain", "snowfall"],
                        "classes": ["Inget", "Regn", "Sno", "Snoblandat"],
                        "precip_threshold": 0.35,
                        "sleet_threshold": 0.15,
                        "metrics": {
                            "train_rows": 25766,
                            "val_rows": 6442,
                            "best_iteration": 130,
                            "val_accuracy": 0.8991,
                            "val_precip_f1": 0.6754,
                            "val_mlogloss": 0.22918,
                        },
                    },
                    f,
                )

            train_state = root / "train_state.json"
            train_state.write_text(
                json.dumps({"last_train_utc": "2026-11-22T15:05:22Z", "history_days": 1400}),
                encoding="utf-8",
            )

            mlflow_db = root / "mlflow.db"
            _create_mlflow_db(mlflow_db)

            payload = build_model_status(model_dir, mlflow_db, train_state, limit=10)

            self.assertEqual(payload["total_runs"], 2)
            self.assertEqual(len(payload["runs"]), 2)
            self.assertEqual(payload["runs"][0]["name"], "second-run")
            self.assertEqual(payload["runs"][0]["duration_s"], 15.0)
            self.assertFalse(payload["runs"][0]["promoted"])
            self.assertEqual(payload["runs"][0]["unseen_rows"], 15)

            self.assertEqual(payload["runs"][1]["name"], "first-run")
            self.assertTrue(payload["runs"][1]["promoted"])
            self.assertEqual(payload["runs"][1]["precip_val_accuracy"], 90.1)
            self.assertEqual(payload["runs"][1]["precip_val_f1"], 68.2)

            solar = payload["active_models"]["solar"]
            self.assertEqual(solar["feature_count"], 2)
            self.assertEqual(solar["val_mae"], 17.4)
            self.assertEqual(solar["run_name"], "first-run")
            self.assertTrue(solar["devil"]["active"])
            self.assertEqual(solar["devil"]["val_mae_with"], 16.86)

            precip = payload["active_models"]["precip"]
            self.assertEqual(precip["val_accuracy"], 89.9)
            self.assertEqual(precip["val_f1"], 67.5)
            self.assertEqual(precip["run_name"], "first-run")
            self.assertEqual(payload["train_state"]["history_days"], 1400)

    def test_handles_missing_files_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = build_model_status(
                root / "missing_model",
                root / "missing.db",
                root / "missing_state.json",
            )
            self.assertIsNone(payload["active_models"]["solar"])
            self.assertIsNone(payload["active_models"]["precip"])
            self.assertEqual(payload["runs"], [])
            self.assertEqual(payload["total_runs"], 0)


if __name__ == "__main__":
    unittest.main()
