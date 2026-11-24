from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import model_common


class ModelCommonTests(unittest.TestCase):
    def test_time_features_noon_utc(self) -> None:
        dt = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
        tf = model_common.time_features(dt)
        self.assertAlmostEqual(tf["hour_cos"], -1.0, places=6)
        self.assertAlmostEqual(tf["hour_sin"], 0.0, places=6)

    def test_save_and_load_meta_json_replaces_legacy_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "uppsala_solar_meta.json"
            pkl_path = json_path.with_suffix(".pkl")
            with pkl_path.open("wb") as f:
                pickle.dump({"version": 1, "from": "pickle"}, f)

            # Falls back to .pkl when .json does not exist yet.
            self.assertEqual(model_common.load_meta(json_path), {"version": 1, "from": "pickle"})

            # Saving JSON writes .json and removes the legacy .pkl file.
            model_common.save_meta(json_path, {"version": 2, "from": "json"})
            self.assertTrue(json_path.exists())
            self.assertFalse(pkl_path.exists())
            self.assertEqual(model_common.load_meta(json_path), {"version": 2, "from": "json"})


if __name__ == "__main__":
    unittest.main()
