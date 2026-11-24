from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import build_static
import relocate_mlflow


class StaticPageTest(unittest.TestCase):
    def test_marks_the_real_page_as_static(self):
        html = (ROOT_DIR / "index.html").read_text(encoding="utf-8")
        self.assertIn('data-static-api="1"', build_static.static_page(html))

    def test_fails_if_the_html_tag_changes(self):
        with self.assertRaises(ValueError):
            build_static.static_page("<html>")


class RelocateTest(unittest.TestCase):
    def test_replaces_artifact_locations(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "mlflow.db"
            with closing(sqlite3.connect(db)) as con, con:
                con.execute("CREATE TABLE experiments (artifact_location TEXT)")
                con.execute("INSERT INTO experiments VALUES ('file:///C:/old')")
            self.assertEqual(relocate_mlflow.relocate(db, Path(tmp)), 1)
            with closing(sqlite3.connect(db)) as con:
                location = con.execute("SELECT artifact_location FROM experiments").fetchone()[0]
            self.assertEqual(location, Path(tmp).resolve().as_uri())


if __name__ == "__main__":
    unittest.main()
