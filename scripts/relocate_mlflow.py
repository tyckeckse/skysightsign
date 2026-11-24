"""Point MLflow's artifact folders at this checkout, so a database from another machine can be used here."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def relocate(db_path: Path, artifact_dir: Path) -> int:
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    with closing(sqlite3.connect(db_path)) as con, con:
        return con.execute("UPDATE experiments SET artifact_location = ?", (artifact_dir.resolve().as_uri(),)).rowcount


if __name__ == "__main__":
    print(f"Relocated {relocate(ROOT_DIR / 'mlflow.db', ROOT_DIR / 'mlruns')} experiments")
