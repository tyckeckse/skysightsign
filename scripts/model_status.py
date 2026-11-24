from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_common import DEVIL_MODEL_FILE, PRECIP_META_FILE, SOLAR_META_FILE, load_meta


EXPERIMENT_NAME = "uppsala-solar"


def _ms_to_utc_iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _round_opt(value: Any, digits: int = 2, scale: float = 1.0) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    num = float(value)
    if not math.isfinite(num):
        return None
    return round(num * scale, digits)


def _int_opt(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return int(round(float(value)))
    return None


def _parse_bool_tag(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _load_train_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"last_train_utc": None, "history_days": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"last_train_utc": None, "history_days": None}
    if not isinstance(data, dict):
        return {"last_train_utc": None, "history_days": None}
    return {
        "last_train_utc": data.get("last_train_utc"),
        "history_days": _int_opt(data.get("history_days")),
    }


def _summarize_solar_meta(meta: dict[str, Any] | None, devil_exists: bool) -> dict[str, Any] | None:
    if meta is None:
        return None
    metrics = meta.get("metrics") or {}
    devil = meta.get("devil") or {}
    thresholds = meta.get("thresholds") or {}
    features = list(meta.get("feature_order") or [])
    return {
        "version": meta.get("version"),
        "trained_at_utc": meta.get("trained_at_utc"),
        "history_start": meta.get("history_start"),
        "history_end": meta.get("history_end"),
        "objective": metrics.get("objective"),
        "feature_count": len(features),
        "features": features,
        "train_rows": _int_opt(metrics.get("train_rows")),
        "val_rows": _int_opt(metrics.get("val_rows")),
        "best_iteration": _int_opt(metrics.get("best_iteration")),
        "val_mae": _round_opt(metrics.get("mae"), 2),
        "val_rmse": _round_opt(metrics.get("rmse"), 2),
        "val_r2": _round_opt(metrics.get("r2"), 3),
        "thresholds": {
            "sunny_idx": _round_opt(thresholds.get("sunny_idx"), 2),
            "mixed_idx": _round_opt(thresholds.get("mixed_idx"), 2),
            "source": thresholds.get("source"),
        },
        "devil": {
            "active": bool(devil) and devil_exists,
            "threshold": _round_opt(devil.get("threshold"), 2),
            "val_mae_with": _round_opt(devil.get("val_mae_with"), 2),
            "tune_mae_without": _round_opt(devil.get("tune_mae_without"), 2),
            "tune_mae_with": _round_opt(devil.get("tune_mae_with"), 2),
        },
    }


def _summarize_precip_meta(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if meta is None:
        return None
    metrics = meta.get("metrics") or {}
    features = list(meta.get("feature_order") or [])
    return {
        "version": meta.get("version"),
        "trained_at_utc": meta.get("trained_at_utc"),
        "history_start": meta.get("history_start"),
        "history_end": meta.get("history_end"),
        "feature_count": len(features),
        "classes": list(meta.get("classes") or []),
        "train_rows": _int_opt(metrics.get("train_rows")),
        "val_rows": _int_opt(metrics.get("val_rows")),
        "best_iteration": _int_opt(metrics.get("best_iteration")),
        "val_accuracy": _round_opt(metrics.get("val_accuracy"), 1, scale=100.0),
        "val_f1": _round_opt(metrics.get("val_precip_f1"), 1, scale=100.0),
        "val_rain_f1": _round_opt(metrics.get("val_rain_f1"), 1, scale=100.0),
        "val_snow_f1": _round_opt(metrics.get("val_snow_f1"), 1, scale=100.0),
        "val_mlogloss": _round_opt(metrics.get("val_mlogloss"), 3),
        "precip_threshold": _round_opt(meta.get("precip_threshold"), 2),
        "sleet_threshold": _round_opt(meta.get("sleet_threshold"), 2),
    }


def _latest_tagged_run(cur: sqlite3.Cursor, exp_id: int, tag_key: str) -> dict[str, Any] | None:
    row = cur.execute(
        """
        SELECT r.run_uuid, r.name, r.start_time, r.end_time
        FROM runs r
        JOIN tags t ON t.run_uuid = r.run_uuid
        WHERE r.experiment_id = ?
          AND r.lifecycle_stage = 'active'
          AND t.key = ?
          AND LOWER(t.value) = 'true'
        ORDER BY r.start_time DESC
        LIMIT 1
        """,
        (exp_id, tag_key),
    ).fetchone()
    if row is None:
        return None
    rid, name, start_ms, end_ms = row
    return {
        "run_id": rid,
        "name": name or rid[:8],
        "at_utc": _ms_to_utc_iso(end_ms or start_ms),
    }


def _load_mlflow_runs(mlflow_db_path: Path, limit: int) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    empty_latest = {"solar": None, "precip": None}
    if not mlflow_db_path.exists():
        return [], 0, empty_latest

    uri = f"file:{mlflow_db_path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    try:
        cur = conn.cursor()
        exp_row = cur.execute(
            "SELECT experiment_id FROM experiments WHERE name = ?",
            (EXPERIMENT_NAME,),
        ).fetchone()
        if exp_row is None:
            return [], 0, empty_latest
        exp_id = exp_row[0]

        latest_promoted = {
            "solar": _latest_tagged_run(cur, exp_id, "promoted"),
            "precip": _latest_tagged_run(cur, exp_id, "precip_saved"),
        }

        total_row = cur.execute(
            "SELECT COUNT(*) FROM runs WHERE experiment_id = ? AND lifecycle_stage = 'active'",
            (exp_id,),
        ).fetchone()
        total_runs = int(total_row[0]) if total_row else 0

        run_rows = cur.execute(
            """
            SELECT run_uuid, name, status, start_time, end_time
            FROM runs
            WHERE experiment_id = ? AND lifecycle_stage = 'active'
            ORDER BY start_time DESC
            LIMIT ?
            """,
            (exp_id, limit),
        ).fetchall()
        if not run_rows:
            return [], total_runs, latest_promoted

        run_ids = [row[0] for row in run_rows]
        placeholders = ",".join("?" for _ in run_ids)

        metrics_by_run: dict[str, dict[str, float]] = {rid: {} for rid in run_ids}
        for rid, key, value in cur.execute(
            f"SELECT run_uuid, key, value FROM metrics WHERE run_uuid IN ({placeholders})",
            run_ids,
        ):
            metrics_by_run[rid][key] = value

        tags_by_run: dict[str, dict[str, str]] = {rid: {} for rid in run_ids}
        for rid, key, value in cur.execute(
            f"SELECT run_uuid, key, value FROM tags WHERE run_uuid IN ({placeholders}) AND key NOT LIKE 'mlflow.%'",
            run_ids,
        ):
            tags_by_run[rid][key] = value

        runs: list[dict[str, Any]] = []
        for rid, name, status, start_ms, end_ms in run_rows:
            m = metrics_by_run[rid]
            t = tags_by_run[rid]
            duration_s = None
            if isinstance(start_ms, int) and isinstance(end_ms, int) and end_ms >= start_ms:
                duration_s = round((end_ms - start_ms) / 1000.0, 1)

            runs.append(
                {
                    "run_id": rid,
                    "name": name or rid[:8],
                    "status": status,
                    "start_utc": _ms_to_utc_iso(start_ms),
                    "end_utc": _ms_to_utc_iso(end_ms),
                    "duration_s": duration_s,
                    "promoted": _parse_bool_tag(t.get("promoted")),
                    "precip_saved": _parse_bool_tag(t.get("precip_saved")),
                    "objective": t.get("objective"),
                    "val_mae": _round_opt(m.get("val_mae"), 2),
                    "val_mae_with_devil": _round_opt(m.get("val_mae_with_devil"), 2),
                    "val_rmse": _round_opt(m.get("val_rmse"), 2),
                    "val_r2": _round_opt(m.get("val_r2"), 3),
                    "precip_val_accuracy": _round_opt(m.get("precip_val_accuracy"), 1, scale=100.0),
                    "precip_val_f1": _round_opt(m.get("precip_val_f1"), 1, scale=100.0),
                    "precip_val_mlogloss": _round_opt(m.get("precip_val_mlogloss"), 3),
                    "unseen_rows": _int_opt(m.get("unseen_rows")),
                }
            )

        return runs, total_runs, latest_promoted
    finally:
        conn.close()


def build_model_status(
    model_dir: Path,
    mlflow_db_path: Path,
    train_state_path: Path,
    limit: int = 20,
) -> dict[str, Any]:
    solar_meta = load_meta(model_dir / SOLAR_META_FILE)
    precip_meta = load_meta(model_dir / PRECIP_META_FILE)
    devil_exists = (model_dir / DEVIL_MODEL_FILE).exists()
    runs, total_runs, latest_promoted = _load_mlflow_runs(mlflow_db_path, limit=limit)
    solar_summary = _summarize_solar_meta(solar_meta, devil_exists=devil_exists)
    if solar_summary is not None and latest_promoted["solar"] is not None:
        solar_summary["run_name"] = latest_promoted["solar"]["name"]
        solar_summary["promoted_at_utc"] = latest_promoted["solar"]["at_utc"]
    precip_summary = _summarize_precip_meta(precip_meta)
    if precip_summary is not None and latest_promoted["precip"] is not None:
        precip_summary["run_name"] = latest_promoted["precip"]["name"]
        precip_summary["promoted_at_utc"] = latest_promoted["precip"]["at_utc"]
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_state": _load_train_state(train_state_path),
        "active_models": {
            "solar": solar_summary,
            "precip": precip_summary,
        },
        "runs": runs,
        "total_runs": total_runs,
    }
