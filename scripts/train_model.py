from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import pickle
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import requests
import xgboost as xgb
from icon_features import ICON_FEATURES, add_icon_features
from log_utils import kv, setup_logger
from overcast_devil import (
    EXTRA_FEATURES,
    add_extra_cloud,
    apply_devil,
    choose_threshold,
    out_of_fold_predictions,
    overcast_probability,
    train_devil,
)
from precip_model import (
    PRECIP_EXTRA_HOURLY,
    PRECIP_META_FILE,
    PRECIP_MODEL_FILE,
    fetch_smhi_precip_obs,
    precip_duel,
    train_precip_model,
)
from mlflow import MlflowClient


UPPSALA_LAT = 59.8586
UPPSALA_LON = 17.6389
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY_DAYS = 1400  # ~start of MET Nordic (MEPS) in Open-Meteo forecast history, Dec 2022
# Features: past Open-Meteo forecasts. Target: cloud cover observed by SMHI at Uppsala Flygplats.
FORECAST_HISTORY_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
SMHI_CLOUD_URL = "https://opendata-download-metobs.smhi.se/api/version/latest/parameter/16/station/97530/period"
DEFAULT_BOOST_ROUNDS = 100
DEFAULT_EARLY_STOPPING_ROUNDS = 30
DEFAULT_THRESHOLDS = {"mixed_idx": 0.35, "sunny_idx": 0.65}
DEFAULT_TRAIN_LOG_PATH = ROOT_DIR / "data" / "train_model.log"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_EVAL_EVERY = 10
TRAIN_STATE_PATH = ROOT_DIR / "data" / "train_state.json"
# Hourly Open-Meteo fields used as model inputs.
WEATHER_FEATURES = [
    "cloud_cover",
    "precipitation",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "relative_humidity_2m",
]
MODEL_INPUTS = WEATHER_FEATURES + ICON_FEATURES
FEATURE_ORDER = MODEL_INPUTS + [
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]


@dataclass
class Dataset:
    times: list[str]
    weather: dict[str, np.ndarray]
    is_day: np.ndarray
    observed_cloud: np.ndarray


class PeriodicEvalLogger(xgb.callback.TrainingCallback):
    def __init__(self, logger: logging.Logger, every: int) -> None:
        super().__init__()
        self.logger = logger
        self.every = max(1, int(every))
        self.last_epoch_seen: int | None = None
        self.last_logged_epoch: int | None = None
        self.last_train_rmse: float | None = None
        self.last_val_rmse: float | None = None

    @staticmethod
    def _latest_metric(
        evals_log: dict[str, dict[str, list[float]]],
        data_name: str,
        metric_name: str,
    ) -> float | None:
        data_metrics = evals_log.get(data_name, {})
        metric_values = data_metrics.get(metric_name, [])
        if not metric_values:
            return None
        return float(metric_values[-1])

    def _log_epoch(self, epoch: int) -> None:
        payload: dict[str, object] = {"svc": "train", "event": "train_epoch", "iter": int(epoch)}
        if self.last_train_rmse is not None:
            payload["train_rmse"] = round(self.last_train_rmse, 5)
        if self.last_val_rmse is not None:
            payload["val_rmse"] = round(self.last_val_rmse, 5)
        self.logger.info(kv(**payload))
        self.last_logged_epoch = epoch

    def after_iteration(
        self,
        model: xgb.Booster,
        epoch: int,
        evals_log: dict[str, dict[str, list[float]]],
    ) -> bool:
        self.last_epoch_seen = int(epoch)
        self.last_train_rmse = self._latest_metric(evals_log, "train", "rmse")
        self.last_val_rmse = self._latest_metric(evals_log, "val", "rmse")

        if epoch == 0 or epoch % self.every == 0:
            self._log_epoch(epoch)
        return False

    def after_training(self, model: xgb.Booster) -> xgb.Booster:
        if self.last_epoch_seen is not None and self.last_epoch_seen != self.last_logged_epoch:
            self._log_epoch(self.last_epoch_seen)
        return model


def parse_iso_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def day_of_year(dt: datetime) -> int:
    return int(dt.strftime("%j"))


def fetch_smhi_cloud_obs() -> dict[str, float]:
    """Observed total cloud cover (%) keyed by "YYYY-MM-DDTHH:MM" (UTC)."""
    obs: dict[str, float] = {}
    # Quality-checked archive (all but the last ~3 months), as CSV with a metadata header.
    response = requests.get(f"{SMHI_CLOUD_URL}/corrected-archive/data.csv", timeout=120)
    response.raise_for_status()
    for row in csv.reader(io.StringIO(response.content.decode("utf-8-sig")), delimiter=";"):
        if len(row) >= 3 and row[0][:2] in ("19", "20") and len(row[1]) == 8 and row[2]:
            obs[f"{row[0]}T{row[1][:5]}"] = float(row[2])
    # The most recent months, as JSON.
    response = requests.get(f"{SMHI_CLOUD_URL}/latest-months/data.json", timeout=60)
    response.raise_for_status()
    for item in response.json().get("value", []):
        if item.get("value") is None:
            continue
        dt = datetime.fromtimestamp(item["date"] / 1000, tz=timezone.utc)
        obs[dt.strftime("%Y-%m-%dT%H:%M")] = float(item["value"])
    return obs


def fetch_history(history_days: int) -> Dataset:
    end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
    start_date = end_date - timedelta(days=history_days)
    hourly_fields = list(dict.fromkeys(WEATHER_FEATURES + PRECIP_EXTRA_HOURLY + ["is_day"]))
    params = {
        "latitude": f"{UPPSALA_LAT:.4f}",
        "longitude": f"{UPPSALA_LON:.4f}",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(hourly_fields),
        "timezone": "UTC",
    }
    response = requests.get(FORECAST_HISTORY_URL, params=params, timeout=120)
    response.raise_for_status()
    hourly = response.json().get("hourly", {})
    add_icon_features(FORECAST_HISTORY_URL, params, hourly, timeout=300)
    add_extra_cloud(FORECAST_HISTORY_URL, params, hourly, timeout=300)

    times = hourly.get("time", [])
    # dtype=float turns missing (None) values into NaN; build_features skips rows missing MODEL_INPUTS.
    # Extra models (devil-only inputs) may stay NaN: ECMWF's archive starts later than the others.
    weather_keys = list(dict.fromkeys(MODEL_INPUTS + PRECIP_EXTRA_HOURLY + EXTRA_FEATURES))
    weather = {name: np.array(hourly.get(name, []), dtype=float) for name in weather_keys}
    is_day = np.array(hourly.get("is_day", []), dtype=float)

    lengths = {len(times), len(is_day)} | {len(values) for values in weather.values()}
    if len(lengths) != 1 or len(times) == 0:
        raise RuntimeError("Prognoshistorik saknar eller har inkonsekventa tidsserier.")

    obs = fetch_smhi_cloud_obs()
    observed_cloud = np.array([obs.get(t, np.nan) for t in times], dtype=float)

    return Dataset(
        times=times,
        weather=weather,
        is_day=is_day,
        observed_cloud=observed_cloud,
    )


def build_features(data: Dataset) -> tuple[np.ndarray, np.ndarray, list[datetime]]:
    parsed_times = [parse_iso_utc(t + "Z" if not t.endswith("Z") else t) for t in data.times]

    xs: list[list[float]] = []
    ys: list[float] = []
    keep_times: list[datetime] = []

    for i, dt in enumerate(parsed_times):
        if data.is_day[i] != 1:
            continue
        weather_values = [float(data.weather[name][i]) for name in MODEL_INPUTS]
        target = data.observed_cloud[i]
        if not np.all(np.isfinite(weather_values)) or not np.isfinite(target):
            continue

        hour = dt.hour
        doy = day_of_year(dt)
        xs.append(
            weather_values
            + [
                math.sin(2.0 * math.pi * hour / 24.0),
                math.cos(2.0 * math.pi * hour / 24.0),
                math.sin(2.0 * math.pi * doy / 365.25),
                math.cos(2.0 * math.pi * doy / 365.25),
            ]
        )
        ys.append(float(target))
        keep_times.append(dt)

    if len(xs) < 200:
        raise RuntimeError("For fa datapunkter efter filtrering.")

    return np.array(xs, dtype=float), np.array(ys, dtype=float), keep_times


def extra_matrix(data: Dataset, keep_times: list[datetime]) -> np.ndarray:
    """Extra models' cloud cover for the rows build_features kept, in the same order (NaN where missing)."""
    index = {t[:16]: i for i, t in enumerate(data.times)}
    rows = [index[dt.strftime("%Y-%m-%dT%H:%M")] for dt in keep_times]
    return np.column_stack([data.weather[name][rows] for name in EXTRA_FEATURES])


def split_temporal(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = int(len(x) * 0.8)
    if split < 100 or len(x) - split < 50:
        raise RuntimeError("For fa train/validation-rader for robust eval.")
    return x[:split], y[:split], x[split:], y[split:]


def cloud_evaluation_slices(train_rows: int, remaining_rows: int) -> tuple[slice, slice]:
    """Split the rows after training into a tuning window and a later promotion holdout."""
    holdout_start = train_rows + remaining_rows // 2
    return slice(train_rows, holdout_start), slice(holdout_start, train_rows + remaining_rows)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    var = np.sum((y_true - np.mean(y_true)) ** 2)
    if var == 0:
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / var)


def ensure_dirs() -> tuple[Path, Path]:
    model_dir = ROOT_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    return ROOT_DIR, model_dir


def save_train_state(payload: dict) -> None:
    TRAIN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_existing_thresholds(meta_path: Path) -> tuple[float, float]:
    mixed = float(DEFAULT_THRESHOLDS["mixed_idx"])
    sunny = float(DEFAULT_THRESHOLDS["sunny_idx"])
    if not meta_path.exists():
        return mixed, sunny
    try:
        with meta_path.open("rb") as f:
            meta = pickle.load(f)
        if not isinstance(meta, dict):
            return mixed, sunny
        th = meta.get("thresholds", {})
        if isinstance(th, dict):
            mixed = float(th.get("mixed_idx", mixed))
            sunny = float(th.get("sunny_idx", sunny))
        return mixed, sunny
    except Exception:
        return mixed, sunny


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trana Uppsala-solmodell med XGBoost.")
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_BOOST_ROUNDS,
        help=f"Antal boosting-rundor (default: {DEFAULT_BOOST_ROUNDS}).",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help=f"Antal historikdagar for traning (default: {DEFAULT_HISTORY_DAYS}).",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=DEFAULT_EARLY_STOPPING_ROUNDS,
        help=(
            "Stoppa nar val-metriken inte blir battre pa N rundor "
            f"(default: {DEFAULT_EARLY_STOPPING_ROUNDS}, 0 = av)."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_TRAIN_LOG_PATH,
        help=f"Path till train-loggfil (default: {DEFAULT_TRAIN_LOG_PATH}).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=DEFAULT_LOG_LEVEL,
        help=f"Loggniva (default: {DEFAULT_LOG_LEVEL}).",
    )
    parser.add_argument(
        "--log-eval-every",
        type=int,
        default=DEFAULT_LOG_EVAL_EVERY,
        help=f"Logga eval varje N iterationer (default: {DEFAULT_LOG_EVAL_EVERY}).",
    )
    return parser.parse_args()


MODEL_NAME = "uppsala-solar"
DEVIL_FILE = "uppsala_overcast_devil.ubj"
MIN_UNSEEN_ROWS = 48  # fewer rows than this is too noisy to judge, so the champion stays


def setup_mlflow() -> None:
    """Track runs in a local SQLite file; artifacts go to ./mlruns no matter where the script is started from."""
    mlflow.set_tracking_uri(f"sqlite:///{ROOT_DIR / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(MODEL_NAME) is None:
        mlflow.create_experiment(MODEL_NAME, artifact_location=(ROOT_DIR / "mlruns").as_uri())
    mlflow.set_experiment(MODEL_NAME)


def load_champion(model_path: Path) -> xgb.Booster | None:
    """The live model file, which is written on every promotion, so no MLflow artifacts are needed."""
    if not model_path.exists():
        return None
    # A file that exists but can't be read raises, so a broken file never gets replaced without a duel.
    champion = xgb.Booster()
    champion.load_model(model_path)
    # The champion may use fewer inputs (e.g. before ICON was added); it is scored on its own columns.
    return champion if set(champion.feature_names or []) <= set(FEATURE_ORDER) else None


def champion_matrix(champion: xgb.Booster, x: np.ndarray) -> xgb.DMatrix:
    """Pick out the columns the champion was trained on, in its own order."""
    cols = [FEATURE_ORDER.index(name) for name in champion.feature_names]
    return xgb.DMatrix(x[:, cols], feature_names=champion.feature_names)


def champion_predictions(champion: xgb.Booster, x: np.ndarray, extra: np.ndarray, meta_path: Path, devil_path: Path) -> np.ndarray:
    """The live forecast as the app makes it: the champion, plus its devil if it has one."""
    base = champion.predict(champion_matrix(champion, x))
    try:
        with meta_path.open("rb") as f:
            threshold = pickle.load(f)["devil"]["threshold"]
        devil = xgb.Booster()
        devil.load_model(devil_path)
        return apply_devil(base, overcast_probability(devil, x, FEATURE_ORDER, extra, base), threshold)
    except (OSError, KeyError, TypeError, ValueError, xgb.core.XGBoostError):
        return base  # older champion without a devil (or its devil no longer fits the inputs)


def load_history_end(meta_path: Path) -> datetime | None:
    """Last observation time used to train, tune, or evaluate the champion for promotion."""
    try:
        with meta_path.open("rb") as f:
            return datetime.fromisoformat(pickle.load(f)["history_end"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


LOCK_PATH = ROOT_DIR / "data" / "train.lock"
STALE_LOCK_SECONDS = 30 * 60  # a run killed by sleep leaves its lock behind; ignore it after 30 min


@contextmanager
def train_lock(lock_path: Path = LOCK_PATH):
    """Yield True if we got the lock, False if another training is already running."""
    if lock_path.exists() and time.time() - lock_path.stat().st_mtime > STALE_LOCK_SECONDS:
        lock_path.unlink(missing_ok=True)
    try:
        # O_EXCL makes "create only if missing" a single atomic step, so two runs can't both win.
        os.close(os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        yield False
        return
    try:
        yield True
    finally:
        lock_path.unlink(missing_ok=True)


def main() -> None:
    with train_lock() as acquired:
        if not acquired:
            print("Another training is running; skipping.")
            return
        train()


def train() -> None:
    args = parse_args()
    logger = setup_logger("sunshine.train", args.log_file, level=args.log_level)
    started_monotonic = time.monotonic()

    try:
        if args.rounds < 1:
            raise ValueError("--rounds maste vara minst 1.")
        if args.history_days < 30:
            raise ValueError("--history-days maste vara minst 30.")
        if args.early_stopping_rounds < 0:
            raise ValueError("--early-stopping-rounds maste vara minst 0.")
        if args.log_eval_every < 1:
            raise ValueError("--log-eval-every maste vara minst 1.")

        _, model_dir = ensure_dirs()
        model_path = model_dir / "uppsala_solar_model.ubj"
        meta_path = model_dir / "uppsala_solar_meta.pkl"
        devil_path = model_dir / DEVIL_FILE
        precip_model_path = model_dir / PRECIP_MODEL_FILE
        precip_meta_path = model_dir / PRECIP_META_FILE

        logger.info(
            kv(
                svc="train",
                event="train_start",
                history_days=args.history_days,
                rounds=args.rounds,
                early_stopping_rounds=args.early_stopping_rounds,
                log_eval_every=args.log_eval_every,
            )
        )

        existing_mixed, existing_sunny = load_existing_thresholds(meta_path)

        data = fetch_history(args.history_days)
        x, y, keep_times = build_features(data)
        x_train, y_train, x_eval, _ = split_temporal(x, y)
        tune, holdout = cloud_evaluation_slices(len(x_train), len(x_eval))
        x_val, y_val = x[tune], y[tune]
        extra = extra_matrix(data, keep_times)
        extra_train, extra_val = extra[: len(x_train)], extra[tune]

        logger.info(
            kv(
                svc="train",
                event="train_data_ready",
                train_rows=len(x_train),
                val_rows=len(x_val),
                holdout_rows=len(y[holdout]),
            )
        )

        dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=FEATURE_ORDER)
        dval = xgb.DMatrix(x_val, label=y_val, feature_names=FEATURE_ORDER)
        params = {
            # Absolute error aims at the median, so clear hours are not pulled up toward the mean.
            "objective": "reg:absoluteerror",
            "eval_metric": ["rmse", "mae"],
            "max_depth": 4,
            "eta": 0.06,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "lambda": 1.0,
            "min_child_weight": 2.0,
            "seed": 42,
        }
        callbacks: list[xgb.callback.TrainingCallback] = [PeriodicEvalLogger(logger, args.log_eval_every)]
        if args.early_stopping_rounds > 0:
            callbacks.append(
                xgb.callback.EarlyStopping(
                    rounds=args.early_stopping_rounds,
                    data_name="val",
                    metric_name="mae",
                    save_best=True,
                )
            )

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=args.rounds,
            evals=[(dtrain, "train"), (dval, "val")],
            callbacks=callbacks,
            verbose_eval=False,
        )

        pred_val = booster.predict(dval)
        val_mae = mae(y_val, pred_val)
        val_rmse = rmse(y_val, pred_val)
        val_r2 = r2(y_val, pred_val)

        # Train the devil on out-of-fold forecasts and tune its threshold on the validation window.
        rounds_used = int(booster.num_boosted_rounds())

        def train_main(xs: np.ndarray, ys: np.ndarray) -> xgb.Booster:
            dm = xgb.DMatrix(xs, label=ys, feature_names=FEATURE_ORDER)
            return xgb.train(params, dm, num_boost_round=rounds_used, verbose_eval=False)

        oof_train = out_of_fold_predictions(x_train, y_train, train_main, FEATURE_ORDER)
        devil = train_devil(x_train, extra_train, oof_train, y_train, FEATURE_ORDER)
        p_val = overcast_probability(devil, x_val, FEATURE_ORDER, extra_val, pred_val)
        devil_threshold, tune_mae_without, tune_mae_with = choose_threshold(pred_val, p_val, y_val)
        final_val = apply_devil(pred_val, p_val, devil_threshold)
        final_val_mae = mae(y_val, final_val)
        logger.info(
            kv(
                svc="train",
                event="devil_summary",
                threshold=devil_threshold,
                tune_mae_without=round(tune_mae_without, 4),
                tune_mae_with=round(tune_mae_with, 4),
                val_mae_without=round(val_mae, 4),
                val_mae_with=round(final_val_mae, 4),
                pulled_share=round(float(np.mean(final_val != pred_val)), 4),
            )
        )

        # Score only holdout rows newer than any data used in the champion's promotion.
        setup_mlflow()
        champion = load_champion(model_path)
        champion_end = load_history_end(meta_path)
        champion_mae: float | None = None
        challenger_mae: float | None = None
        unseen_rows = 0
        if champion is None or champion_end is None or not devil_path.exists():
            promote = True
        else:
            unseen = np.array([t > champion_end for t in keep_times[holdout]])
            unseen_rows = int(unseen.sum())
            if unseen_rows >= MIN_UNSEEN_ROWS:
                x_duel, extra_duel, y_duel = x[holdout][unseen], extra[holdout][unseen], y[holdout][unseen]
                champion_pred = champion_predictions(champion, x_duel, extra_duel, meta_path, devil_path)
                challenger_base = booster.predict(xgb.DMatrix(x_duel, feature_names=FEATURE_ORDER))
                challenger_prob = overcast_probability(devil, x_duel, FEATURE_ORDER, extra_duel, challenger_base)
                challenger_pred = apply_devil(challenger_base, challenger_prob, devil_threshold)
                champion_mae = mae(y_duel, champion_pred)
                challenger_mae = mae(y_duel, challenger_pred)
            promote = challenger_mae is not None and challenger_mae < champion_mae

        threshold_mixed = existing_mixed
        threshold_sunny = existing_sunny
        threshold_source = "existing_or_default"

        logger.info(
            kv(
                svc="train",
                event="train_eval_summary",
                mae=round(val_mae, 4),
                rmse=round(val_rmse, 4),
                r2=round(val_r2, 6),
                best_iteration=int(booster.best_iteration) if args.early_stopping_rounds > 0 else None,
                best_val_mae=round(float(booster.best_score), 5) if args.early_stopping_rounds > 0 else None,
            )
        )
        logger.info(
            kv(
                svc="train",
                event="thresholds_final",
                mixed_idx=round(threshold_mixed, 4),
                sunny_idx=round(threshold_sunny, 4),
                source=threshold_source,
            )
        )

        exported_meta = {
            "version": 2,
            "feature_order": FEATURE_ORDER,
            "target": "smhi_observed_cloud_cover_pct",
            "thresholds": {
                "sunny_idx": float(threshold_sunny),
                "mixed_idx": float(threshold_mixed),
                "source": threshold_source,
            },
            "night_rule": "is_day_from_open_meteo",
            "devil": {
                "model_file": DEVIL_FILE,
                "threshold": devil_threshold,
                "tune_mae_without": tune_mae_without,
                "tune_mae_with": tune_mae_with,
                "val_mae_with": final_val_mae,
            },
            "metrics": {
                "mae": val_mae,
                "rmse": val_rmse,
                "r2": val_r2,
                "train_rows": int(len(x_train)),
                "val_rows": int(len(x_val)),
                "boost_rounds_requested": int(args.rounds),
                "boost_rounds_used": int(getattr(booster, "num_boosted_rounds", lambda: args.rounds)()),
                "early_stopping_rounds": int(args.early_stopping_rounds),
                "best_iteration": int(booster.best_iteration) if args.early_stopping_rounds > 0 else None,
                "objective": params["objective"],
                "best_val_mae": float(booster.best_score) if args.early_stopping_rounds > 0 else None,
            },
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "history_start": keep_times[0].isoformat(),
            "history_end": keep_times[-1].isoformat(),
            "location": {"name": "Uppsala", "lat": UPPSALA_LAT, "lon": UPPSALA_LON},
        }

        # Companion 4-class precipitation model trained against SMHI parameter 13 (present weather).
        precip_obs = fetch_smhi_precip_obs()
        precip_booster, precip_meta, (precip_x_val, precip_y_val, precip_val_times) = train_precip_model(
            data.times, data.weather, precip_obs
        )
        precip_metrics = precip_meta["metrics"]
        logger.info(
            kv(
                svc="train",
                event="precip_train_summary",
                val_mlogloss=precip_metrics["val_mlogloss"],
                val_accuracy=precip_metrics["val_accuracy"],
                val_precip_f1=precip_metrics["val_precip_f1"],
                val_rain_f1=precip_metrics["val_rain_f1"],
                val_snow_f1=precip_metrics["val_snow_f1"],
                precip_threshold=precip_meta["precip_threshold"],
                train_rows=precip_metrics["train_rows"],
                val_rows=precip_metrics["val_rows"],
            )
        )

        # Fair duel for the precip model too, independent of whether the cloud model was promoted.
        precip_champion: xgb.Booster | None = None
        precip_champion_meta: dict | None = None
        if precip_model_path.exists() and precip_meta_path.exists():
            try:
                precip_champion = xgb.Booster()
                precip_champion.load_model(precip_model_path)
                with precip_meta_path.open("rb") as f:
                    precip_champion_meta = pickle.load(f)
            except (OSError, pickle.UnpicklingError, xgb.core.XGBoostError):
                precip_champion, precip_champion_meta = None, None
        precip_duel_result = precip_duel(
            precip_booster,
            precip_champion,
            precip_champion_meta,
            precip_x_val,
            precip_y_val,
            precip_val_times,
            MIN_UNSEEN_ROWS,
        )
        save_precip = bool(precip_duel_result["save"])
        logger.info(kv(svc="train", event="precip_duel", **precip_duel_result))

        with mlflow.start_run():
            mlflow.log_metrics({"val_mae": val_mae, "val_rmse": val_rmse, "val_r2": val_r2})
            mlflow.log_metric("val_mae_with_devil", final_val_mae)
            if devil_threshold is not None:
                mlflow.log_metric("devil_threshold", devil_threshold)
            mlflow.log_metrics(
                {
                    "precip_val_mlogloss": float(precip_metrics["val_mlogloss"]),
                    "precip_val_accuracy": float(precip_metrics["val_accuracy"]),
                    "precip_val_f1": float(precip_metrics["val_precip_f1"]),
                    "precip_rain_f1": float(precip_metrics["val_rain_f1"]),
                    "precip_snow_f1": float(precip_metrics["val_snow_f1"]),
                    "precip_threshold": float(precip_meta["precip_threshold"]),
                }
            )
            mlflow.log_metric("unseen_rows", unseen_rows)
            if champion_mae is not None:
                mlflow.log_metrics({"champion_unseen_mae": champion_mae, "challenger_unseen_mae": challenger_mae})
            mlflow.set_tag("promoted", str(promote).lower())
            mlflow.set_tag("precip_saved", str(save_precip).lower())
            mlflow.log_metric("precip_unseen_rows", precip_duel_result["unseen_rows"])
            if precip_duel_result["champion_logloss"] is not None:
                mlflow.log_metrics(
                    {
                        "precip_champion_unseen_logloss": precip_duel_result["champion_logloss"],
                        "precip_challenger_unseen_logloss": precip_duel_result["challenger_logloss"],
                    }
                )
            mlflow.set_tag("objective", params["objective"])
            model_info = mlflow.xgboost.log_model(booster, name="model", registered_model_name=MODEL_NAME)
            mlflow.xgboost.log_model(devil, name="devil")
            mlflow.xgboost.log_model(precip_booster, name="precip")
            mlflow.log_dict(exported_meta, "meta.json")
            mlflow.log_dict(precip_meta, "precip_meta.json")

        if promote:
            MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", model_info.registered_model_version)
            booster.save_model(model_path)
            devil.save_model(devil_path)
            with meta_path.open("wb") as f:
                pickle.dump(exported_meta, f)
        if save_precip:
            precip_booster.save_model(precip_model_path)
            with precip_meta_path.open("wb") as f:
                pickle.dump(precip_meta, f)
        logger.info(
            kv(
                svc="train",
                event="champion_decision",
                promoted=promote,
                unseen_rows=unseen_rows,
                min_unseen_rows=MIN_UNSEEN_ROWS,
                challenger_unseen_mae=None if challenger_mae is None else round(challenger_mae, 4),
                champion_unseen_mae=None if champion_mae is None else round(champion_mae, 4),
            )
        )

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        next_train_state = {
            "last_train_utc": now_iso,
            "history_days": int(args.history_days),
        }
        save_train_state(next_train_state)

        logger.info(
            kv(
                svc="train",
                event="model_saved",
                model_path=model_path,
                meta_path=meta_path,
            )
        )
        elapsed_ms = int((time.monotonic() - started_monotonic) * 1000)
        logger.info(
            kv(
                svc="train",
                event="train_end",
                status="ok",
                elapsed_ms=elapsed_ms,
                val_rmse=round(val_rmse, 4),
                best_iteration=int(booster.best_iteration) if args.early_stopping_rounds > 0 else None,
            )
        )
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - started_monotonic) * 1000)
        logger.exception(
            kv(
                svc="train",
                event="train_failed",
                msg="Training failed",
                error=str(exc),
                elapsed_ms=elapsed_ms,
            )
        )
        raise


if __name__ == "__main__":
    main()
