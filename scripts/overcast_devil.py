"""The "devil": a yes/no model on top of the main model that pulls a forecast to 100% when full overcast is likely.

The main model hedges on overcast hours (says ~70 when the answer is 100). The devil sees the main model's forecast,
its inputs and cloud cover from more weather models, and estimates P(observed >= 95%). Above a threshold learned
on held-out hours, the forecast becomes 100; otherwise the main model's number is kept.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import requests
import xgboost as xgb

# More weather models in Open-Meteo (ICON is already a main-model input). Same names in the live and archive APIs.
EXTRA_MODELS = ["dmi_seamless", "gfs_seamless", "ukmo_seamless", "meteofrance_seamless", "ecmwf_ifs025"]
EXTRA_FEATURES = [f"extra_cloud_{model}" for model in EXTRA_MODELS]
# Cloud-cover columns counted when asking "how many models say overcast?".
SOURCE_COLUMNS = ["cloud_cover", "icon_cloud_cover"] + EXTRA_FEATURES
OVERCAST_PCT = 95.0
# Beyond ~2 days several extra models stop (DMI, UKMO), so the devil would see inputs it was never trained on.
DEVIL_MAX_LEAD_HOURS = 48
THRESHOLDS = [round(0.10 + 0.05 * i, 2) for i in range(15)]  # 0.10 ... 0.80
OOF_FOLDS = 5
DEVIL_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 5,
    "seed": 42,
}
DEVIL_ROUNDS = 300


def add_extra_cloud(url: str, base_params: dict, hourly: dict, timeout: int = 120) -> None:
    """Fetch cloud cover from EXTRA_MODELS and add extra_cloud_* series to `hourly`, aligned on its "time" list.

    Hours a model does not cover (or a failed request) become None; XGBoost treats them as missing.
    """
    params = {key: value for key, value in base_params.items() if key not in ("hourly", "daily")}
    params.update({"hourly": "cloud_cover", "models": ",".join(EXTRA_MODELS)})
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        extra = response.json().get("hourly", {})
    except (requests.RequestException, ValueError):
        extra = {}

    index = {t: i for i, t in enumerate(extra.get("time", []))}
    for model, name in zip(EXTRA_MODELS, EXTRA_FEATURES):
        series = extra.get(f"cloud_cover_{model}") or []
        hourly[name] = [
            series[index[t]] if t in index and index[t] < len(series) else None for t in hourly.get("time", [])
        ]


def devil_matrix(x: np.ndarray, feature_order: list[str], extra: np.ndarray, base_pred: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Devil inputs: main-model inputs, extra models, the main model's forecast, and how many sources say overcast."""
    columns = {name: x[:, i] for i, name in enumerate(feature_order)}
    columns.update({name: extra[:, i] for i, name in enumerate(EXTRA_FEATURES)})
    sources = np.column_stack([columns[name] for name in SOURCE_COLUMNS])
    with np.errstate(invalid="ignore"):
        n_overcast = np.sum(sources >= OVERCAST_PCT, axis=1).astype(float)
    n_sources = np.sum(np.isfinite(sources), axis=1).astype(float)
    names = feature_order + EXTRA_FEATURES + ["main_pred", "n_sources_overcast", "n_sources"]
    return np.column_stack([x, extra, base_pred, n_overcast, n_sources]), names


def out_of_fold_predictions(
    x: np.ndarray, y: np.ndarray, train_main: Callable[[np.ndarray, np.ndarray], xgb.Booster], feature_order: list[str],
) -> np.ndarray:
    """Main-model forecasts for every training row from a model that never saw that row.

    Otherwise the devil would learn from the main model's (too good) in-sample forecasts and trust it too much.
    """
    pred = np.zeros(len(x))
    for block in np.array_split(np.arange(len(x)), OOF_FOLDS):
        rest = np.setdiff1d(np.arange(len(x)), block)
        model = train_main(x[rest], y[rest])
        pred[block] = model.predict(xgb.DMatrix(x[block], feature_names=feature_order))
    return pred


def train_devil(x: np.ndarray, extra: np.ndarray, base_pred: np.ndarray, y: np.ndarray, feature_order: list[str]) -> xgb.Booster:
    features, names = devil_matrix(x, feature_order, extra, base_pred)
    dtrain = xgb.DMatrix(features, label=(y >= OVERCAST_PCT).astype(float), feature_names=names)
    return xgb.train(DEVIL_PARAMS, dtrain, num_boost_round=DEVIL_ROUNDS)


def overcast_probability(devil: xgb.Booster, x: np.ndarray, feature_order: list[str], extra: np.ndarray, base_pred: np.ndarray) -> np.ndarray:
    features, names = devil_matrix(x, feature_order, extra, base_pred)
    return devil.predict(xgb.DMatrix(features, feature_names=names))


def apply_devil(base_pred: np.ndarray, p_overcast: np.ndarray, threshold: float | None) -> np.ndarray:
    """Pull to 100 where the devil is at least `threshold` sure of overcast. None = devil switched off."""
    out = np.asarray(base_pred, dtype=float).copy()
    if threshold is not None:
        out[np.asarray(p_overcast) >= threshold] = 100.0
    return out


def choose_threshold(base_pred: np.ndarray, p_overcast: np.ndarray, y: np.ndarray) -> tuple[float | None, float, float]:
    """Threshold with the lowest mean absolute error. Returns (threshold, mae_without, mae_with).

    If no threshold beats leaving the main model alone, the threshold is None and the devil stays quiet.
    """
    mae_without = float(np.mean(np.abs(y - base_pred)))
    best, best_mae = None, mae_without
    for threshold in THRESHOLDS:
        score = float(np.mean(np.abs(y - apply_devil(base_pred, p_overcast, threshold))))
        if score < best_mae:
            best, best_mae = threshold, score
    return best, mae_without, best_mae
