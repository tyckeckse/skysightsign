"""One-off experiment: does adding other weather models' cloud forecasts help? Does not touch the live model."""
from __future__ import annotations

import mlflow
import numpy as np
import requests
import xgboost as xgb

from train_model import (
    DEFAULT_BOOST_ROUNDS,
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_HISTORY_DAYS,
    FEATURE_ORDER,
    FORECAST_HISTORY_URL,
    ROOT_DIR,
    UPPSALA_LAT,
    UPPSALA_LON,
    build_features,
    fetch_history,
    mae,
    split_temporal,
)

# Other models in Open-Meteo's forecast archive. Ours today is the default (MET Nordic for Scandinavia).
EXTRA_MODELS = ["icon_seamless", "dmi_seamless", "gfs_seamless", "ukmo_seamless", "meteofrance_seamless", "ecmwf_ifs025"]


def fetch_extra_cloud(start: str, end: str) -> dict[str, dict[str, float]]:
    """Cloud cover (%) per extra model, keyed by Open-Meteo time string. Missing hours are left out."""
    params = {
        "latitude": f"{UPPSALA_LAT:.4f}",
        "longitude": f"{UPPSALA_LON:.4f}",
        "start_date": start,
        "end_date": end,
        "hourly": "cloud_cover",
        "models": ",".join(EXTRA_MODELS),
        "timezone": "UTC",
    }
    response = requests.get(FORECAST_HISTORY_URL, params=params, timeout=300)
    response.raise_for_status()
    hourly = response.json()["hourly"]
    return {
        model: {t: v for t, v in zip(hourly["time"], hourly[f"cloud_cover_{model}"]) if v is not None}
        for model in EXTRA_MODELS
    }


def train_and_score(x_train, y_train, x_val, y_val, names: list[str]) -> tuple[xgb.Booster, np.ndarray]:
    params = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "max_depth": 4,
        "eta": 0.06,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "min_child_weight": 2.0,
        "seed": 42,
    }
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=names)
    dval = xgb.DMatrix(x_val, label=y_val, feature_names=names)
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=DEFAULT_BOOST_ROUNDS,
        evals=[(dval, "val")],
        callbacks=[xgb.callback.EarlyStopping(rounds=DEFAULT_EARLY_STOPPING_ROUNDS, metric_name="mae", save_best=True)],
        verbose_eval=False,
    )
    return booster, booster.predict(dval)


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{ROOT_DIR / 'mlflow.db'}")
    mlflow.set_experiment("feature-compare")

    data = fetch_history(DEFAULT_HISTORY_DAYS)
    x, y, keep_times = build_features(data)

    # Same date range as the base data; one extra column per model, NaN where that model has no forecast.
    extra = fetch_extra_cloud(data.times[0][:10], data.times[-1][:10])
    keys = [t.strftime("%Y-%m-%dT%H:%M") for t in keep_times]
    extra_cols = np.array([[extra[m].get(k, np.nan) for m in EXTRA_MODELS] for k in keys], dtype=float)
    for m, col in zip(EXTRA_MODELS, extra_cols.T):
        print(f"{m:22} coverage {np.isfinite(col).mean():.0%}")

    variants = {
        "baseline": (x, FEATURE_ORDER),
        "plus_models": (np.hstack([x, extra_cols]), FEATURE_ORDER + [f"cloud_{m}" for m in EXTRA_MODELS]),
    }
    for name, (features, names) in variants.items():
        x_train, y_train, x_val, y_val = split_temporal(features, y)
        with mlflow.start_run(run_name=name):
            booster, pred = train_and_score(x_train, y_train, x_val, y_val, names)
            clear, overcast = y_val <= 5, y_val >= 95
            metrics = {
                "val_mae": mae(y_val, pred),
                "clear_mae": mae(y_val[clear], pred[clear]),
                "overcast_mae": mae(y_val[overcast], pred[overcast]),
                "clear_mean_pred": float(np.mean(pred[clear])),
                "overcast_mean_pred": float(np.mean(pred[overcast])),
            }
            mlflow.log_params({"features": len(names), "best_iteration": booster.best_iteration})
            mlflow.log_metrics(metrics)
            print(name, {k: round(v, 1) for k, v in metrics.items()})
            if name == "plus_models":
                gain = booster.get_score(importance_type="gain")
                top = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:8]
                print("  most useful inputs:", ", ".join(f"{k} ({v:.0f})" for k, v in top))


if __name__ == "__main__":
    main()
