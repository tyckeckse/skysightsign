"""One-off experiment: squared vs absolute error on the same data. Does not touch the live model."""
from __future__ import annotations

import mlflow
import numpy as np
import xgboost as xgb

from train_model import (
    DEFAULT_BOOST_ROUNDS,
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_HISTORY_DAYS,
    FEATURE_ORDER,
    ROOT_DIR,
    build_features,
    fetch_history,
    mae,
    rmse,
    split_temporal,
)

# Each objective is early-stopped on the metric it actually optimizes.
OBJECTIVES = {"reg:squarederror": "rmse", "reg:absoluteerror": "mae"}


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{ROOT_DIR / 'mlflow.db'}")
    mlflow.set_experiment("objective-compare")  # separate from the champion experiment

    # Fetch and split once, so both models see identical rows.
    data = fetch_history(DEFAULT_HISTORY_DAYS)
    x, y, _ = build_features(data)
    x_train, y_train, x_val, y_val = split_temporal(x, y)
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=FEATURE_ORDER)
    dval = xgb.DMatrix(x_val, label=y_val, feature_names=FEATURE_ORDER)
    clear = y_val <= 5  # the clear hours where squared error hedges

    for objective, stop_metric in OBJECTIVES.items():
        params = {
            "objective": objective,
            "eval_metric": ["rmse", "mae"],
            "max_depth": 4,
            "eta": 0.06,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "lambda": 1.0,
            "min_child_weight": 2.0,
            "seed": 42,
        }
        with mlflow.start_run(run_name=objective):
            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=DEFAULT_BOOST_ROUNDS,
                evals=[(dval, "val")],
                callbacks=[
                    xgb.callback.EarlyStopping(
                        rounds=DEFAULT_EARLY_STOPPING_ROUNDS, metric_name=stop_metric, save_best=True
                    )
                ],
                verbose_eval=False,
            )
            pred = booster.predict(dval)
            metrics = {
                "val_mae": mae(y_val, pred),
                "val_rmse": rmse(y_val, pred),
                "clear_mae": mae(y_val[clear], pred[clear]),
                "clear_mean_pred": float(np.mean(pred[clear])),
            }
            mlflow.log_params({"objective": objective, "best_iteration": booster.best_iteration})
            mlflow.log_metrics(metrics)
            print(objective, {k: round(v, 2) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
