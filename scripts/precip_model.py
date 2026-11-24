"""4-class precipitation classifier trained on SMHI present-weather observations at Uppsala Flygplats.

Predicts hourly precipitation type (Inget, Regn, Sno, Snoblandat) from Open-Meteo weather inputs,
independent of the cloud-cover model so the sunshine forecast stays untouched.
"""
from __future__ import annotations

import csv
import io
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import requests
import xgboost as xgb
from icon_features import ICON_FEATURES


SMHI_WEATHER_URL = "https://opendata-download-metobs.smhi.se/api/version/latest/parameter/13/station/97530/period"
PRECIP_MODEL_FILE = "uppsala_precip_model.ubj"
PRECIP_META_FILE = "uppsala_precip_meta.pkl"

PRECIP_CLASSES = ["Inget", "Regn", "Sno", "Snoblandat"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(PRECIP_CLASSES)}
PRECIP_ACTIVE_LABELS = {"Regn", "Sno", "Snoblandat"}

# Extra Open-Meteo hourly fields needed specifically for precipitation type.
PRECIP_EXTRA_HOURLY = [
    "temperature_2m",
    "dew_point_2m",
    "rain",
    "showers",
    "snowfall",
    "weather_code",
]

BASE_WEATHER_FEATURES = [
    "cloud_cover",
    "precipitation",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "relative_humidity_2m",
]

PRECIP_WEATHER_INPUTS = BASE_WEATHER_FEATURES + PRECIP_EXTRA_HOURLY + ICON_FEATURES
PRECIP_FEATURE_ORDER = PRECIP_WEATHER_INPUTS + [
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]

# Official SMHI parameter 13 (Radande vader) codes mapped to 4 classes.
# Missing/ambiguous codes (e.g. 509/510 = no obs, 122 = drizzle or snow grains) return None and are skipped.
_SLEET_CODES = {
    23, 26, 68, 69, 83, 84,
    167, 168,
    259, 282, 286, 290,
}
_SNOW_CODES = (
    {22, 85, 86, 124, 145, 146, 210, 211, 223, 239, 283, 284, 287, 291}
    | set(range(70, 79))
    | set(range(170, 179))
    | set(range(185, 188))
    | set(range(270, 280))
)
_RAIN_CODES = (
    {21, 24, 25, 91, 92, 123, 125, 143, 144, 147, 148, 280, 281, 285, 289}
    | set(range(50, 68))
    | set(range(80, 83))
    | set(range(150, 167))
    | set(range(180, 185))
    | set(range(250, 258))
    | set(range(260, 268))
)
_NONE_CODES = (
    set(range(0, 20))
    | set(range(27, 50))
    | set(range(100, 121))
    | set(range(126, 140))
    | {508}
)

PRECIP_THRESHOLDS = [round(0.20 + 0.05 * i, 2) for i in range(12)]  # 0.20 ... 0.75
DEFAULT_PRECIP_THRESHOLD = 0.40
DEFAULT_SLEET_THRESHOLD = 0.15
DEFAULT_SLEET_MIX_RATIO = 0.30
DAILY_PRECIP_MIN_SHARE = 0.25
DAILY_PRECIP_MIN_HOURS = 2

# Intensity comes from Open-Meteo's hourly amount, not the model: a high P(snow) only means snow is likely.
# Snow uses snowfall (cm/h); rain and sleet use precipitation (mm/h, water equivalent).
SNOW_LIGHT_CM, SNOW_HEAVY_CM = 0.2, 1.0
LIQUID_LIGHT_MM, LIQUID_HEAVY_MM = 0.5, 4.0
INTENSITY_ORDER = ["Latt", "Mattlig", "Kraftig"]

PRECIP_XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": len(PRECIP_CLASSES),
    "eval_metric": "mlogloss",
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 3,
    "lambda": 1.0,
    "seed": 42,
}
DEFAULT_PRECIP_ROUNDS = 250
DEFAULT_PRECIP_EARLY_STOPPING = 30


def smhi_weather_code_to_class(code: int) -> int | None:
    """Map an SMHI parameter 13 weather code to 0=Inget, 1=Regn, 2=Sno, 3=Snoblandat, or None if missing/ambiguous."""
    if code in _SLEET_CODES:
        return CLASS_TO_ID["Snoblandat"]
    if code in _SNOW_CODES:
        return CLASS_TO_ID["Sno"]
    if code in _RAIN_CODES:
        return CLASS_TO_ID["Regn"]
    if code in _NONE_CODES:
        return CLASS_TO_ID["Inget"]
    return None


def fetch_smhi_precip_obs() -> dict[str, int]:
    """Observed precipitation class (0..3) at Uppsala Flygplats keyed by "YYYY-MM-DDTHH:MM" (UTC)."""
    obs: dict[str, int] = {}
    response = requests.get(f"{SMHI_WEATHER_URL}/corrected-archive/data.csv", timeout=120)
    response.raise_for_status()
    for row in csv.reader(io.StringIO(response.content.decode("utf-8-sig")), delimiter=";"):
        if len(row) >= 3 and row[0][:2] in ("19", "20") and len(row[1]) == 8 and row[2]:
            try:
                cls = smhi_weather_code_to_class(int(row[2]))
            except ValueError:
                continue
            if cls is not None:
                obs[f"{row[0]}T{row[1][:5]}"] = cls

    response = requests.get(f"{SMHI_WEATHER_URL}/latest-months/data.json", timeout=60)
    response.raise_for_status()
    for item in response.json().get("value", []):
        raw_val = item.get("value")
        if raw_val is None:
            continue
        try:
            cls = smhi_weather_code_to_class(int(raw_val))
        except (TypeError, ValueError):
            continue
        if cls is None:
            continue
        dt = datetime.fromtimestamp(item["date"] / 1000, tz=timezone.utc)
        obs[dt.strftime("%Y-%m-%dT%H:%M")] = cls
    return obs


def build_precip_features(
    times: list[str],
    weather: dict[str, np.ndarray],
    precip_obs: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, list[datetime]]:
    """Build (X, y, timestamps) across all hours (day + night) with valid weather inputs and SMHI param 13 labels."""
    xs: list[list[float]] = []
    ys: list[int] = []
    keep_times: list[datetime] = []

    required = BASE_WEATHER_FEATURES + PRECIP_EXTRA_HOURLY
    for i, raw_time in enumerate(times):
        target = precip_obs.get(raw_time)
        if target is None:
            continue

        req_vals = [float(weather[name][i]) for name in required]
        if not np.all(np.isfinite(req_vals)):
            continue

        icon_vals = [float(weather[name][i]) if name in weather else math.nan for name in ICON_FEATURES]
        iso = raw_time + "Z" if not raw_time.endswith("Z") else raw_time
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)
        hour = dt.hour
        doy = int(dt.strftime("%j"))
        two_pi = 2.0 * math.pi
        xs.append(
            req_vals
            + icon_vals
            + [
                math.sin(two_pi * hour / 24.0),
                math.cos(two_pi * hour / 24.0),
                math.sin(two_pi * doy / 365.25),
                math.cos(two_pi * doy / 365.25),
            ]
        )
        ys.append(int(target))
        keep_times.append(dt)

    if len(xs) < 200:
        raise RuntimeError("For fa nederbordsrader efter filtrering.")

    return np.array(xs, dtype=float), np.array(ys, dtype=int), keep_times


def _binary_f1(y_true_pos: np.ndarray, y_pred_pos: np.ndarray) -> float:
    tp = float(np.sum(y_true_pos & y_pred_pos))
    fp = float(np.sum((~y_true_pos) & y_pred_pos))
    fn = float(np.sum(y_true_pos & (~y_pred_pos)))
    if tp == 0.0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2.0 * precision * recall / (precision + recall))


def choose_precip_threshold(probs: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Pick the P(precip) threshold that maximizes binary precipitation F1 on held-out hours."""
    p_precip = 1.0 - np.asarray(probs, dtype=float)[:, CLASS_TO_ID["Inget"]]
    true_pos = np.asarray(y_true, dtype=int) > 0
    best_t = DEFAULT_PRECIP_THRESHOLD
    best_f1 = -1.0
    for t in PRECIP_THRESHOLDS:
        f1 = _binary_f1(true_pos, p_precip >= t)
        if f1 > best_f1 + 1e-12:
            best_t = t
            best_f1 = f1
    return best_t, max(0.0, best_f1)


def classify_precip_probs(
    probs: np.ndarray | list[float],
    precip_threshold: float = DEFAULT_PRECIP_THRESHOLD,
    sleet_threshold: float = DEFAULT_SLEET_THRESHOLD,
    sleet_mix_ratio: float = DEFAULT_SLEET_MIX_RATIO,
) -> dict[str, Any]:
    """Turn 4-class probabilities [Inget, Regn, Sno, Snoblandat] into a precipitation label and confidence dict."""
    arr = [float(v) for v in probs]
    p_none, p_rain, p_snow, p_sleet = arr[0], arr[1], arr[2], arr[3]
    p_precip = max(0.0, min(1.0, 1.0 - p_none))

    if p_precip < precip_threshold:
        label = "Inget"
    elif p_sleet >= sleet_threshold:
        label = "Snoblandat"
    else:
        rain_snow_sum = p_rain + p_snow
        if rain_snow_sum > 0.0 and min(p_rain, p_snow) / rain_snow_sum >= sleet_mix_ratio:
            label = "Snoblandat"
        elif p_snow >= p_rain:
            label = "Sno"
        else:
            label = "Regn"

    return {
        "precip_label": label,
        "p_precip": round(p_precip, 3),
        "p_rain": round(p_rain, 3),
        "p_snow": round(p_snow, 3),
        "p_sleet": round(p_sleet, 3),
    }


def precip_intensity(label: str | None, weather: dict[str, float | None]) -> str | None:
    """Latt / Mattlig / Kraftig from Open-Meteo's hourly amount, or None when no precipitation is predicted."""
    if label not in PRECIP_ACTIVE_LABELS:
        return None
    if label == "Sno":
        raw, light, heavy = weather.get("snowfall"), SNOW_LIGHT_CM, SNOW_HEAVY_CM
    else:
        raw, light, heavy = weather.get("precipitation"), LIQUID_LIGHT_MM, LIQUID_HEAVY_MM
    amount = float(raw) if raw is not None and math.isfinite(float(raw)) else 0.0
    if amount < light:
        return "Latt"
    if amount > heavy:
        return "Kraftig"
    return "Mattlig"


def predict_precip_for_hour(
    booster: xgb.Booster,
    meta: dict,
    utc_ms: int,
    weather: dict[str, float | None],
) -> dict[str, Any]:
    """Predict hourly precipitation class and probabilities for a single hour."""
    feature_order = meta.get("feature_order") or PRECIP_FEATURE_ORDER
    dt = datetime.fromtimestamp(utc_ms / 1000, tz=timezone.utc)
    hour = dt.hour
    doy = int(dt.strftime("%j"))
    two_pi = 2.0 * math.pi
    values = {
        **weather,
        "hour_sin": math.sin(two_pi * hour / 24.0),
        "hour_cos": math.cos(two_pi * hour / 24.0),
        "doy_sin": math.sin(two_pi * doy / 365.25),
        "doy_cos": math.cos(two_pi * doy / 365.25),
    }
    fv = [values.get(name) for name in feature_order]
    x = np.array([fv], dtype=float)
    probs = booster.predict(xgb.DMatrix(x, feature_names=feature_order))[0]
    out = classify_precip_probs(
        probs,
        precip_threshold=float(meta.get("precip_threshold", DEFAULT_PRECIP_THRESHOLD)),
        sleet_threshold=float(meta.get("sleet_threshold", DEFAULT_SLEET_THRESHOLD)),
        sleet_mix_ratio=float(meta.get("sleet_mix_ratio", DEFAULT_SLEET_MIX_RATIO)),
    )
    out["precip_intensity"] = precip_intensity(out["precip_label"], weather)
    return out


def aggregate_daily_precip(
    hourly_preds: list[dict[str, Any]],
    precip_threshold: float = DEFAULT_PRECIP_THRESHOLD,
    sleet_mix_ratio: float = DEFAULT_SLEET_MIX_RATIO,
) -> dict[str, Any]:
    """Summarize all daytime hourly precipitation predictions into a single daily precipitation label.

    The day's label replaces the sky label in the UI, so a short shower must not turn a sunny day into "Regn":
    it triggers only when a real share of daytime hours is wet (>= DAILY_PRECIP_MIN_SHARE, at least
    DAILY_PRECIP_MIN_HOURS). `precip_threshold` is kept for API compatibility; the hourly labels already use it.
    """
    if not hourly_preds:
        return {"precip_label": "Inget", "precip_intensity": None, "p_precip_max": None}

    p_max = max(float(p.get("p_precip") or 0.0) for p in hourly_preds)
    active = [p for p in hourly_preds if p.get("precip_label") in PRECIP_ACTIVE_LABELS]
    needed = max(DAILY_PRECIP_MIN_HOURS, math.ceil(DAILY_PRECIP_MIN_SHARE * len(hourly_preds)))

    if len(active) < needed:
        return {"precip_label": "Inget", "precip_intensity": None, "p_precip_max": round(p_max, 3)}

    rain_mass = sum(float(p.get("p_rain") or 0.0) for p in active)
    snow_mass = sum(float(p.get("p_snow") or 0.0) for p in active)
    sleet_mass = sum(float(p.get("p_sleet") or 0.0) for p in active)
    active_labels = {p.get("precip_label") for p in active}

    rain_snow_sum = rain_mass + snow_mass
    if "Snoblandat" in active_labels and (sleet_mass >= max(rain_mass, snow_mass) or len(active_labels) > 1):
        label = "Snoblandat"
    elif "Regn" in active_labels and "Sno" in active_labels:
        label = "Snoblandat"
    elif rain_snow_sum > 0.0 and min(rain_mass, snow_mass) / rain_snow_sum >= sleet_mix_ratio:
        label = "Snoblandat"
    elif snow_mass >= rain_mass:
        label = "Sno"
    else:
        label = "Regn"

    # The strongest intensity reached by enough wet hours, so one heavy burst does not make the whole day heavy.
    intensity = "Latt"
    for level in INTENSITY_ORDER:
        reached = [p for p in active if INTENSITY_ORDER.index(p.get("precip_intensity") or "Latt") >= INTENSITY_ORDER.index(level)]
        if len(reached) >= DAILY_PRECIP_MIN_HOURS:
            intensity = level

    return {"precip_label": label, "precip_intensity": intensity, "p_precip_max": round(p_max, 3)}


def _multiclass_logloss(probs: np.ndarray, y_true: np.ndarray) -> float:
    clipped = np.clip(np.asarray(probs, dtype=float), 1e-7, 1.0 - 1e-7)
    idx = np.arange(len(y_true))
    return float(-np.mean(np.log(clipped[idx, y_true])))


def precip_duel(
    challenger: xgb.Booster,
    champion: xgb.Booster | None,
    champion_meta: dict | None,
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_times: list[datetime],
    min_unseen_rows: int,
) -> dict[str, Any]:
    """Decide whether the new precip model replaces the live one, scored on the same validation hours.

    Only hours newer than the live model's training data count, so neither model has seen them.
    """
    result: dict[str, Any] = {"save": True, "unseen_rows": 0, "champion_logloss": None, "challenger_logloss": None}
    if champion is None or not isinstance(champion_meta, dict):
        return result
    if list(champion_meta.get("feature_order") or []) != list(PRECIP_FEATURE_ORDER):
        return result  # live model uses other inputs and cannot be scored on these rows
    try:
        champion_end = datetime.fromisoformat(champion_meta["history_end"])
    except (KeyError, TypeError, ValueError):
        return result

    unseen = np.array([t > champion_end for t in val_times], dtype=bool)
    result["unseen_rows"] = int(unseen.sum())
    if result["unseen_rows"] < min_unseen_rows:
        result["save"] = False  # too few fresh hours to judge, so the live model stays
        return result

    dval = xgb.DMatrix(x_val[unseen], feature_names=PRECIP_FEATURE_ORDER)
    champion_logloss = _multiclass_logloss(champion.predict(dval), y_val[unseen])
    challenger_logloss = _multiclass_logloss(challenger.predict(dval), y_val[unseen])
    result.update(
        save=challenger_logloss < champion_logloss,
        champion_logloss=round(champion_logloss, 5),
        challenger_logloss=round(challenger_logloss, 5),
    )
    return result


def train_precip_model(
    times: list[str],
    weather: dict[str, np.ndarray],
    precip_obs: dict[str, int],
    rounds: int = DEFAULT_PRECIP_ROUNDS,
    early_stopping_rounds: int = DEFAULT_PRECIP_EARLY_STOPPING,
) -> tuple[xgb.Booster, dict[str, Any], tuple[np.ndarray, np.ndarray, list[datetime]]]:
    """Train the 4-class precipitation classifier on the oldest 80% of hours.

    The newest 20% is split in two: a tuning window (early stopping, precip threshold) and a later holdout
    that nothing in this run is fitted on. Metrics are reported on the holdout, and its rows (x, y, times)
    are returned so the caller can duel against the live model on them.
    """
    x, y, keep_times = build_precip_features(times, weather, precip_obs)
    split = int(len(x) * 0.8)
    holdout_start = split + (len(x) - split) // 2
    if split < 100 or len(x) - holdout_start < 50:
        raise RuntimeError("For fa train/validation-rader for nederbordsmodell.")

    x_train, y_train = x[:split], y[:split]
    x_tune, y_tune = x[split:holdout_start], y[split:holdout_start]
    x_val, y_val = x[holdout_start:], y[holdout_start:]

    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=PRECIP_FEATURE_ORDER)
    dtune = xgb.DMatrix(x_tune, label=y_tune, feature_names=PRECIP_FEATURE_ORDER)
    dval = xgb.DMatrix(x_val, label=y_val, feature_names=PRECIP_FEATURE_ORDER)

    callbacks: list[xgb.callback.TrainingCallback] = []
    if early_stopping_rounds > 0:
        callbacks.append(
            xgb.callback.EarlyStopping(
                rounds=early_stopping_rounds,
                data_name="tune",
                metric_name="mlogloss",
                save_best=True,
            )
        )

    booster = xgb.train(
        params=PRECIP_XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=rounds,
        evals=[(dtrain, "train"), (dtune, "tune")],
        callbacks=callbacks,
        verbose_eval=False,
    )

    precip_threshold, tune_precip_f1 = choose_precip_threshold(booster.predict(dtune), y_tune)

    val_probs = booster.predict(dval)
    val_logloss = _multiclass_logloss(val_probs, y_val)

    val_preds = np.array(
        [CLASS_TO_ID[classify_precip_probs(p, precip_threshold=precip_threshold)["precip_label"]] for p in val_probs],
        dtype=int,
    )
    val_precip_f1 = _binary_f1(y_val > 0, val_preds > 0)
    val_rain_f1 = _binary_f1(y_val == CLASS_TO_ID["Regn"], val_preds == CLASS_TO_ID["Regn"])
    val_snow_f1 = _binary_f1(y_val == CLASS_TO_ID["Sno"], val_preds == CLASS_TO_ID["Sno"])
    val_accuracy = float(np.mean(val_preds == y_val))

    class_counts = {name: int(np.sum(y == idx)) for idx, name in enumerate(PRECIP_CLASSES)}
    meta = {
        "version": 1,
        "model_file": PRECIP_MODEL_FILE,
        "classes": list(PRECIP_CLASSES),
        "feature_order": list(PRECIP_FEATURE_ORDER),
        "target": "smhi_param13_present_weather",
        "precip_threshold": float(precip_threshold),
        "sleet_threshold": float(DEFAULT_SLEET_THRESHOLD),
        "sleet_mix_ratio": float(DEFAULT_SLEET_MIX_RATIO),
        "class_counts": class_counts,
        "metrics": {
            "val_mlogloss": round(val_logloss, 5),
            "val_accuracy": round(val_accuracy, 4),
            "tune_precip_f1": round(tune_precip_f1, 4),
            "val_precip_f1": round(val_precip_f1, 4),
            "val_rain_f1": round(val_rain_f1, 4),
            "val_snow_f1": round(val_snow_f1, 4),
            "train_rows": int(len(x_train)),
            "tune_rows": int(len(x_tune)),
            "val_rows": int(len(x_val)),
            "best_iteration": int(booster.best_iteration) if early_stopping_rounds > 0 else None,
        },
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "history_start": keep_times[0].isoformat(),
        "history_end": keep_times[-1].isoformat(),
    }
    return booster, meta, (x_val, y_val, keep_times[holdout_start:])
