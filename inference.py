"""Frozen RINTIK inference and deterministic sidecar construction.

The numeric order mirrors the official submission notebook:
feature engineering -> Stage 1 level/carry -> Stage 2 deviation A/B ->
causal EWM reconstruction -> Post 1 -> Post 2 -> Post 3.

The official checkpoint is immutable. Runtime-only state that the checkpoint
does not contain is supplied by a separately checksummed JSON sidecar.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

OFFICIAL_MODEL_SHA256 = "adce8fda2f465679ea6e8fe31124822a45a555b9dc154e12076ef5b6140699b0"
OFFICIAL_TRAIN_SHA256 = "b1488e92ddf438e8d224b4d4d6d55455b2c0495df5ce79a2f3c7a41cfe02917f"
OFFICIAL_TEST_SHA256 = "292b7d1a26f6b23448e2525c7a4d45b72b00d8f385accd004fa709fa76905d03"
OFFICIAL_SUBMISSION_SHA256 = "eb8745be1ff55357c5c7d0fe981cc55c19125ec11d5a9b198c92df929a3951ad"
OFFICIAL_STATE_SHA256 = "a24da3473bf42a10fb9bde05e37495e5b69adffde9193dd4648daa9e0898996e"
STATE_SCHEMA_VERSION = 1
PIPELINE_VERSION = "official-submission-v1+daily-calendar-window-v1"

REQUIRED_SENSOR_COLS = [
    "env_id", "co2", "pressure", "air_temperature", "env_battery",
    "soil_ec_line1", "soil_humidity_line1", "soil_temp_line1",
    "soil_ec_line2", "soil_humidity_line2", "soil_temp_line2",
    "soil_ec_line3", "soil_humidity_line3", "soil_temp_line3",
    "water_volume_line1", "water_volume_line2", "water_volume_line3",
    "gdd", "standard_day_degree", "daily_mean_temperature",
    "daily_max_above_Tbase", "daily_max", "daily_max_reduction", "ontario_units",
]
INPUT_COLUMNS = ["datetime", *REQUIRED_SENSOR_COLS]
CRITICAL_FEATURES = ["air_temperature", "co2", "pressure"]


class ArtifactIntegrityError(RuntimeError):
    """Raised when a protected artifact or its sidecar cannot be trusted."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
        separators=(",", ": "),
    ).encode("utf-8") + b"\n"


def canonical_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing input columns: {', '.join(missing)}")
    out = frame.loc[:, INPUT_COLUMNS].copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="raise")
    for column in REQUIRED_SENSOR_COLS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values("datetime").reset_index(drop=True)


def canonical_frame_sha256(frame: pd.DataFrame) -> str:
    normalized = canonical_input_frame(frame)
    normalized["datetime"] = normalized["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    payload = normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_frozen_artifact(
    path: str | Path,
    expected_sha256: str = OFFICIAL_MODEL_SHA256,
) -> tuple[dict[str, Any], str]:
    model_path = Path(path)
    if not model_path.is_file():
        raise ArtifactIntegrityError(f"Frozen checkpoint not found: {model_path}")
    actual = sha256_file(model_path)
    if actual != expected_sha256:
        raise ArtifactIntegrityError(
            f"Frozen checkpoint SHA256 mismatch: expected {expected_sha256}, got {actual}"
        )
    # torch.load uses pickle internally. It is permitted only after matching the
    # pinned hash of the trusted competition artifact.
    if os.environ.get("RINTIK_LOW_MEMORY") == "1":
        artifact = _load_artifact_without_resident_torch(model_path, actual)
    else:
        import torch

        artifact = torch.load(model_path, map_location="cpu")
    required = {
        "artifact_version", "target", "stage1", "stage2",
        "feature_engineering", "reconstruction", "post_modelling",
    }
    missing = sorted(required.difference(artifact))
    if missing:
        raise ArtifactIntegrityError(f"Checkpoint schema missing keys: {missing}")
    return artifact, actual


def _load_artifact_without_resident_torch(model_path: Path, verified_sha256: str) -> dict[str, Any]:
    """Hosting hemat memori (mis. Render 512 MB): torch dipakai sekali di proses anak.

    Checkpoint hanya berisi tipe Python dan numpy, jadi isinya dapat dibawa ke proses
    utama lewat pickle tanpa memuat torch (±150 MB) secara permanen. Hash checkpoint
    sudah diverifikasi pemanggil; proses anak memverifikasinya ulang sebelum torch.load.
    """
    cache_dir = Path(os.environ.get("RINTIK_ARTIFACT_CACHE_DIR", model_path.parent))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"artifact_{verified_sha256}.pickle"
    if not cache_path.is_file():
        code = (
            "import hashlib,pickle,sys,torch\n"
            "src,dst,expected=sys.argv[1],sys.argv[2],sys.argv[3]\n"
            "h=hashlib.sha256()\n"
            "f=open(src,'rb')\n"
            "[h.update(b) for b in iter(lambda:f.read(1<<20),b'')]\n"
            "f.close()\n"
            "assert h.hexdigest()==expected,'checkpoint hash mismatch'\n"
            "a=torch.load(src,map_location='cpu')\n"
            "tmp=dst+'.tmp'\n"
            "pickle.dump(a,open(tmp,'wb'),protocol=pickle.HIGHEST_PROTOCOL)\n"
            "import os; os.replace(tmp,dst)\n"
        )
        subprocess.run(
            [sys.executable, "-c", code, str(model_path), str(cache_path), verified_sha256],
            check=True,
        )
    with open(cache_path, "rb") as handle:
        return pickle.load(handle)


def load_state_sidecar(
    path: str | Path,
    model_sha256: str,
    expected_state_sha256: str = OFFICIAL_STATE_SHA256,
) -> tuple[dict[str, Any], str]:
    state_path = Path(path)
    if not state_path.is_file():
        raise ArtifactIntegrityError(f"Inference state sidecar not found: {state_path}")
    state_sha = sha256_file(state_path)
    if state_sha != expected_state_sha256:
        raise ArtifactIntegrityError(
            f"Inference state SHA256 mismatch: expected {expected_state_sha256}, got {state_sha}"
        )
    try:
        sidecar = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"Invalid inference state sidecar: {exc}") from exc
    if sidecar.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ArtifactIntegrityError("Unsupported inference state schema version")
    if sidecar.get("pipeline_version") != PIPELINE_VERSION:
        raise ArtifactIntegrityError("Inference state pipeline version mismatch")
    recorded_model = sidecar.get("sources", {}).get("model", {}).get("sha256")
    if recorded_model != model_sha256:
        raise ArtifactIntegrityError("Inference state was not built from the loaded checkpoint")
    if sidecar.get("sources", {}).get("train", {}).get("sha256") != OFFICIAL_TRAIN_SHA256:
        raise ArtifactIntegrityError("Inference state train provenance hash is not pinned")
    if sidecar.get("sources", {}).get("test", {}).get("sha256") != OFFICIAL_TEST_SHA256:
        raise ArtifactIntegrityError("Inference state test provenance hash is not pinned")
    state = sidecar.get("inference_state", {})
    needed = {"daily_median_features", "daily_median", "last10_mean", "carry_r0", "day_context"}
    if not needed.issubset(state):
        raise ArtifactIntegrityError("Inference state is incomplete")
    if len(state["daily_median_features"]) != len(state["daily_median"]):
        raise ArtifactIntegrityError("Daily median feature/value lengths differ")
    return sidecar, state_sha


def esf(x: Iterable[float]) -> np.ndarray:
    values = np.asarray(x, float)
    return 0.6108 * np.exp(17.27 * values / (values + 237.3))


def causal_ewm(x: Iterable[float], alpha: float) -> np.ndarray:
    values = np.asarray(x, float)
    if alpha < 1:
        # Deliberately keep pandas' default adjust=True: this is notebook parity.
        return pd.Series(values).ewm(alpha=alpha).mean().to_numpy()
    return values


def clip100(values: Iterable[float]) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 100.0)


def tail_compression(values: Iterable[float], quantile: float, strength: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    center = float(np.median(values))
    distance = np.abs(values - center)
    threshold = float(np.quantile(distance, quantile))
    return values - strength * np.sign(values - center) * np.maximum(distance - threshold, 0.0)


def lead_lag_center(values: Iterable[float], weight: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    previous = np.roll(values, 1)
    previous[0] = values[0]
    following = np.roll(values, -1)
    following[-1] = values[-1]
    return (1 - weight) * values + weight * (0.5 * previous + 0.5 * following)


def rolling_median_blend(values: Iterable[float], window: int, weight: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = pd.Series(values).rolling(window, center=True, min_periods=1).median().to_numpy()
    return (1 - weight) * values + weight * median


def curvature(values: Iterable[float], strength: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    first = np.diff(values, prepend=values[0])
    second = np.diff(first, prepend=first[0])
    return values + strength * second


def slot_blend(values: Iterable[float], timestamps: Iterable[Any], weight: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    times = pd.Series(pd.to_datetime(list(timestamps))).reset_index(drop=True)
    slots = times.dt.hour.to_numpy() * 2 + times.dt.minute.to_numpy() // 30
    slot_mean = (
        pd.DataFrame({"slot": slots, "value": values})
        .groupby("slot")["value"]
        .transform("mean")
        .to_numpy()
    )
    return (1 - weight) * values + weight * slot_mean


def load_boosters(artifact: dict[str, Any]) -> dict[str, Any]:
    stage1_lgb = lgb.Booster(model_str=artifact["stage1"]["lgbm"]["model_string"])
    stage1_xgb = xgb.Booster()
    stage1_xgb.load_model(bytearray(artifact["stage1"]["xgb"]["raw_model"]))

    def xgb_list(items: list[dict[str, Any]]) -> list[xgb.Booster]:
        output = []
        for item in items:
            booster = xgb.Booster()
            booster.load_model(bytearray(item["raw_model"]))
            output.append(booster)
        return output

    return {
        "s1_lgbm": stage1_lgb,
        "s1_xgb": stage1_xgb,
        "A_xgb": xgb_list(artifact["stage2"]["A"]["xgb"]),
        "A_lgbm": [lgb.Booster(model_str=item["model_string"]) for item in artifact["stage2"]["A"]["lgbm"]],
        "B_xgb": xgb_list(artifact["stage2"]["B"]["xgb"]),
        "B_lgbm": [lgb.Booster(model_str=item["model_string"]) for item in artifact["stage2"]["B"]["lgbm"]],
    }


def build_features(frame: pd.DataFrame, drivers: list[str]) -> pd.DataFrame:
    full = frame.copy()
    full["es"] = esf(full["air_temperature"])
    full["day"] = full["datetime"].dt.floor("D")
    full["slot"] = (full["datetime"].dt.hour * 2 + full["datetime"].dt.minute // 30).astype(int)
    for column in drivers:
        grouped = full.groupby("day")[column]
        full[column + "_dm"] = grouped.transform("mean")
        full[column + "_dsd"] = grouped.transform("std")
        full[column + "_dmin"] = grouped.transform("min")
        full[column + "_dmax"] = grouped.transform("max")
        full[column + "_dev"] = full[column] - full[column + "_dm"]
    full["phys_ratio"] = esf(full["air_temperature_dm"]) / full["es"]
    for harmonic in range(1, 11):
        full[f"sh{harmonic}"] = np.sin(2 * np.pi * harmonic * full["slot"] / 48)
        full[f"ch{harmonic}"] = np.cos(2 * np.pi * harmonic * full["slot"] / 48)
    full["dT"] = full["air_temperature_dev"]
    full["des"] = full["es_dev"]
    full["dT2"] = full["dT"] ** 2
    full["dTdes"] = full["dT"] * full["des"]
    lagged = {}
    for feature in ["dT", "des", "soil_humidity_line2_dev", "co2_dev"]:
        for lag in [1, 2, 3, 6, 12]:
            lagged[f"{feature}_l{lag}"] = full.groupby("day")[feature].shift(lag)
    return pd.concat([full, pd.DataFrame(lagged, index=full.index)], axis=1).copy()


def _daily_feature_frame(full: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    engineering = artifact["feature_engineering"]
    columns = list(dict.fromkeys(
        engineering["DAG"]
        + [column + suffix for column in engineering["DRIV"] for suffix in ["_dm", "_dsd", "_dmin", "_dmax"]]
    ))
    daily = full.groupby("day").agg(**{column: (column, "first") for column in columns})
    daily["doy"] = daily.index.dayofyear
    return daily


def _stage1_base(
    daily: pd.DataFrame,
    artifact: dict[str, Any],
    boosters: dict[str, Any],
    state: dict[str, Any],
) -> pd.Series:
    features = state["daily_median_features"]
    median = pd.Series(state["daily_median"], index=features)
    values = daily.reindex(columns=features).fillna(median)
    ridge = artifact["stage1"]["ridge"]
    ridge_prediction = (
        ((values.to_numpy() - np.asarray(ridge["scaler_mean"])) / np.asarray(ridge["scaler_scale"]))
        @ np.asarray(ridge["coef"])
        + float(ridge["intercept"])
    )
    blend = artifact["stage1"]["blend"]
    prediction = (
        blend["ridge"] * ridge_prediction
        + blend["lgbm"] * boosters["s1_lgbm"].predict(values.to_numpy())
        + blend["xgb"] * boosters["s1_xgb"].predict(
            xgb.DMatrix(values.to_numpy(), feature_names=features)
        )
    )
    return pd.Series(0.85 * prediction + 0.15 * state["last10_mean"], index=daily.index)


def predict_internal(
    input_frame: pd.DataFrame,
    artifact: dict[str, Any],
    boosters: dict[str, Any],
    state: dict[str, Any],
) -> pd.DataFrame:
    test = canonical_input_frame(input_frame)
    if test.empty:
        raise ValueError("Inference input is empty")
    context_rows = state["day_context"]["rows"]
    context = canonical_input_frame(pd.DataFrame(context_rows))
    if context["datetime"].max() >= test["datetime"].min():
        raise ValueError("Runtime input overlaps or precedes the frozen pre-test context")
    frame = pd.concat([context, test], ignore_index=True)
    full = build_features(frame, artifact["feature_engineering"]["DRIV"])
    daily = _daily_feature_frame(full, artifact)
    base = _stage1_base(daily, artifact, boosters, state)

    test_days = list(pd.Index(test["datetime"].dt.floor("D").unique()))
    carry = artifact["stage1"]["carry"]
    level = base.copy()
    for index, day in enumerate(test_days):
        level.loc[day] = (
            base.loc[day]
            + carry["weight"] * state["carry_r0"] * (carry["rho"] ** index)
        )
    full["lvl"] = full["day"].map(level)
    full["phys_dev"] = full["lvl"] * full["phys_ratio"] - full["lvl"]
    lagged_phys = {}
    for lag in [1, 2, 3, 6, 12]:
        lagged_phys[f"phys_dev_l{lag}"] = full.groupby("day")["phys_dev"].shift(lag)
    full = pd.concat([full, pd.DataFrame(lagged_phys, index=full.index)], axis=1)

    test_indices = np.arange(len(context), len(frame))

    def ensemble(group: str) -> np.ndarray:
        config = artifact["stage2"][group]
        features = config["features"]
        values = full[features].fillna(0.0).to_numpy()[test_indices]
        xgb_prediction = np.mean([
            model.predict(xgb.DMatrix(values, feature_names=features))
            for model in boosters[f"{group}_xgb"]
        ], axis=0)
        lgb_prediction = np.mean([
            model.predict(values) for model in boosters[f"{group}_lgbm"]
        ], axis=0)
        weight = config["tree_blend_weight_xgb"]
        return weight * xgb_prediction + (1 - weight) * lgb_prediction

    stage2 = artifact["stage2"]
    deviation = stage2["amplitude"] * (
        stage2["A_B_blend_weight_A"] * ensemble("A")
        + (1 - stage2["A_B_blend_weight_A"]) * ensemble("B")
    )
    level_values = full["lvl"].to_numpy()[test_indices]
    reconstruction = artifact["reconstruction"]
    prediction = np.clip(
        causal_ewm(level_values + deviation, reconstruction["smoothing_alpha"]),
        *reconstruction["physical_clip"],
    )
    return pd.DataFrame({
        "datetime": test["datetime"].to_numpy(),
        "pred_internal": prediction,
    })


def apply_postprocessing(
    pred_internal: Iterable[float],
    timestamps: Iterable[Any],
    artifact: dict[str, Any],
) -> dict[str, np.ndarray]:
    values = np.asarray(pred_internal, dtype=float)
    if values.size == 0:
        raise ValueError("Post-processing input is empty")
    post = artifact["post_modelling"]
    post_1 = clip100(tail_compression(values, post["tail_quantile"], post["tail_strength"]))
    ml1 = clip100(post_1 + post["affine_shift"])
    ml2 = clip100(lead_lag_center(ml1, post["lead_lag_weight"]))
    ml3 = clip100(rolling_median_blend(ml2, post["rolling_window"], post["rolling_weight_stage2"]))
    ml4 = clip100(tail_compression(ml3, post["tail_quantile"], post["tail_strength"]))
    post_2 = clip100(0.20 * ml3 + 0.80 * ml4)

    center = float(np.median(post_2))
    beam1 = clip100(center + post["beam_amplitude"] * (post_2 - center))
    beam2 = clip100(curvature(beam1, post["curvature_strength"]))
    beam3 = clip100(rolling_median_blend(
        beam2, post["rolling_window"], post["rolling_weight_beam"]
    ))
    beam4 = clip100(slot_blend(beam3, timestamps, post["slot_blend_weight"]))
    ewm = causal_ewm(beam4, post["beam_ewm_alpha"])
    ewm_weight = post["beam_ewm_weight"]
    final = clip100((1 - ewm_weight) * beam4 + ewm_weight * ewm)
    return {"post_1": post_1, "post_2": post_2, "post_3": final}


def predict_full_horizon(
    input_frame: pd.DataFrame,
    artifact: dict[str, Any],
    boosters: dict[str, Any],
    state: dict[str, Any],
) -> pd.DataFrame:
    internal = predict_internal(input_frame, artifact, boosters, state)
    stages = apply_postprocessing(internal["pred_internal"], internal["datetime"], artifact)
    output = internal.copy()
    for name, values in stages.items():
        output[name] = values
    output["air_humidity"] = np.round(output["post_3"], 4)
    return output


def build_state_sidecar(
    artifact: dict[str, Any],
    boosters: dict[str, Any],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    source_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    target = artifact["target"]
    if target not in train_frame.columns:
        raise ValueError(f"Training target {target!r} is missing")
    train = train_frame.copy()
    test = canonical_input_frame(test_frame)
    train["datetime"] = pd.to_datetime(train["datetime"], errors="raise")
    for column in REQUIRED_SENSOR_COLS + [target]:
        train[column] = pd.to_numeric(train[column], errors="coerce")

    combined = pd.concat([train, test], ignore_index=True, sort=False)
    featured = build_features(combined, artifact["feature_engineering"]["DRIV"])
    daily = _daily_feature_frame(featured, artifact)
    train_days = list(featured.iloc[:len(train)]["day"].unique())
    features = artifact["stage1"]["ridge"]["features"]
    median = daily.loc[train_days].reindex(columns=features).median()
    day_mean = featured.iloc[:len(train)].groupby("day")[target].mean()
    provisional_state = {
        "daily_median_features": features,
        "daily_median": [float(median[column]) for column in features],
        "last10_mean": float(day_mean.reindex(train_days).tail(10).mean()),
        "carry_r0": 0.0,
    }
    base = _stage1_base(daily, artifact, boosters, provisional_state)
    residual = (day_mean.reindex(train_days) - base.reindex(train_days)).dropna()
    carry_r0 = float(residual.tail(artifact["stage1"]["carry"]["last_n"]).mean())

    first_test_day = test["datetime"].dt.floor("D").min()
    context = train.loc[train["datetime"].dt.floor("D") == first_test_day, INPUT_COLUMNS].copy()
    context["datetime"] = context["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    context_rows = []
    for record in context.to_dict(orient="records"):
        context_rows.append({
            key: (None if pd.isna(value) else value)
            for key, value in record.items()
        })

    manifests = []
    for day, group in test.groupby(test["datetime"].dt.strftime("%Y-%m-%d"), sort=True):
        manifests.append({
            "date": day,
            "row_count": int(len(group)),
            "first_timestamp": group["datetime"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "last_timestamp": group["datetime"].max().strftime("%Y-%m-%d %H:%M:%S"),
            "payload_sha256": canonical_frame_sha256(group),
        })

    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "sources": source_metadata,
        "official_test": {
            "columns": INPUT_COLUMNS,
            "total_rows": int(len(test)),
            "payload_sha256": canonical_frame_sha256(test),
            "first_timestamp": test["datetime"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "last_timestamp": test["datetime"].max().strftime("%Y-%m-%d %H:%M:%S"),
            "days": manifests,
        },
        "inference_state": {
            "daily_median_features": features,
            "daily_median": provisional_state["daily_median"],
            "last10_mean": provisional_state["last10_mean"],
            "carry_r0": carry_r0,
            "day_context": {
                "date": first_test_day.strftime("%Y-%m-%d"),
                "row_count": int(len(context_rows)),
                "rows": context_rows,
            },
        },
    }
