"""Pipeline inference HBD Air Humidity dari artifact pickle (tanpa torch).

Urutan persis notebook FINALIZATION_IPYNB_HBD.ipynb:
feature engineering -> stage1 level+carry -> stage2 deviasi A/B ->
rekonstruksi EWM 0.72 -> post-1 tail -> post-2 multilayer -> post-3 beam -> final.
Tidak ada retraining, tidak ada ubahan recipe.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

REQUIRED_SENSOR_COLS = [
    "env_id", "co2", "pressure", "air_temperature", "env_battery",
    "soil_ec_line1", "soil_humidity_line1", "soil_temp_line1",
    "soil_ec_line2", "soil_humidity_line2", "soil_temp_line2",
    "soil_ec_line3", "soil_humidity_line3", "soil_temp_line3",
    "water_volume_line1", "water_volume_line2", "water_volume_line3",
    "gdd", "standard_day_degree", "daily_mean_temperature",
    "daily_max_above_Tbase", "daily_max", "daily_max_reduction", "ontario_units",
]


def esf(x):
    x = np.asarray(x, float)
    return 0.6108 * np.exp(17.27 * x / (x + 237.3))


def causal_ewm(x, a):
    x = np.asarray(x, float)
    if a < 1:
        return pd.Series(x).ewm(alpha=a).mean().to_numpy()
    return x


def clip100(v):
    return np.clip(np.asarray(v, dtype=float), 0.0, 100.0)


def tail_compression(v, quantile, strength):
    v = np.asarray(v, dtype=float)
    center = float(np.median(v))
    distance = np.abs(v - center)
    threshold = float(np.quantile(distance, quantile))
    return v - strength * np.sign(v - center) * np.maximum(distance - threshold, 0.0)


def lead_lag_center(v, weight):
    v = np.asarray(v, dtype=float)
    prev = np.roll(v, 1)
    prev[0] = v[0]
    nxt = np.roll(v, -1)
    nxt[-1] = v[-1]
    return (1 - weight) * v + weight * (0.5 * prev + 0.5 * nxt)


def rolling_median_blend(v, window, weight):
    v = np.asarray(v, dtype=float)
    med = pd.Series(v).rolling(window, center=True, min_periods=1).median().to_numpy()
    return (1 - weight) * v + weight * med


def curvature(v, strength):
    v = np.asarray(v, dtype=float)
    d1 = np.diff(v, prepend=v[0])
    d2 = np.diff(d1, prepend=d1[0])
    return v + strength * d2


def slot_blend(v, timestamps, weight):
    v = np.asarray(v, dtype=float)
    t = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
    slot = t.dt.hour.to_numpy() * 2 + t.dt.minute.to_numpy() // 30
    slot_mean = pd.DataFrame({"slot": slot, "value": v}).groupby("slot")["value"].transform("mean").to_numpy()
    return (1 - weight) * v + weight * slot_mean


def load_boosters(art):
    gb1 = lgb.Booster(model_str=art["stage1"]["lgbm"]["model_string"])
    xb1 = xgb.Booster()
    xb1.load_model(bytearray(art["stage1"]["xgb"]["raw_model"]))

    def load_xgb_list(items):
        out = []
        for m in items:
            b = xgb.Booster()
            b.load_model(bytearray(m["raw_model"]))
            out.append(b)
        return out

    Ax = load_xgb_list(art["stage2"]["A"]["xgb"])
    Al = [lgb.Booster(model_str=m["model_string"]) for m in art["stage2"]["A"]["lgbm"]]
    Bx = load_xgb_list(art["stage2"]["B"]["xgb"])
    Bl = [lgb.Booster(model_str=m["model_string"]) for m in art["stage2"]["B"]["lgbm"]]
    return {"s1_lgbm": gb1, "s1_xgb": xb1, "A_xgb": Ax, "A_lgbm": Al, "B_xgb": Bx, "B_lgbm": Bl}


def build_features(frame, DRIV):
    full = frame.copy()
    full["es"] = esf(full["air_temperature"])
    full["day"] = full["datetime"].dt.floor("D")
    full["slot"] = (full["datetime"].dt.hour * 2 + full["datetime"].dt.minute // 30).astype(int)
    for c in DRIV:
        g = full.groupby("day")[c]
        full[c + "_dm"] = g.transform("mean")
        full[c + "_dsd"] = g.transform("std")
        full[c + "_dmin"] = g.transform("min")
        full[c + "_dmax"] = g.transform("max")
        full[c + "_dev"] = full[c] - full[c + "_dm"]
    full["phys_ratio"] = esf(full["air_temperature_dm"]) / full["es"]
    for k in range(1, 11):
        full[f"sh{k}"] = np.sin(2 * np.pi * k * full["slot"] / 48)
        full[f"ch{k}"] = np.cos(2 * np.pi * k * full["slot"] / 48)
    full["dT"] = full["air_temperature_dev"]
    full["des"] = full["es_dev"]
    full["dT2"] = full["dT"] ** 2
    full["dTdes"] = full["dT"] * full["des"]
    for f in ["dT", "des", "soil_humidity_line2_dev", "co2_dev"]:
        for L in [1, 2, 3, 6, 12]:
            full[f"{f}_l{L}"] = full.groupby("day")[f].shift(L)
    return full


def predict_df(test_df, art, boosters):
    TARGET = art["target"]
    FE = art["feature_engineering"]
    DRIV, DAG = FE["DRIV"], FE["DAG"]
    S2, S2B = art["stage2"]["A"]["features"], art["stage2"]["B"]["features"]

    test = test_df.sort_values("datetime").reset_index(drop=True)
    ctx_rows = art.get("inference_state", {}).get("day_context", {}).get("rows", [])
    if ctx_rows:
        ctx = pd.DataFrame(ctx_rows)
        ctx["datetime"] = pd.to_datetime(ctx["datetime"])
        have = set(test["datetime"].astype(str))
        ctx = ctx[~ctx["datetime"].astype(str).isin(have)]
        frame = pd.concat([ctx, test], ignore_index=True).sort_values("datetime").reset_index(drop=True)
        n_skip = len(ctx)
    else:
        frame = test.copy()
        n_skip = 0

    full = build_features(frame, DRIV)
    DFEAT = list(dict.fromkeys(DAG + [c + s for c in DRIV for s in ["_dm", "_dsd", "_dmin", "_dmax"]]))
    dtf = full.groupby("day").agg(**{f: (f, "first") for f in DFEAT})
    dtf["doy"] = dtf.index.dayofyear

    st = art["inference_state"]
    FEATS = st["daily_median_features"]
    med = pd.Series(st["daily_median"], index=FEATS)
    Xa = dtf.reindex(columns=FEATS).fillna(med)

    s1 = art["stage1"]["ridge"]
    p_ridge = ((Xa.values - s1["scaler_mean"]) / s1["scaler_scale"]) @ s1["coef"] + s1["intercept"]
    p = (art["stage1"]["blend"]["ridge"] * p_ridge
         + art["stage1"]["blend"]["lgbm"] * boosters["s1_lgbm"].predict(Xa.values)
         + art["stage1"]["blend"]["xgb"] * boosters["s1_xgb"].predict(xgb.DMatrix(Xa.values, feature_names=FEATS)))
    base = pd.Series(0.85 * p + 0.15 * st["last10_mean"], index=dtf.index)

    test_days = list(pd.Index(test["datetime"].dt.floor("D").unique()))
    carry = art["stage1"]["carry"]
    lvl_series = base.copy()
    for k, dd in enumerate(test_days):
        if dd in lvl_series.index:
            lvl_series[dd] = base[dd] + carry["weight"] * st["carry_r0"] * (carry["rho"] ** k)
    full["lvl"] = full["day"].map(lvl_series)

    tei = list(range(n_skip, n_skip + len(test)))
    fx = full.copy()
    fx["phys_dev"] = full["lvl"] * full["phys_ratio"] - full["lvl"]
    for L in [1, 2, 3, 6, 12]:
        fx[f"phys_dev_l{L}"] = fx.groupby("day")["phys_dev"].shift(L)

    def ensemble(feat, mx, ml):
        Xe = fx[feat].fillna(0.0).values[tei]
        px = np.mean([m.predict(xgb.DMatrix(Xe, feature_names=feat)) for m in mx], axis=0)
        pl = np.mean([m.predict(Xe) for m in ml], axis=0)
        w = art["stage2"]["A"]["tree_blend_weight_xgb"]
        return w * px + (1 - w) * pl

    dev_te = art["stage2"]["amplitude"] * (
        art["stage2"]["A_B_blend_weight_A"] * ensemble(S2, boosters["A_xgb"], boosters["A_lgbm"])
        + (1 - art["stage2"]["A_B_blend_weight_A"]) * ensemble(S2B, boosters["B_xgb"], boosters["B_lgbm"]))

    lvl_te = full["lvl"].to_numpy()[tei]
    rec = art["reconstruction"]
    pred_internal = np.clip(causal_ewm(lvl_te + dev_te, rec["smoothing_alpha"]), *rec["physical_clip"])

    pm = art["post_modelling"]
    post_1 = clip100(tail_compression(pred_internal, pm["tail_quantile"], pm["tail_strength"]))
    ml3 = clip100(rolling_median_blend(
        clip100(lead_lag_center(clip100(post_1 + pm["affine_shift"]), pm["lead_lag_weight"])),
        pm["rolling_window"], pm["rolling_weight_stage2"]))
    post_2 = clip100(0.20 * ml3 + 0.80 * clip100(tail_compression(ml3, pm["tail_quantile"], pm["tail_strength"])))
    beam4 = clip100(slot_blend(
        clip100(rolling_median_blend(
            clip100(curvature(clip100(np.median(post_2) + pm["beam_amplitude"] * (post_2 - np.median(post_2))),
                              pm["curvature_strength"])),
            pm["rolling_window"], pm["rolling_weight_beam"])),
        test["datetime"], pm["slot_blend_weight"]))
    final_prediction = clip100(0.97 * beam4 + 0.03 * causal_ewm(beam4, pm["beam_ewm_alpha"]))

    preds = [{"datetime": d, "air_humidity": float(v)} for d, v in zip(
        test["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"), np.round(final_prediction, 4))]
    vals = final_prediction
    stats = {"mean": round(float(vals.mean()), 2), "min": round(float(vals.min()), 2), "max": round(float(vals.max()), 2)}
    return preds, stats
