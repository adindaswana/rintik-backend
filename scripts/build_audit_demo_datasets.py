"""Generate the three neutral 90-day synthetic CSVs for the "Pengawasan 90 Hari" demo.

Output: fixtures/audit_demo/rintik_demo_001.csv, _002.csv, _003.csv (+ checksums.json).
Each file: 1 Jan – 31 Mar 2026, 48 slots/day = 4 320 rows, one site.

The CSVs carry OBSERVATIONS ONLY (sensors, the P3 value per slot, pipeline technical events, and a
reference RH with its source/verification metadata). They contain no scenario name, label, score or
decision. The design intent of each file is documented for developers in docs/AUDIT_DEMO_DATASETS.md and
is never read by the backend.

Realism anchors (from train.csv): hourly temperature/CO2 profiles, daily-feature identities
(standard_day_degree = daily_mean - 10, daily_max_above_Tbase = daily_max - 10, ontario ≈ 1.567·mean
- 1.285·max + 30.57), cumulative-counter daily increments, and RH from Magnus vapour-pressure physics.
Everything here is synthetic; nothing is a field measurement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "audit_demo"
START = "2026-01-01 00:00:00"
DAYS = 90
SLOTS = 48
SITE = "SITE-GH-01"
SEEDS = {"rintik_demo_001.csv": 20260101, "rintik_demo_002.csv": 20260202, "rintik_demo_003.csv": 20260303}
STAGES = "post3"

# hourly means from train.csv (Jun–Aug 2023)
TEMP_HOURLY = [20.5, 20.2, 19.9, 18.9, 19.7, 21.9, 26.8, 29.9, 31.7, 32.8, 33.7, 33.6, 34.3, 34.8, 35.2, 34.9, 33.7, 32.1, 27.9, 25.5, 24.4, 23.2, 22.6, 21.6]
CO2_HOURLY = [517, 525, 524, 536, 530, 506, 490, 477, 467, 463, 456, 450, 448, 443, 442, 440, 440, 443, 456, 488, 499, 503, 508, 507]


def es(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def half_hour_profile(hourly):
    hourly = np.asarray(hourly, dtype=float)
    nxt = np.roll(hourly, -1)
    return np.ravel(np.column_stack([hourly, (hourly + nxt) / 2]))  # 48 values


def ar1(rng, n, phi, sd):
    out = np.empty(n)
    out[0] = rng.normal(0, sd)
    scale = sd * np.sqrt(1 - phi ** 2)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + rng.normal(0, scale)
    return out


def base_environment(rng):
    n = DAYS * SLOTS
    times = pd.date_range(START, periods=n, freq="30min")
    day = np.arange(n) // SLOTS
    slot = np.arange(n) % SLOTS
    temp_profile = half_hour_profile(TEMP_HOURLY) - np.mean(TEMP_HOURLY)
    co2_profile = half_hour_profile(CO2_HOURLY) - np.mean(CO2_HOURLY)

    daily_mean = 27.0 + ar1(rng, DAYS, 0.7, 2.4)
    amplitude = np.clip(rng.normal(1.15, 0.12, DAYS), 0.85, 1.45)
    temperature = daily_mean[day] + amplitude[day] * temp_profile[slot] + ar1(rng, n, 0.8, 0.55)

    # vapour pressure: daily level + daytime transpiration bump; RH from Magnus
    daytime = np.clip(np.sin(np.pi * (slot / 2 - 6) / 12), 0, None)
    vapour = (1.95 + ar1(rng, DAYS, 0.75, 0.16))[day] * (1 + 0.14 * daytime) + ar1(rng, n, 0.85, 0.05)
    true_rh = np.clip(100 * vapour / es(temperature), 15, 99)

    co2 = 480 + ar1(rng, DAYS, 0.6, 10)[day] + co2_profile[slot] + rng.normal(0, 11, n)
    pressure = 1009 + ar1(rng, DAYS, 0.85, 3.6)[day] + 0.8 * np.sin(4 * np.pi * slot / SLOTS) + rng.normal(0, 0.25, n)

    soil_t1 = 23.8 + 0.35 * (daily_mean[day] - 27.5) + 1.1 * np.sin(2 * np.pi * (slot / 2 - 10) / 24) + rng.normal(0, 0.12, n)
    soil_t2 = soil_t1 + 0.9 + rng.normal(0, 0.1, n)
    soil_t3 = soil_t1 + 1.0 + rng.normal(0, 0.1, n)

    irrigation = np.isin(slot, [14, 34])
    # Mean-reverting soil moisture (drying by day, irrigation pulses) so 90 days stay stationary around train means.
    soil = []
    target = np.array([26.3, 26.9, 26.9])
    level = target.copy()
    for i in range(n):
        level += 0.04 * (target - level) - np.array([0.06, 0.08, 0.10]) * (0.6 + daytime[i])
        if irrigation[i]:
            level += np.array([1.4, 1.9, 2.4]) * rng.uniform(0.85, 1.15)
        level = np.clip(level, [17, 16, 16], [36, 46, 50])
        soil.append(level + rng.normal(0, 0.06, 3))
    soil = np.asarray(soil)
    ec = np.column_stack([
        361 + 22 * (26.3 - soil[:, 0]) + rng.normal(0, 18, n),
        408 + 30 * (26.9 - soil[:, 1]) + rng.normal(0, 25, n),
        482 + 34 * (26.9 - soil[:, 2]) + rng.normal(0, 30, n),
    ])

    water = []
    for start_level, mean_inc, sd_inc in [(4000, 629, 300), (2600, 408, 200), (1500, 249, 140)]:
        daily = np.clip(rng.normal(mean_inc, sd_inc, DAYS), 0, None)
        series = np.empty(n)
        total = start_level
        for i in range(n):
            if irrigation[i]:
                total += daily[day[i]] / 2
            series[i] = total
        water.append(series)

    frame = pd.DataFrame({"datetime": times, "site_id": SITE, "env_id": 300000 + np.arange(n)})
    frame["co2"] = co2
    frame["pressure"] = pressure
    frame["air_temperature"] = temperature
    frame["env_battery"] = 100.0
    for k in range(3):
        frame[f"soil_ec_line{k + 1}"] = ec[:, k]
        frame[f"soil_humidity_line{k + 1}"] = soil[:, k]
        frame[f"soil_temp_line{k + 1}"] = [soil_t1, soil_t2, soil_t3][k]
    for k in range(3):
        frame[f"water_volume_line{k + 1}"] = water[k]

    day_stats = pd.DataFrame({"t": temperature, "d": day}).groupby("d")["t"].agg(["mean", "max"])
    dmt = day_stats["mean"].to_numpy()
    dmax = day_stats["max"].to_numpy()
    gdd = 20 + np.cumsum(np.clip(0.72 * (dmt - 10) + rng.normal(0, 1.5, DAYS), 0, None))
    reduction = np.clip(rng.normal(16.7, 2.2, DAYS), 10.5, 21.8)
    ontario = 1.567 * dmt - 1.285 * dmax + 30.566 + rng.normal(0, 0.6, DAYS)
    frame["gdd"] = gdd[day]
    frame["standard_day_degree"] = (dmt - 10)[day]
    frame["daily_mean_temperature"] = dmt[day]
    frame["daily_max_above_Tbase"] = (dmax - 10)[day]
    frame["daily_max"] = dmax[day]
    frame["daily_max_reduction"] = reduction[day]
    frame["ontario_units"] = ontario[day]
    return frame, true_rh, day, slot


def random_missing(rng, frame, share):
    for column in [c for c in frame.columns if c not in ("datetime", "site_id", "env_id", "env_battery")]:
        mask = rng.random(len(frame)) < share
        frame.loc[mask, column] = np.nan


def finish(frame, true_rh, p3, reference_mask, reference_status, rng, stage=None, status=None, error=None):
    frame["rintik_p3"] = np.clip(p3, 0, 100)
    frame["pipeline_run_id"] = "run-" + frame["datetime"].dt.strftime("%Y%m%d")
    frame["pipeline_stage"] = stage if stage is not None else STAGES
    frame["pipeline_status"] = status if status is not None else "ok"
    frame["pipeline_error"] = error if error is not None else ""
    reference = np.where(reference_mask, np.clip(true_rh + rng.normal(0, 0.9, len(frame)), 0, 100), np.nan)
    frame["rh_reference"] = reference
    frame["rh_reference_source"] = np.where(reference_mask, "SIMULATED_CALIBRATED_HYGROMETER", "")
    frame["rh_reference_status"] = np.where(reference_mask, reference_status, "")
    frame["ref_site_id"] = np.where(reference_mask, SITE, "")
    return frame


def model_error(rng, n, bias, sd):
    return bias + ar1(rng, n, 0.85, sd)


def build(name, seed):
    rng = np.random.default_rng(seed)
    frame, true_rh, day, slot = base_environment(rng)
    n = len(frame)
    window = day >= DAYS - 7
    spot_days = np.isin(day, [14, 29, 44, 59, 74])
    status = np.full(n, "verified_simulated", dtype=object)

    if name == "rintik_demo_001.csv":
        random_missing(rng, frame, 0.03)
        for d, s, value in [(32, 25, 58.9), (70, 8, -4.2)]:  # isolated temperature spikes, like the 13 in train.csv
            frame.loc[d * SLOTS + s, "air_temperature"] = value
        p3 = true_rh + model_error(rng, n, 0.3, 2.6)
        ref_mask = window | spot_days
        status[rng.random(n) < 0.02] = "pending"
        ref_mask &= rng.random(n) > 0.01
        return finish(frame, true_rh, p3, ref_mask, status, rng)

    if name == "rintik_demo_002.csv":
        random_missing(rng, frame, 0.03)
        frame.loc[51 * SLOTS + 30, "air_temperature"] = 60.1
        drift = 0.2 + 7.0 * (day / (DAYS - 1))  # gradual systematic over-estimation
        p3 = true_rh + model_error(rng, n, 0.0, 2.8) + drift
        ref_mask = window | spot_days
        status[rng.random(n) < 0.02] = "pending"
        ref_mask &= rng.random(n) > 0.01
        return finish(frame, true_rh, p3, ref_mask, status, rng)

    # rintik_demo_003.csv
    random_missing(rng, frame, 0.03)
    p3 = true_rh + model_error(rng, n, 0.2, 2.7)
    stage = np.full(n, STAGES, dtype=object)
    pstatus = np.full(n, "ok", dtype=object)
    error = np.full(n, "", dtype=object)
    # air-temperature sensor outage, days 21–27: model fed with gaps -> damped, offset P3
    outage = (day >= 20) & (day <= 26)
    frame.loc[outage & (rng.random(n) < 0.7), "air_temperature"] = np.nan
    damped = 0.45 * true_rh + 0.55 * pd.Series(true_rh).groupby(day).transform("mean").to_numpy() + 6
    p3 = np.where(outage, damped + rng.normal(0, 1.0, n), p3)
    # extreme readings, days 40–44 (two days heavy enough to fail the critical gate)
    for d, count in [(39, 5), (40, 14), (41, 14), (42, 5), (43, 5)]:
        picks = rng.choice(SLOTS, size=count, replace=False)
        idx = d * SLOTS + picks
        frame.loc[idx, "air_temperature"] = rng.choice([-5.6, 57.3, 61.8, 59.2], size=count)
        p3[idx] = np.clip(p3[idx] + rng.choice([-22, 24], size=count), 0, 100)
        co2_idx = d * SLOTS + rng.choice(SLOTS, size=3, replace=False)
        frame.loc[co2_idx, "co2"] = 3600.0
    # fertiliser change, days 60–68: soil EC shifts away from the training distribution
    shift = (day >= 59) & (day <= 67)
    for k in (1, 2, 3):
        frame.loc[shift, f"soil_ec_line{k}"] = frame.loc[shift, f"soil_ec_line{k}"] * 2.6
    # pipeline failure, days 55–56: Stage 2 time-out, no P3
    failed = (day >= 54) & (day <= 55)
    stage[failed] = "stage1"
    pstatus[failed] = "error"
    error[failed] = "STAGE2_TIMEOUT"
    p3 = np.where(failed, np.nan, p3)
    # day 64: ingest retry warning, batch still completes
    error[day == 63] = "INGEST_RETRY"
    # audit window: pressure sensor dropout on 26–27 Mar, reference hygrometer offline on 29 Mar
    dropout = (day >= 84) & (day <= 85)
    frame.loc[dropout & (rng.random(n) < 0.8), "pressure"] = np.nan
    ref_mask = (window | np.isin(day, [14, 29])) & ~(day == 87)
    status[rng.random(n) < 0.03] = "pending"
    ref_mask &= rng.random(n) > 0.01
    return finish(frame, true_rh, p3, ref_mask, status, rng, stage, pstatus, error)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for name, seed in SEEDS.items():
        frame = build(name, seed)
        numeric = [c for c in frame.columns if frame[c].dtype.kind == "f"]
        frame[numeric] = frame[numeric].round(3)
        path = OUT / name
        frame.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S", lineterminator="\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksums[name] = {"sha256": digest, "rows": len(frame), "seed": seed}
        print(f"wrote {path.name}: {len(frame)} rows sha256={digest}")
    (OUT / "checksums.json").write_text(json.dumps({
        "generator": "scripts/build_audit_demo_datasets.py", "period": "2026-01-01..2026-03-31", "slots_per_day": SLOTS,
        "synthetic": True, "field_measurement": False, "files": checksums,
        "note": "Neutral file names; no scenario labels, scores or decisions inside the CSVs. See docs/AUDIT_DEMO_DATASETS.md.",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
