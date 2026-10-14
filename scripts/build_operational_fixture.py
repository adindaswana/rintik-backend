"""Build the deterministic October 2026 operational-simulation fixture.

This generator creates exogenous sensor inputs only.  It does not copy or
redate official test rows and it never creates an air_humidity model input.
The equations are a transparent, bounded greenhouse-like simulation intended
for product demonstration, not a field-calibrated environmental model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20261014
START = "2026-10-14 00:00:00"
DAYS = 12
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "fixtures"
CSV_PATH = OUTPUT_DIR / "operational_2026_exogenous.csv"
PROVENANCE_PATH = OUTPUT_DIR / "operational_2026_provenance.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    timestamps = pd.date_range(START, periods=DAYS * 48, freq="30min")
    slot = np.arange(len(timestamps)) % 48
    day = np.arange(len(timestamps)) // 48
    hour = slot / 2.0

    daily_weather = np.array([0.0, 0.8, -0.4, 1.1, 0.3, -0.9, -0.2, 0.7, 1.3, 0.4, -0.5, 0.2])
    daylight = np.sin(2 * np.pi * (hour - 7.5) / 24.0)
    daylight = np.clip(daylight, -0.82, 1.0)
    air_temperature = 25.2 + 6.1 * daylight + daily_weather[day] + rng.normal(0, 0.28, len(slot))
    pressure = 1011.8 + 2.4 * np.sin(2 * np.pi * day / 6.5) - 0.11 * (air_temperature - 25) + rng.normal(0, 0.16, len(slot))
    co2 = 505 - 48 * np.maximum(daylight, 0) + 22 * np.maximum(-daylight, 0) + 7 * np.sin(2 * np.pi * day / 5) + rng.normal(0, 5, len(slot))

    irrigation_pulse = ((slot == 11) | (slot == 35)).astype(float)
    soil_humidity_1 = np.empty(len(slot))
    soil_humidity_2 = np.empty(len(slot))
    soil_humidity_3 = np.empty(len(slot))
    levels = np.array([29.4, 25.7, 27.2], dtype=float)
    soils = [soil_humidity_1, soil_humidity_2, soil_humidity_3]
    for index in range(len(slot)):
        if index and slot[index] == 0:
            levels += np.array([0.15, -0.05, 0.08])
        levels -= np.array([0.035, 0.028, 0.031]) * (0.55 + max(daylight[index], 0))
        if irrigation_pulse[index]:
            levels += np.array([1.45, 1.08, 1.22])
        levels = np.clip(levels, [22.5, 20.5, 21.5], [34.0, 31.5, 32.5])
        for channel, values in enumerate(soils):
            values[index] = levels[channel] + rng.normal(0, 0.045)

    soil_temp_1 = 23.1 + 2.1 * np.sin(2 * np.pi * (hour - 10) / 24) + 0.22 * daily_weather[day] + rng.normal(0, 0.09, len(slot))
    soil_temp_2 = soil_temp_1 + 0.65 + rng.normal(0, 0.07, len(slot))
    soil_temp_3 = soil_temp_1 + 0.95 + rng.normal(0, 0.07, len(slot))
    soil_ec_1 = 245 + 4.6 * (30 - soil_humidity_1) + rng.normal(0, 4.0, len(slot))
    soil_ec_2 = 205 + 5.0 * (27 - soil_humidity_2) + rng.normal(0, 4.5, len(slot))
    soil_ec_3 = 235 + 4.8 * (28 - soil_humidity_3) + rng.normal(0, 4.2, len(slot))

    water_1 = 38100 - day * 35 - np.cumsum(irrigation_pulse) * 4.8
    water_2 = 24900 - day * 27 - np.cumsum(irrigation_pulse) * 3.9
    water_3 = 15120 - day * 19 - np.cumsum(irrigation_pulse) * 3.2

    frame = pd.DataFrame({
        "datetime": timestamps,
        "env_id": 260000 + np.arange(len(slot)),
        "co2": co2,
        "pressure": pressure,
        "air_temperature": air_temperature,
        "env_battery": 99.4 - np.arange(len(slot)) * 0.0014,
        "soil_ec_line1": soil_ec_1,
        "soil_humidity_line1": soil_humidity_1,
        "soil_temp_line1": soil_temp_1,
        "soil_ec_line2": soil_ec_2,
        "soil_humidity_line2": soil_humidity_2,
        "soil_temp_line2": soil_temp_2,
        "soil_ec_line3": soil_ec_3,
        "soil_humidity_line3": soil_humidity_3,
        "soil_temp_line3": soil_temp_3,
        "water_volume_line1": water_1,
        "water_volume_line2": water_2,
        "water_volume_line3": water_3,
    })

    daily = frame.assign(date=frame["datetime"].dt.date).groupby("date")["air_temperature"]
    day_mean = daily.transform("mean")
    day_max = daily.transform("max")
    frame["gdd"] = 905.0 + np.repeat(np.cumsum(np.maximum(daily.mean().to_numpy() - 10.0, 0)), 48)
    frame["standard_day_degree"] = np.maximum(day_mean - 10.0, 0)
    frame["daily_mean_temperature"] = day_mean
    frame["daily_max_above_Tbase"] = np.maximum(day_max - 10.0, 0)
    frame["daily_max"] = day_max
    frame["daily_max_reduction"] = day_max - day_mean
    frame["ontario_units"] = 0.45 * day_mean + 0.25 * day_max + 5.3

    numeric_columns = frame.columns.drop("datetime")
    frame[numeric_columns] = frame[numeric_columns].round(4)
    return frame


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = build()
    frame.to_csv(CSV_PATH, index=False, date_format="%Y-%m-%d %H:%M:%S", lineterminator="\n")
    provenance = {
        "schema_version": "rintik-operational-simulation-v1",
        "seed": SEED,
        "period": {"start": "2026-10-14", "end": "2026-10-25", "days": DAYS, "slots_per_day": 48},
        "row_count": len(frame),
        "csv_sha256": sha256(CSV_PATH),
        "source": "deterministic_synthetic_exogenous_fixture",
        "field_measurement": False,
        "field_validated_distribution": False,
        "contains_air_humidity": False,
        "model_input_only": True,
        "method": "bounded diurnal equations, daily weather offsets, correlated soil/irrigation dynamics, seeded Gaussian perturbations",
        "warning": "Synthetic operational scenario for demonstration; not October 2026 field measurements or accuracy evidence.",
    }
    PROVENANCE_PATH.write_text(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {CSV_PATH} ({len(frame)} rows, sha256={provenance['csv_sha256']})")
    print(f"wrote {PROVENANCE_PATH}")


if __name__ == "__main__":
    main()
