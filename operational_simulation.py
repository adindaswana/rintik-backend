"""Isolated October 2026 operational simulation over the frozen RINTIK runtime.

The official 2023 replay remains owned by :mod:`daily_batch`.  This service has
its own persistence namespace and an integrity-pinned synthetic exogenous
fixture.  Humidity-sensor values are generated only after inference and are
never passed into the model.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import inference as inf
from daily_batch import BatchError, FrozenRuntime, parse_iso_date, utc_now


RUN_NAMESPACE = "synthetic-operational-2026-v1"
PIPELINE_LABEL = f"{inf.PIPELINE_VERSION}+isolated-operational-simulation-v1"
START_DATE = date(2026, 10, 14)
END_DATE = date(2026, 10, 25)
ALLOWED_DATES = [
    (START_DATE + timedelta(days=offset)).isoformat()
    for offset in range((END_DATE - START_DATE).days + 1)
]
SCENARIOS = {"normal", "stuck", "deviating", "missing", "repaired"}


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _json_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class OperationalSimulationService:
    def __init__(
        self,
        runtime: FrozenRuntime,
        database_path: str | Path,
        fixture_path: str | Path,
        provenance_path: str | Path,
    ):
        self.runtime = runtime
        self.database_path = Path(database_path)
        self.fixture_path = Path(fixture_path)
        self.provenance_path = Path(provenance_path)
        self.provenance = self._load_provenance()
        self.fixture = self._load_fixture()
        self.fixture_sha256 = inf.sha256_file(self.fixture_path)
        self._initialize_database()

    def _connect(self):
        import sqlite3

        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _load_provenance(self) -> dict[str, Any]:
        try:
            value = json.loads(self.provenance_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise inf.ArtifactIntegrityError("Operational simulation provenance is unavailable") from exc
        required = {
            "schema_version", "seed", "period", "row_count", "csv_sha256",
            "source", "field_measurement", "contains_air_humidity",
        }
        if not required.issubset(value):
            raise inf.ArtifactIntegrityError("Operational simulation provenance is incomplete")
        if value["field_measurement"] is not False or value["contains_air_humidity"] is not False:
            raise inf.ArtifactIntegrityError("Operational fixture provenance violates simulation boundaries")
        return value

    def _load_fixture(self) -> pd.DataFrame:
        if not self.fixture_path.is_file():
            raise inf.ArtifactIntegrityError("Operational simulation fixture is unavailable")
        if inf.sha256_file(self.fixture_path) != self.provenance["csv_sha256"]:
            raise inf.ArtifactIntegrityError("Operational simulation fixture SHA256 mismatch")
        frame = inf.canonical_input_frame(pd.read_csv(self.fixture_path))
        dates = frame["datetime"].dt.strftime("%Y-%m-%d")
        if len(frame) != 576 or sorted(dates.unique()) != ALLOWED_DATES:
            raise inf.ArtifactIntegrityError("Operational simulation fixture period is invalid")
        if any(int((dates == day).sum()) != 48 for day in ALLOWED_DATES):
            raise inf.ArtifactIntegrityError("Operational simulation fixture must contain 48 slots per day")
        return frame

    def _initialize_database(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS operational_readings (
                    timestamp TEXT PRIMARY KEY,
                    date TEXT NOT NULL,
                    source TEXT NOT NULL CHECK(source IN ('manual', 'synthetic_fixture')),
                    row_json TEXT NOT NULL,
                    validation_status TEXT NOT NULL CHECK(validation_status = 'valid'),
                    fixture_sha256 TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_operational_readings_date
                    ON operational_readings(date);

                CREATE TABLE IF NOT EXISTS operational_runs (
                    run_id TEXT PRIMARY KEY,
                    date TEXT NOT NULL UNIQUE,
                    payload_sha256 TEXT NOT NULL,
                    model_sha256 TEXT NOT NULL,
                    state_sha256 TEXT NOT NULL,
                    pipeline_version TEXT NOT NULL,
                    window_mode TEXT NOT NULL,
                    window_start_date TEXT NOT NULL,
                    window_end_date TEXT NOT NULL,
                    window_date_count INTEGER NOT NULL,
                    window_row_count INTEGER NOT NULL,
                    manual_count INTEGER NOT NULL,
                    fixture_count INTEGER NOT NULL,
                    humidity_scenario TEXT NOT NULL,
                    historical_max_abs_diff REAL NOT NULL,
                    completed_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS operational_internal_predictions (
                    timestamp TEXT PRIMARY KEY,
                    date TEXT NOT NULL,
                    pred_internal REAL NOT NULL,
                    first_run_id TEXT NOT NULL REFERENCES operational_runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS operational_published_predictions (
                    run_id TEXT NOT NULL REFERENCES operational_runs(run_id),
                    timestamp TEXT NOT NULL,
                    date TEXT NOT NULL,
                    pred_internal REAL NOT NULL,
                    post_1 REAL NOT NULL,
                    post_2 REAL NOT NULL,
                    post_3 REAL NOT NULL,
                    air_humidity REAL NOT NULL,
                    PRIMARY KEY (run_id, timestamp)
                );

                CREATE TABLE IF NOT EXISTS operational_humidity_sensor (
                    timestamp TEXT PRIMARY KEY,
                    date TEXT NOT NULL,
                    physical_rh REAL,
                    source TEXT NOT NULL,
                    scenario TEXT NOT NULL,
                    state TEXT NOT NULL,
                    recovery_rh REAL,
                    recovery_source TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            connection.commit()

    @staticmethod
    def _expected_slots(day: str) -> list[str]:
        start = pd.Timestamp(f"{day} 00:00:00")
        return [value.strftime("%Y-%m-%d %H:%M:%S") for value in pd.date_range(start, periods=48, freq="30min")]

    @staticmethod
    def _validate_date(day: str) -> None:
        parse_iso_date(day)
        if day not in ALLOWED_DATES:
            raise BatchError(
                "UNSUPPORTED_OPERATIONAL_DATE",
                "Operational simulation supports 14–25 October 2026 only.",
            )

    @staticmethod
    def _validate_ranges(row: pd.Series) -> None:
        checks = {
            "env_id": (1, 1_000_000_000),
            "co2": (0, 5000),
            "pressure": (850, 1100),
            "air_temperature": (-30, 70),
            "env_battery": (0, 100),
            "soil_ec_line1": (0, 5000),
            "soil_ec_line2": (0, 5000),
            "soil_ec_line3": (0, 5000),
            "soil_humidity_line1": (0, 100),
            "soil_humidity_line2": (0, 100),
            "soil_humidity_line3": (0, 100),
            "soil_temp_line1": (-30, 70),
            "soil_temp_line2": (-30, 70),
            "soil_temp_line3": (-30, 70),
            "water_volume_line1": (0, 1_000_000),
            "water_volume_line2": (0, 1_000_000),
            "water_volume_line3": (0, 1_000_000),
            "gdd": (0, 100_000),
            "standard_day_degree": (0, 100),
            "daily_mean_temperature": (-30, 70),
            "daily_max_above_Tbase": (0, 100),
            "daily_max": (-30, 70),
            "daily_max_reduction": (0, 100),
            "ontario_units": (-100, 200),
        }
        for key, (lower, upper) in checks.items():
            value = float(row[key])
            if not lower <= value <= upper:
                raise BatchError(
                    "VALUE_OUT_OF_RANGE",
                    f"{key} must be between {lower} and {upper} for this simulation.",
                )
        if not float(row["env_id"]).is_integer():
            raise BatchError("INVALID_IDENTIFIER", "env_id must be an integer identifier.")
        for key in inf.REQUIRED_SENSOR_COLS:
            value = float(row[key])
            if not math.isfinite(value):
                raise BatchError("NON_FINITE_VALUE", f"{key} must contain a finite number.")

    def _normalize_reading(self, day: str, reading: Any) -> pd.Series:
        self._validate_date(day)
        if not isinstance(reading, dict):
            raise BatchError("INVALID_READING", "reading must be a JSON object.")
        missing = [column for column in inf.INPUT_COLUMNS if column not in reading]
        extras = [column for column in reading if column not in inf.INPUT_COLUMNS]
        if missing:
            raise BatchError("MISSING_COLUMNS", f"Missing required columns: {', '.join(missing)}")
        if extras:
            raise BatchError("UNEXPECTED_COLUMNS", f"Unexpected columns: {', '.join(extras)}")
        try:
            frame = inf.canonical_input_frame(pd.DataFrame([reading]))
        except (TypeError, ValueError) as exc:
            raise BatchError("INVALID_READING", str(exc)) from exc
        row = frame.iloc[0]
        timestamp = row["datetime"].strftime("%Y-%m-%d %H:%M:%S")
        if timestamp not in self._expected_slots(day):
            raise BatchError(
                "INVALID_OPERATIONAL_SLOT",
                "Timestamp must be one of the 48 half-hour slots for the selected date.",
            )
        self._validate_ranges(row)
        return row

    def _next_date(self, connection) -> str | None:
        completed = {
            row["date"] for row in connection.execute("SELECT date FROM operational_runs").fetchall()
        }
        return next((day for day in ALLOWED_DATES if day not in completed), None)

    def _assert_editable_date(self, connection, day: str) -> None:
        existing = connection.execute("SELECT 1 FROM operational_runs WHERE date = ?", (day,)).fetchone()
        if existing:
            raise BatchError("OPERATIONAL_DAY_LOCKED", "This day is already processed and cannot be edited.", 409)
        expected = self._next_date(connection)
        if day != expected:
            raise BatchError(
                "OUT_OF_ORDER_OPERATIONAL_DATE",
                f"The next operational date is {expected or 'none'}; received {day}.",
                409,
            )

    @staticmethod
    def _row_payload(row: pd.Series) -> tuple[str, str]:
        payload: dict[str, Any] = {"datetime": row["datetime"].strftime("%Y-%m-%d %H:%M:%S")}
        for column in inf.REQUIRED_SENSOR_COLS:
            payload[column] = float(row[column])
        return payload["datetime"], _stable_json(payload)

    def save_manual(self, day: str, reading: Any, replace: bool = False) -> dict[str, Any]:
        row = self._normalize_reading(day, reading)
        timestamp, row_json = self._row_payload(row)
        now = utc_now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_editable_date(connection, day)
            existing = connection.execute(
                "SELECT source, row_json FROM operational_readings WHERE timestamp = ?", (timestamp,)
            ).fetchone()
            if existing and not replace:
                raise BatchError(
                    "DUPLICATE_OPERATIONAL_TIMESTAMP",
                    "This half-hour slot already exists. Use the explicit edit action to replace it.",
                    409,
                )
            if existing:
                connection.execute(
                    """
                    UPDATE operational_readings
                    SET source = 'manual', row_json = ?, validation_status = 'valid',
                        fixture_sha256 = NULL, updated_at = ?
                    WHERE timestamp = ?
                    """,
                    (row_json, now, timestamp),
                )
                action = "updated"
            else:
                connection.execute(
                    """
                    INSERT INTO operational_readings (
                        timestamp, date, source, row_json, validation_status,
                        fixture_sha256, created_at, updated_at
                    ) VALUES (?, ?, 'manual', ?, 'valid', NULL, ?, ?)
                    """,
                    (timestamp, day, row_json, now, now),
                )
                action = "created"
            connection.commit()
        return {"date": day, "timestamp": timestamp, "source": "manual", "status": "valid", "action": action, "progress": self.get_day(day)["progress"]}

    def fill_missing(self, day: str) -> dict[str, Any]:
        self._validate_date(day)
        fixture_day = self.fixture.loc[self.fixture["datetime"].dt.strftime("%Y-%m-%d") == day]
        inserted = 0
        now = utc_now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_editable_date(connection, day)
            for _, row in fixture_day.iterrows():
                timestamp, row_json = self._row_payload(row)
                exists = connection.execute(
                    "SELECT 1 FROM operational_readings WHERE timestamp = ?", (timestamp,)
                ).fetchone()
                if exists:
                    continue
                connection.execute(
                    """
                    INSERT INTO operational_readings (
                        timestamp, date, source, row_json, validation_status,
                        fixture_sha256, created_at, updated_at
                    ) VALUES (?, ?, 'synthetic_fixture', ?, 'valid', ?, ?, ?)
                    """,
                    (timestamp, day, row_json, self.fixture_sha256, now, now),
                )
                inserted += 1
            connection.commit()
        result = self.get_day(day)
        result["inserted_fixture_rows"] = inserted
        return result

    def get_fixture_reading(self, day: str, timestamp: str) -> dict[str, Any]:
        self._validate_date(day)
        try:
            parsed = pd.Timestamp(timestamp)
        except (TypeError, ValueError) as exc:
            raise BatchError("INVALID_OPERATIONAL_SLOT", "timestamp must be a valid half-hour slot.") from exc
        normalized = parsed.strftime("%Y-%m-%d %H:%M:%S")
        if normalized not in self._expected_slots(day):
            raise BatchError("INVALID_OPERATIONAL_SLOT", "timestamp is outside the selected operational date.")
        match = self.fixture.loc[
            self.fixture["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S") == normalized
        ]
        if match.empty:
            raise BatchError("FIXTURE_SLOT_NOT_FOUND", "No synthetic fixture row exists for this slot.", 404)
        _, payload = self._row_payload(match.iloc[0])
        return {
            "date": day, "timestamp": normalized, "source": "synthetic_fixture_preview",
            "saved": False, "reading": json.loads(payload), "provenance": self._provenance_summary(),
        }

    @staticmethod
    def _frame_from_rows(rows) -> pd.DataFrame:
        return inf.canonical_input_frame(pd.DataFrame([json.loads(row["row_json"]) for row in rows]))

    def get_day(self, day: str) -> dict[str, Any]:
        self._validate_date(day)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM operational_readings WHERE date = ? ORDER BY timestamp", (day,)
            ).fetchall()
            run = connection.execute("SELECT run_id FROM operational_runs WHERE date = ?", (day,)).fetchone()
        values = []
        for row in rows:
            payload = json.loads(row["row_json"])
            values.append({
                "timestamp": row["timestamp"], "source": row["source"],
                "validation_status": row["validation_status"], "values": {
                    key: payload[key] for key in inf.REQUIRED_SENSOR_COLS
                },
            })
        manual = sum(item["source"] == "manual" for item in values)
        fixture = len(values) - manual
        missing = [slot for slot in self._expected_slots(day) if slot not in {item["timestamp"] for item in values}]
        return {
            "mode": RUN_NAMESPACE,
            "simulation": True,
            "date": day,
            "locked": bool(run),
            "run_id": run["run_id"] if run else None,
            "progress": {
                "available": len(values), "required": 48, "manual": manual,
                "synthetic_fixture": fixture, "missing": len(missing), "ready": len(values) == 48,
            },
            "missing_slots": missing,
            "readings": values,
            "provenance": self._provenance_summary(),
        }

    def list_days(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            run_rows = {row["date"]: dict(row) for row in connection.execute("SELECT date, run_id, humidity_scenario, completed_at FROM operational_runs").fetchall()}
            count_rows = {row["date"]: dict(row) for row in connection.execute("SELECT date, COUNT(*) AS total, SUM(CASE WHEN source='manual' THEN 1 ELSE 0 END) AS manual FROM operational_readings GROUP BY date").fetchall()}
            next_date = self._next_date(connection)
        days = []
        for day in ALLOWED_DATES:
            counts = count_rows.get(day, {"total": 0, "manual": 0})
            run = run_rows.get(day)
            days.append({
                "date": day, "status": "completed" if run else "collecting" if counts["total"] else "empty",
                "reading_count": int(counts["total"]), "manual_count": int(counts["manual"] or 0),
                "fixture_count": int(counts["total"] - (counts["manual"] or 0)),
                "run_id": run["run_id"] if run else None,
                "humidity_scenario": run["humidity_scenario"] if run else None,
                "completed_at": run["completed_at"] if run else None,
            })
        return {
            "mode": RUN_NAMESPACE, "simulation": True, "days": days,
            "completed_days": sum(item["status"] == "completed" for item in days),
            "next_expected_date": next_date, "period_complete": next_date is None,
            "provenance": self._provenance_summary(),
        }

    def _provenance_summary(self) -> dict[str, Any]:
        return {
            "source": self.provenance["source"],
            "fixture_sha256": self.provenance["csv_sha256"],
            "seed": self.provenance["seed"],
            "field_measurement": False,
            "field_validated_distribution": False,
            "contains_air_humidity": False,
            "label": "synthetic_operational_simulation_not_field_measurement",
        }

    def _prefix(self, connection, through_date: str) -> tuple[list[str], pd.DataFrame, dict[str, int]]:
        target_index = ALLOWED_DATES.index(through_date)
        dates = ALLOWED_DATES[: target_index + 1]
        frames = []
        counts = {"manual": 0, "synthetic_fixture": 0}
        for day in dates:
            rows = connection.execute(
                "SELECT row_json, source FROM operational_readings WHERE date = ? ORDER BY timestamp", (day,)
            ).fetchall()
            if len(rows) != 48:
                raise BatchError("OPERATIONAL_DAY_INCOMPLETE", f"{day} has {len(rows)}/48 readings.", 409)
            frames.append(self._frame_from_rows(rows))
            if day == through_date:
                counts["manual"] = sum(row["source"] == "manual" for row in rows)
                counts["synthetic_fixture"] = 48 - counts["manual"]
        return dates, pd.concat(frames, ignore_index=True), counts

    @staticmethod
    def _humidity_values(model: np.ndarray, scenario: str) -> tuple[list[float | None], list[float | None], str, str]:
        index = np.arange(len(model), dtype=float)
        noise = 0.65 * np.sin(index * 0.71) + 0.22 * np.cos(index * 0.29)
        if scenario == "normal":
            physical = np.clip(model + noise, 0, 100)
            return physical.tolist(), [None] * len(model), "available_simulated", "monitoring"
        if scenario == "stuck":
            physical = np.full(len(model), round(float(np.mean(model)) + 4.0, 2))
            return physical.tolist(), [None] * len(model), "stuck_simulated", "suspected_sensor_malfunction"
        if scenario == "deviating":
            physical = np.clip(model + 12.0 + noise * 0.25, 0, 100)
            return physical.tolist(), [None] * len(model), "deviating_simulated", "suspected_sensor_malfunction"
        if scenario == "missing":
            return [None] * len(model), model.tolist(), "missing_simulated", "suspected_sensor_outage"
        physical = np.clip(model + noise * 0.7, 0, 100)
        return physical.tolist(), [None] * len(model), "simulated_repaired", "monitoring_repaired_simulation"

    def run(self, day: str, scenario: str) -> dict[str, Any]:
        self._validate_date(day)
        if scenario not in SCENARIOS:
            raise BatchError("INVALID_HUMIDITY_SCENARIO", f"Scenario must be one of: {', '.join(sorted(SCENARIOS))}.")
        with closing(self._connect()) as connection:
            existing = connection.execute("SELECT 1 FROM operational_runs WHERE date = ?", (day,)).fetchone()
            if existing:
                return self.get_result(day, idempotent=True)
            if self._next_date(connection) != day:
                raise BatchError("OUT_OF_ORDER_OPERATIONAL_DATE", "Operational dates must be processed chronologically.", 409)
            dates, prefix, counts = self._prefix(connection, day)

        internal = self.runtime.predict_internal(prefix)
        timestamps = internal["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        internal_dates = internal["datetime"].dt.strftime("%Y-%m-%d")
        historical_diff = 0.0
        with closing(self._connect()) as connection:
            archived = connection.execute("SELECT timestamp, pred_internal FROM operational_internal_predictions ORDER BY timestamp").fetchall()
        if archived:
            current = dict(zip(timestamps, internal["pred_internal"]))
            differences = [abs(row["pred_internal"] - current[row["timestamp"]]) for row in archived if row["timestamp"] in current]
            if len(differences) != len(archived):
                raise BatchError("TEMPORAL_STATE_DIVERGENCE", "An archived operational timestamp disappeared.", 500)
            historical_diff = float(max(differences, default=0.0))
            if historical_diff > 1e-10:
                raise BatchError("TEMPORAL_STATE_DIVERGENCE", f"Historical operational pred_internal changed by {historical_diff:.12g}.", 500)

        active_dates = dates[-15:]
        window = internal.loc[internal_dates.isin(active_dates)].reset_index(drop=True)
        for name, values in self.runtime.postprocess(window).items():
            window[name] = values
        window["date"] = window["datetime"].dt.strftime("%Y-%m-%d")
        published = window.loc[window["date"] == day].copy()
        published["air_humidity"] = np.round(published["post_3"], 4)
        if len(published) != 48:
            raise BatchError("PUBLICATION_ROW_MISMATCH", "Operational publication must contain 48 rows.", 500)

        day_rows = prefix.loc[prefix["datetime"].dt.strftime("%Y-%m-%d") == day]
        payload_sha = inf.canonical_frame_sha256(day_rows)
        run_material = "|".join([RUN_NAMESPACE, day, payload_sha, self.runtime.model_sha256, self.runtime.state_sha256])
        run_id = hashlib.sha256(run_material.encode("utf-8")).hexdigest()[:24]
        physical, recovery, sensor_state, verdict = self._humidity_values(published["post_3"].to_numpy(dtype=float), scenario)
        now = utc_now()

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM operational_runs WHERE date = ?", (day,)).fetchone():
                connection.commit()
                return self.get_result(day, idempotent=True)
            connection.execute(
                """
                INSERT INTO operational_runs (
                    run_id, date, payload_sha256, model_sha256, state_sha256,
                    pipeline_version, window_mode, window_start_date, window_end_date,
                    window_date_count, window_row_count, manual_count, fixture_count,
                    humidity_scenario, historical_max_abs_diff, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, day, payload_sha, self.runtime.model_sha256, self.runtime.state_sha256,
                    PIPELINE_LABEL, "expanding" if len(dates) <= 15 else "rolling_15_calendar_days",
                    active_dates[0], active_dates[-1], len(active_dates), len(window),
                    counts["manual"], counts["synthetic_fixture"], scenario, historical_diff, now,
                ),
            )
            for timestamp, value_date, value in zip(timestamps, internal_dates, internal["pred_internal"]):
                connection.execute(
                    "INSERT OR IGNORE INTO operational_internal_predictions (timestamp, date, pred_internal, first_run_id) VALUES (?, ?, ?, ?)",
                    (timestamp, value_date, float(value), run_id),
                )
            for index, row in enumerate(published.itertuples(index=False)):
                timestamp = row.datetime.strftime("%Y-%m-%d %H:%M:%S")
                connection.execute(
                    """
                    INSERT INTO operational_published_predictions (
                        run_id, timestamp, date, pred_internal, post_1, post_2, post_3, air_humidity
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (run_id, timestamp, day, float(row.pred_internal), float(row.post_1), float(row.post_2), float(row.post_3), float(row.air_humidity)),
                )
                connection.execute(
                    """
                    INSERT INTO operational_humidity_sensor (
                        timestamp, date, physical_rh, source, scenario, state,
                        recovery_rh, recovery_source, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp, day, physical[index], "SYNTHETIC_OPERATIONAL_HUMIDITY_NOT_GROUND_TRUTH",
                        scenario, sensor_state, recovery[index],
                        "model_estimate" if recovery[index] is not None else None, now,
                    ),
                )
            connection.commit()
        return self.get_result(day, idempotent=False)

    def get_result(self, day: str, idempotent: bool = False) -> dict[str, Any]:
        self._validate_date(day)
        with closing(self._connect()) as connection:
            run = connection.execute("SELECT * FROM operational_runs WHERE date = ?", (day,)).fetchone()
            if not run:
                raise BatchError("RUN_NOT_FOUND", "No completed operational simulation exists for this date.", 404)
            predictions = connection.execute(
                "SELECT timestamp, pred_internal, post_1, post_2, post_3, air_humidity FROM operational_published_predictions WHERE run_id = ? ORDER BY timestamp",
                (run["run_id"],),
            ).fetchall()
            sensors = connection.execute(
                "SELECT timestamp, physical_rh, source, scenario, state, recovery_rh, recovery_source FROM operational_humidity_sensor WHERE date = ? ORDER BY timestamp",
                (day,),
            ).fetchall()
        model = np.asarray([row["post_3"] for row in predictions], dtype=float)
        physical_present = np.asarray([row["physical_rh"] for row in sensors if row["physical_rh"] is not None], dtype=float)
        disagreement = None
        if len(physical_present) == len(model):
            disagreement = float(np.mean(np.abs(physical_present - model)))
        state = sensors[0]["state"]
        verdict = (
            "suspected_sensor_outage" if state == "missing_simulated"
            else "suspected_sensor_malfunction" if state in {"stuck_simulated", "deviating_simulated"}
            else "monitoring_repaired_simulation" if state == "simulated_repaired"
            else "monitoring"
        )
        return {
            "run_id": run["run_id"], "date": day, "status": "completed",
            "source": "real_original_pth", "simulation": True,
            "simulation_scope": RUN_NAMESPACE, "pipeline_version": run["pipeline_version"],
            "model_sha256": run["model_sha256"], "state_sha256": run["state_sha256"],
            "window": {"mode": run["window_mode"], "start_date": run["window_start_date"], "end_date": run["window_end_date"], "date_count": run["window_date_count"], "row_count": run["window_row_count"]},
            "input_composition": {"manual": run["manual_count"], "synthetic_fixture": run["fixture_count"], "total": 48},
            "historical_pred_internal_max_abs_diff": run["historical_max_abs_diff"],
            "idempotent": idempotent, "predictions": [dict(row) for row in predictions],
            "monitoring": {
                "mode": "synthetic_operational_humidity_demo", "simulation": True,
                "date": day, "sensor_state": state, "scenario": run["humidity_scenario"],
                "assessment": {
                    "verdict": verdict,
                    "metrics": {"row_count": 48, "physical_present_count": len(physical_present), "mean_absolute_disagreement_rh": disagreement, "interpretation": "illustrative_conflict_signal_not_accuracy"},
                    "thresholds": {"label": "illustrative_unvalidated"},
                },
                "sensor": {"source": "SYNTHETIC_OPERATIONAL_HUMIDITY_NOT_GROUND_TRUTH", "values": [{"timestamp": row["timestamp"], "physical_rh": row["physical_rh"]} for row in sensors]},
                "rintik": {"source": "real_original_pth", "values": [{"timestamp": row["timestamp"], "p3": row["post_3"]} for row in predictions]},
                "recovery": {"status": "active_estimated" if state == "missing_simulated" else "inactive", "source": "model_estimate" if state == "missing_simulated" else None, "values": [{"timestamp": row["timestamp"], "recovery_rh": row["recovery_rh"]} for row in sensors if row["recovery_rh"] is not None]},
                "provenance": {"physical_sensor_simulation": True, "model_input": False, "accuracy_ground_truth": False},
            },
            "provenance": self._provenance_summary(),
        }

    def get_series(self, day: str) -> dict[str, Any]:
        data = self.get_day(day)
        return {
            "schema_version": "operator-series-v1", "date": day,
            "row_count": len(data["readings"]),
            "first_timestamp": data["readings"][0]["timestamp"] if data["readings"] else None,
            "last_timestamp": data["readings"][-1]["timestamp"] if data["readings"] else None,
            "source": "synthetic_operational_exogenous", "simulation": True,
            "run_status": "completed" if data["locked"] else "collecting",
            "run_id": data["run_id"],
            "points": [{"timestamp": row["timestamp"], "source": row["source"], "values": row["values"]} for row in data["readings"]],
            "provenance": data["provenance"],
        }

    def reset(self) -> dict[str, Any]:
        tables = [
            "operational_humidity_sensor", "operational_published_predictions",
            "operational_internal_predictions", "operational_runs", "operational_readings",
        ]
        deleted: dict[str, int] = {}
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            for table in tables:
                deleted[table] = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                connection.execute(f"DELETE FROM {table}")
            connection.commit()
        return {
            "mode": RUN_NAMESPACE, "simulation": True, "status": "reset",
            "deleted_rows": deleted, "next_expected_date": ALLOWED_DATES[0],
            "model_sha256": self.runtime.model_sha256, "state_sha256": self.runtime.state_sha256,
            "fixture_sha256": self.fixture_sha256,
        }
