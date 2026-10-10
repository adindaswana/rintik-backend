"""Synthetic demo monitoring and recovery, isolated from frozen inference."""

from __future__ import annotations

import hashlib
import io
import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import inference as inf
from daily_batch import BatchError, DailyBatchService, parse_iso_date, utc_now


SYNTHETIC_FIXTURE_SHA256 = (
    "891070499559c984b0761ba04d5a705e7a52bd181fb58d033e18a519eccf63f3"
)
SYNTHETIC_SOURCE = "SYNTHETIC_DEMO_NOT_GROUND_TRUTH"
MONITORING_CONFIG_VERSION = "synthetic-monitoring-v1"
THRESHOLD_LABEL = "illustrative_unvalidated"
TRUST_LABEL = "illustrative_operational_health_not_accuracy"
FIXTURE_COLUMNS = [
    "datetime",
    "simulated_sensor_air_humidity",
    "sensor_source",
    "fault_scenario",
    "annotation",
]
EXPECTED_SCENARIOS = {
    "2023-09-04": "stuck_99_fictional",
    "2023-09-05": "stuck_99_fictional",
    "2023-09-06": "outage_missing_fictional",
}
STUCK_PHYSICAL_RANGE_MAX = 1.0
STUCK_MODEL_RANGE_MIN = 20.0
DISAGREEMENT_SIGNAL_MIN = 10.0


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def deterministic_id(prefix: str, *parts: str) -> str:
    material = "|".join(parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(material).hexdigest()[:24]}"


def evaluate_recovery_gate(
    health: dict[str, Any],
    audit: dict[str, Any],
    demo_override: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not health["critical_input_healthy"]:
        reasons.append("critical_input_unhealthy")
    if not health["pipeline_healthy"]:
        reasons.append("pipeline_unhealthy")
    if health["band"] != "good":
        reasons.append("operational_health_not_good")

    audit_status = audit["status"]
    audit_outcome = audit["outcome"]
    independent_pass = (
        audit_status == "valid_pass"
        and audit_outcome == "pass"
        and bool(audit["independent_reference"])
    )
    simulated_pass = audit_status == "simulated_demo" and audit_outcome == "pass"
    if audit_outcome == "fail" or audit_status == "valid_fail":
        reasons.append("accuracy_audit_failed")
    elif audit_outcome == "expired" or audit_status == "expired":
        reasons.append("accuracy_audit_expired")
    elif audit_outcome == "not_yet_verified" or audit_status == "not_yet_verified":
        reasons.append("accuracy_not_yet_verified")
    elif not independent_pass and not (simulated_pass and demo_override):
        reasons.append("independent_audit_or_explicit_demo_override_required")

    allowed = not reasons
    if independent_pass:
        policy_basis = "independent_valid_pass"
    elif simulated_pass and demo_override and allowed:
        policy_basis = "explicit_simulated_demo_override"
    else:
        policy_basis = "restricted"
    return {
        "allowed": allowed,
        "status": "allowed" if allowed else "restricted",
        "policy_basis": policy_basis,
        "reasons": reasons,
        "demo_override": bool(demo_override),
    }


class MonitoringService:
    def __init__(
        self,
        daily: DailyBatchService,
        fixture_path: str | Path,
        provenance_path: str | Path,
    ):
        self.daily = daily
        self.fixture_path = Path(fixture_path)
        self.provenance_path = Path(provenance_path)
        self.fixture_sha256 = inf.sha256_file(self.fixture_path)
        if self.fixture_sha256 != SYNTHETIC_FIXTURE_SHA256:
            raise inf.ArtifactIntegrityError(
                "Synthetic demo fixture does not match the pinned SHA256"
            )
        self.provenance = json.loads(self.provenance_path.read_text(encoding="utf-8"))
        self._validate_provenance()
        self.fixture = self._load_and_validate_fixture()
        self.fixture_dates = self.fixture["datetime"].dt.strftime("%Y-%m-%d")
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        return self.daily._connect()

    def _validate_provenance(self) -> None:
        if self.provenance.get("synthetic_sensor_file_sha256") != self.fixture_sha256:
            raise inf.ArtifactIntegrityError(
                "Synthetic fixture provenance hash does not match the fixture"
            )
        if self.provenance.get("real_test_csv_sha256") != inf.OFFICIAL_TEST_SHA256:
            raise inf.ArtifactIntegrityError(
                "Synthetic fixture provenance is tied to another test source"
            )
        if self.provenance.get("accuracy_ground_truth") is not False:
            raise inf.ArtifactIntegrityError(
                "Synthetic fixture must explicitly declare accuracy_ground_truth=false"
            )
        if self.provenance.get("model_input") is not False:
            raise inf.ArtifactIntegrityError(
                "Synthetic fixture must explicitly declare model_input=false"
            )

    def _load_and_validate_fixture(self) -> pd.DataFrame:
        try:
            frame = pd.read_csv(self.fixture_path)
        except Exception as exc:
            raise inf.ArtifactIntegrityError("Synthetic fixture could not be parsed") from exc
        if list(frame.columns) != FIXTURE_COLUMNS:
            raise inf.ArtifactIntegrityError("Synthetic fixture schema is not canonical")
        try:
            frame["datetime"] = pd.to_datetime(frame["datetime"], format="%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError) as exc:
            raise inf.ArtifactIntegrityError("Synthetic fixture timestamps are invalid") from exc
        if frame["datetime"].duplicated().any():
            raise inf.ArtifactIntegrityError("Synthetic fixture contains duplicate timestamps")
        if len(frame) != self.daily.runtime.sidecar["official_test"]["total_rows"]:
            raise inf.ArtifactIntegrityError("Synthetic fixture row count is not 734")
        if set(frame["sensor_source"]) != {SYNTHETIC_SOURCE}:
            raise inf.ArtifactIntegrityError("Synthetic fixture source label is invalid")

        values = pd.to_numeric(
            frame["simulated_sensor_air_humidity"], errors="coerce"
        )
        invalid_nonempty = frame["simulated_sensor_air_humidity"].notna() & values.isna()
        if invalid_nonempty.any():
            raise inf.ArtifactIntegrityError("Synthetic fixture contains non-numeric RH values")
        finite = values.dropna().map(math.isfinite)
        if not bool(finite.all()) or not bool(values.dropna().between(0, 100).all()):
            raise inf.ArtifactIntegrityError("Synthetic fixture RH values must be finite and 0-100")
        frame["physical_rh"] = values

        fixture_dates = frame["datetime"].dt.strftime("%Y-%m-%d")
        if list(dict.fromkeys(fixture_dates)) != self.daily.runtime.ordered_dates:
            raise inf.ArtifactIntegrityError("Synthetic fixture dates are not the official chronology")
        for day in self.daily.runtime.ordered_dates:
            daily = frame.loc[fixture_dates == day]
            manifest = self.daily.runtime.manifests[day]
            timestamps = daily["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
            if (
                len(daily) != manifest["row_count"]
                or timestamps.iloc[0] != manifest["first_timestamp"]
                or timestamps.iloc[-1] != manifest["last_timestamp"]
                or not bool(daily["datetime"].diff().dropna().eq(pd.Timedelta(minutes=30)).all())
            ):
                raise inf.ArtifactIntegrityError(
                    f"Synthetic fixture does not align with official timestamps for {day}"
                )
            expected_scenario = EXPECTED_SCENARIOS.get(day)
            if day >= "2023-09-07":
                expected_scenario = "simulated_repaired"
            elif expected_scenario is None:
                expected_scenario = "healthy_fictional"
            if set(daily["fault_scenario"]) != {expected_scenario}:
                raise inf.ArtifactIntegrityError(
                    f"Synthetic fixture scenario is invalid for {day}"
                )
            if expected_scenario == "outage_missing_fictional":
                if daily["physical_rh"].notna().any():
                    raise inf.ArtifactIntegrityError("Synthetic outage must contain only missing RH")
            elif daily["physical_rh"].isna().any():
                raise inf.ArtifactIntegrityError(
                    f"Synthetic non-outage day contains missing RH for {day}"
                )
        return frame

    def _initialize_database(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS demo_sensor_observations (
                    timestamp TEXT PRIMARY KEY,
                    date TEXT NOT NULL,
                    physical_rh REAL,
                    physical_source TEXT NOT NULL,
                    fault_scenario TEXT NOT NULL,
                    annotation TEXT NOT NULL,
                    fixture_sha256 TEXT NOT NULL,
                    imported_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_health_logs (
                    health_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE REFERENCES runs(run_id),
                    date TEXT NOT NULL UNIQUE,
                    input_quality REAL NOT NULL,
                    data_drift REAL NOT NULL,
                    pipeline_health REAL NOT NULL,
                    output_plausibility REAL NOT NULL,
                    trust_score REAL NOT NULL,
                    band TEXT NOT NULL,
                    critical_input_healthy INTEGER NOT NULL,
                    pipeline_healthy INTEGER NOT NULL,
                    reasons_json TEXT NOT NULL,
                    trust_label TEXT NOT NULL,
                    config_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS demo_accuracy_audits (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    audit_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    independent_reference INTEGER NOT NULL,
                    allows_automatic_recovery INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    idempotency_key TEXT UNIQUE,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS monitoring_runs (
                    monitoring_id TEXT PRIMARY KEY,
                    date TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    health_id TEXT NOT NULL REFERENCES model_health_logs(health_id),
                    audit_id_at_run TEXT NOT NULL,
                    sensor_state TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    reasons_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    thresholds_json TEXT NOT NULL,
                    fixture_sha256 TEXT NOT NULL,
                    config_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sensor_incidents (
                    incident_id TEXT PRIMARY KEY,
                    date TEXT NOT NULL UNIQUE,
                    monitoring_id TEXT NOT NULL REFERENCES monitoring_runs(monitoring_id),
                    incident_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    simulation INTEGER NOT NULL,
                    reason_codes_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS incident_actions (
                    action_id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL REFERENCES sensor_incidents(incident_id),
                    action_type TEXT NOT NULL,
                    actor_label TEXT NOT NULL,
                    simulation INTEGER NOT NULL,
                    demo_override INTEGER NOT NULL,
                    gate_status TEXT NOT NULL,
                    gate_reasons_json TEXT NOT NULL,
                    audit_id TEXT NOT NULL,
                    note TEXT,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS recovery_estimates (
                    timestamp TEXT PRIMARY KEY,
                    date TEXT NOT NULL,
                    incident_id TEXT NOT NULL REFERENCES sensor_incidents(incident_id),
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    raw_physical_rh REAL,
                    recovery_rh REAL NOT NULL,
                    recovery_source TEXT NOT NULL CHECK(recovery_source = 'model_estimate'),
                    policy_basis TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_demo_observations_date
                    ON demo_sensor_observations(date);
                CREATE INDEX IF NOT EXISTS idx_incident_actions_incident
                    ON incident_actions(incident_id);
                CREATE INDEX IF NOT EXISTS idx_recovery_date
                    ON recovery_estimates(date);
                """
            )
            initial_id = "audit_initial_simulated_demo"
            connection.execute(
                """
                INSERT OR IGNORE INTO demo_accuracy_audits (
                    audit_id, status, outcome, independent_reference,
                    allows_automatic_recovery, label, idempotency_key, created_at
                ) VALUES (?, 'simulated_demo', 'pass', 0, 0, ?, NULL, ?)
                """,
                (initial_id, "simulation_not_independent_accuracy_evidence", utc_now()),
            )
            connection.commit()

    @staticmethod
    def _validate_key(value: str | None) -> str:
        if value is None or not value.strip() or len(value) > 128:
            raise BatchError(
                "INVALID_IDEMPOTENCY_KEY",
                "Idempotency-Key must contain 1 to 128 non-blank characters.",
            )
        return value

    def _current_audit_row(self, connection: sqlite3.Connection) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM demo_accuracy_audits ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        if not row:
            raise BatchError("AUDIT_STATE_MISSING", "No audit state is available.", 500)
        return row

    @staticmethod
    def _audit_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "audit_id": row["audit_id"],
            "status": row["status"],
            "outcome": row["outcome"],
            "independent_reference": bool(row["independent_reference"]),
            "allows_automatic_recovery": bool(row["allows_automatic_recovery"]),
            "label": row["label"],
            "created_at": row["created_at"],
        }

    def get_current_audit(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            return self._audit_dict(self._current_audit_row(connection))

    def create_demo_audit(
        self, scenario: str, idempotency_key: str | None
    ) -> dict[str, Any]:
        key = self._validate_key(idempotency_key)
        allowed = {"pass", "fail", "expired", "not_yet_verified"}
        if not isinstance(scenario, str) or scenario not in allowed:
            raise BatchError(
                "INVALID_AUDIT_SCENARIO",
                f"Demo audit scenario must be one of: {', '.join(sorted(allowed))}.",
            )
        audit_id = deterministic_id("audit", scenario, key)
        now = utc_now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM demo_accuracy_audits WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if existing:
                if existing["outcome"] != scenario:
                    raise BatchError(
                        "IDEMPOTENCY_KEY_CONFLICT",
                        "Idempotency-Key is already associated with another audit scenario.",
                        409,
                    )
                result = self._audit_dict(existing)
                result["idempotent"] = True
                connection.commit()
                return result
            connection.execute(
                """
                INSERT INTO demo_accuracy_audits (
                    audit_id, status, outcome, independent_reference,
                    allows_automatic_recovery, label, idempotency_key, created_at
                ) VALUES (?, 'simulated_demo', ?, 0, 0, ?, ?, ?)
                """,
                (
                    audit_id,
                    scenario,
                    "simulated_audit_scenario_not_field_accuracy_evidence",
                    key,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM demo_accuracy_audits WHERE audit_id = ?", (audit_id,)
            ).fetchone()
            connection.commit()
        result = self._audit_dict(row)
        result["idempotent"] = False
        return result

    def _health_for_run(
        self, connection: sqlite3.Connection, day: str, run_id: str
    ) -> dict[str, Any]:
        payload = connection.execute(
            "SELECT csv_payload FROM daily_inputs WHERE date = ?", (day,)
        ).fetchone()["csv_payload"]
        frame = pd.read_csv(io.StringIO(payload))
        critical_input_healthy = True
        critical_missing_counts: dict[str, int] = {}
        for column in inf.CRITICAL_FEATURES:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            critical_missing_counts[column] = int((~np.isfinite(values)).sum())
            if not bool(np.isfinite(values).any()):
                critical_input_healthy = False
        output_rows = connection.execute(
            """
            SELECT p.post_3 FROM published_predictions p
            JOIN runs r ON r.run_id = p.run_id WHERE r.date = ? ORDER BY p.timestamp
            """,
            (day,),
        ).fetchall()
        outputs = np.asarray([row["post_3"] for row in output_rows], dtype=float)
        pipeline_healthy = len(outputs) == self.daily.runtime.manifests[day]["row_count"]
        output_plausible = bool(
            pipeline_healthy
            and np.isfinite(outputs).all()
            and np.logical_and(outputs >= 0, outputs <= 100).all()
        )
        q = 100.0 if critical_input_healthy else 40.0
        d = 90.0
        p = 100.0 if pipeline_healthy else 0.0
        o = 90.0 if output_plausible else 0.0
        trust = round(0.40 * q + 0.30 * d + 0.20 * p + 0.10 * o, 4)
        band = "good" if trust >= 85 else "caution" if trust >= 70 else "low"
        reasons = ["drift_component_uses_unvalidated_demo_baseline"]
        if any(critical_missing_counts.values()):
            reasons.append("partial_missing_values_are_native_official_fixture")
        if not critical_input_healthy:
            reasons.append("critical_input_unhealthy")
        if not pipeline_healthy:
            reasons.append("pipeline_incomplete")
        if not output_plausible:
            reasons.append("output_not_plausible")
        return {
            "health_id": deterministic_id("health", run_id, MONITORING_CONFIG_VERSION),
            "run_id": run_id,
            "date": day,
            "components": {
                "input_quality": q,
                "data_drift": d,
                "pipeline_health": p,
                "output_plausibility": o,
            },
            "trust_score": trust,
            "band": band,
            "critical_input_healthy": critical_input_healthy,
            "pipeline_healthy": pipeline_healthy,
            "reasons": reasons,
            "label": TRUST_LABEL,
            "config_version": MONITORING_CONFIG_VERSION,
        }

    @staticmethod
    def _health_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "health_id": row["health_id"],
            "run_id": row["run_id"],
            "date": row["date"],
            "components": {
                "input_quality": row["input_quality"],
                "data_drift": row["data_drift"],
                "pipeline_health": row["pipeline_health"],
                "output_plausibility": row["output_plausibility"],
            },
            "trust_score": row["trust_score"],
            "band": row["band"],
            "critical_input_healthy": bool(row["critical_input_healthy"]),
            "pipeline_healthy": bool(row["pipeline_healthy"]),
            "reasons": json.loads(row["reasons_json"]),
            "label": row["trust_label"],
            "config_version": row["config_version"],
        }

    def monitor_day(self, day: str) -> dict[str, Any]:
        parse_iso_date(day)
        model_result = self.daily.get_result(day)
        fixture_day = self.fixture.loc[self.fixture_dates == day].copy()
        if fixture_day.empty:
            raise BatchError(
                "UNSUPPORTED_SCENARIO_DATE",
                "No synthetic demo fixture exists for this date.",
                404,
            )
        physical = fixture_day["physical_rh"].to_numpy(dtype=float)
        model = np.asarray(
            [row["post_3"] for row in model_result["predictions"]], dtype=float
        )
        if len(physical) != len(model):
            raise BatchError(
                "MONITORING_ALIGNMENT_ERROR",
                "Synthetic observations do not align with the model publication.",
                500,
            )
        present = np.isfinite(physical)
        present_count = int(present.sum())
        physical_range = float(np.ptp(physical[present])) if present_count else None
        model_range = float(np.ptp(model))
        mean_disagreement = (
            float(np.mean(np.abs(physical[present] - model[present])))
            if present_count
            else None
        )
        scenario = str(fixture_day["fault_scenario"].iloc[0])
        reasons: list[str] = []
        if present_count == 0:
            sensor_state = "missing_simulated"
            verdict = "suspected_sensor_outage"
            reasons.append("physical_rh_all_missing")
        elif (
            physical_range is not None
            and physical_range <= STUCK_PHYSICAL_RANGE_MAX
            and model_range >= STUCK_MODEL_RANGE_MIN
        ):
            sensor_state = "stuck_simulated"
            verdict = "suspected_sensor_malfunction"
            reasons.extend(["physical_series_nearly_flat", "model_series_dynamic"])
            if mean_disagreement is not None and mean_disagreement >= DISAGREEMENT_SIGNAL_MIN:
                reasons.append("large_mean_disagreement")
        elif scenario == "simulated_repaired":
            sensor_state = "simulated_repaired"
            verdict = "monitoring_repaired_simulation"
            reasons.append("fixture_marks_simulated_repair")
        else:
            sensor_state = "available_simulated"
            verdict = "monitoring"

        thresholds = {
            "label": THRESHOLD_LABEL,
            "stuck_physical_range_max_rh": STUCK_PHYSICAL_RANGE_MAX,
            "stuck_model_range_min_rh": STUCK_MODEL_RANGE_MIN,
            "mean_disagreement_signal_min_rh": DISAGREEMENT_SIGNAL_MIN,
            "accuracy_interpretation": "not_an_accuracy_metric",
        }
        metrics = {
            "row_count": len(model),
            "physical_present_count": present_count,
            "physical_range_rh": physical_range,
            "model_range_rh": model_range,
            "mean_absolute_disagreement_rh": mean_disagreement,
            "interpretation": "illustrative_conflict_signal_not_accuracy",
        }
        run_id = model_result["run_id"]
        monitoring_id = deterministic_id(
            "monitor", day, run_id, self.fixture_sha256, MONITORING_CONFIG_VERSION
        )
        now = utc_now()

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT monitoring_id FROM monitoring_runs WHERE date = ?", (day,)
            ).fetchone()
            if existing:
                connection.commit()
                return self.get_monitoring(day, idempotent=True)

            health = self._health_for_run(connection, day, run_id)
            connection.execute(
                """
                INSERT INTO model_health_logs (
                    health_id, run_id, date, input_quality, data_drift,
                    pipeline_health, output_plausibility, trust_score, band,
                    critical_input_healthy, pipeline_healthy, reasons_json,
                    trust_label, config_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    health["health_id"], run_id, day,
                    health["components"]["input_quality"],
                    health["components"]["data_drift"],
                    health["components"]["pipeline_health"],
                    health["components"]["output_plausibility"],
                    health["trust_score"], health["band"],
                    int(health["critical_input_healthy"]),
                    int(health["pipeline_healthy"]), stable_json(health["reasons"]),
                    health["label"], MONITORING_CONFIG_VERSION, now,
                ),
            )
            for row in fixture_day.itertuples(index=False):
                connection.execute(
                    """
                    INSERT INTO demo_sensor_observations (
                        timestamp, date, physical_rh, physical_source,
                        fault_scenario, annotation, fixture_sha256, imported_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.datetime.strftime("%Y-%m-%d %H:%M:%S"), day,
                        None if pd.isna(row.physical_rh) else float(row.physical_rh),
                        row.sensor_source, row.fault_scenario, row.annotation,
                        self.fixture_sha256, now,
                    ),
                )
            audit = self._current_audit_row(connection)
            connection.execute(
                """
                INSERT INTO monitoring_runs (
                    monitoring_id, date, run_id, health_id, audit_id_at_run,
                    sensor_state, verdict, reasons_json, metrics_json,
                    thresholds_json, fixture_sha256, config_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    monitoring_id, day, run_id, health["health_id"], audit["audit_id"],
                    sensor_state, verdict, stable_json(reasons), stable_json(metrics),
                    stable_json(thresholds), self.fixture_sha256,
                    MONITORING_CONFIG_VERSION, now,
                ),
            )
            if verdict.startswith("suspected_"):
                incident_type = (
                    "sensor_outage" if verdict == "suspected_sensor_outage"
                    else "sensor_malfunction"
                )
                incident_id = deterministic_id(
                    "incident", day, incident_type, monitoring_id
                )
                connection.execute(
                    """
                    INSERT INTO sensor_incidents (
                        incident_id, date, monitoring_id, incident_type, status,
                        simulation, reason_codes_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'suspected', 1, ?, ?, ?)
                    """,
                    (incident_id, day, monitoring_id, incident_type, stable_json(reasons), now, now),
                )
            connection.commit()
        return self.get_monitoring(day, idempotent=False)

    def _incident_dict(
        self, connection: sqlite3.Connection, row: sqlite3.Row | None
    ) -> dict[str, Any] | None:
        if row is None:
            return None
        actions = connection.execute(
            "SELECT * FROM incident_actions WHERE incident_id = ? ORDER BY created_at, action_id",
            (row["incident_id"],),
        ).fetchall()
        return {
            "incident_id": row["incident_id"],
            "date": row["date"],
            "type": row["incident_type"],
            "status": row["status"],
            "simulation": bool(row["simulation"]),
            "reason_codes": json.loads(row["reason_codes_json"]),
            "actions": [
                {
                    "action_id": action["action_id"],
                    "action_type": action["action_type"],
                    "actor_label": action["actor_label"],
                    "simulation": bool(action["simulation"]),
                    "demo_override": bool(action["demo_override"]),
                    "gate_status": action["gate_status"],
                    "gate_reasons": json.loads(action["gate_reasons_json"]),
                    "audit_id": action["audit_id"],
                    "note": action["note"],
                    "created_at": action["created_at"],
                }
                for action in actions
            ],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_monitoring(self, day: str, idempotent: bool = False) -> dict[str, Any]:
        parse_iso_date(day)
        with closing(self._connect()) as connection:
            monitor = connection.execute(
                "SELECT * FROM monitoring_runs WHERE date = ?", (day,)
            ).fetchone()
            if not monitor:
                raise BatchError(
                    "MONITORING_NOT_FOUND",
                    "No completed synthetic monitoring run exists for this date.",
                    404,
                )
            health_row = connection.execute(
                "SELECT * FROM model_health_logs WHERE health_id = ?",
                (monitor["health_id"],),
            ).fetchone()
            health = self._health_dict(health_row)
            audit = self._audit_dict(self._current_audit_row(connection))
            incident_row = connection.execute(
                "SELECT * FROM sensor_incidents WHERE date = ?", (day,)
            ).fetchone()
            incident = self._incident_dict(connection, incident_row)
            observations = connection.execute(
                """
                SELECT timestamp, physical_rh, physical_source, fault_scenario
                FROM demo_sensor_observations WHERE date = ? ORDER BY timestamp
                """,
                (day,),
            ).fetchall()
            predictions = connection.execute(
                """
                SELECT p.timestamp, p.post_3 FROM published_predictions p
                JOIN runs r ON r.run_id = p.run_id
                WHERE r.date = ? ORDER BY p.timestamp
                """,
                (day,),
            ).fetchall()
            recovery_rows = connection.execute(
                """
                SELECT timestamp, raw_physical_rh, recovery_rh,
                       recovery_source, policy_basis
                FROM recovery_estimates WHERE date = ? ORDER BY timestamp
                """,
                (day,),
            ).fetchall()
            demo_override_applied = bool(
                incident
                and any(
                    action["demo_override"] and action["gate_status"] == "allowed"
                    for action in incident["actions"]
                )
            )
            gate = evaluate_recovery_gate(health, audit, demo_override_applied)
            recovery_status = "inactive"
            if recovery_rows:
                recovery_status = (
                    "active_estimated" if gate["allowed"]
                    else "historical_estimate_restricted"
                )
            return {
                "monitoring_id": monitor["monitoring_id"],
                "date": day,
                "mode": "synthetic_demo",
                "simulation": True,
                "idempotent": idempotent,
                "config_version": monitor["config_version"],
                "provenance": {
                    "physical_source": SYNTHETIC_SOURCE,
                    "fixture_sha256": monitor["fixture_sha256"],
                    "model_input": False,
                    "accuracy_ground_truth": False,
                    "warning": "Synthetic demo sensor; not a real measurement or accuracy reference.",
                },
                "sensor": {
                    "state": monitor["sensor_state"],
                    "scenario": observations[0]["fault_scenario"],
                    "values": [
                        {
                            "timestamp": row["timestamp"],
                            "physical_rh": row["physical_rh"],
                            "source": row["physical_source"],
                        }
                        for row in observations
                    ],
                },
                "rintik": {
                    "run_id": monitor["run_id"],
                    "source": "real_original_pth",
                    "values": [
                        {"timestamp": row["timestamp"], "p3": row["post_3"]}
                        for row in predictions
                    ],
                },
                "assessment": {
                    "verdict": monitor["verdict"],
                    "reasons": json.loads(monitor["reasons_json"]),
                    "metrics": json.loads(monitor["metrics_json"]),
                    "thresholds": json.loads(monitor["thresholds_json"]),
                    "audit_id_at_monitoring": monitor["audit_id_at_run"],
                },
                "health": health,
                "audit": audit,
                "recovery_gate": gate,
                "incident": incident,
                "recovery": {
                    "status": recovery_status,
                    "source": "model_estimate" if recovery_rows else None,
                    "values": [dict(row) for row in recovery_rows],
                },
            }

    def list_incidents(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM sensor_incidents ORDER BY date, incident_id"
            ).fetchall()
            return [self._incident_dict(connection, row) for row in rows]

    def confirm_incident(
        self,
        incident_id: str,
        demo_override: bool,
        note: str | None,
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        key = self._validate_key(idempotency_key)
        if note is not None and len(note) > 500:
            raise BatchError("INVALID_NOTE", "Confirmation note is limited to 500 characters.")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            incident = connection.execute(
                "SELECT * FROM sensor_incidents WHERE incident_id = ?", (incident_id,)
            ).fetchone()
            if not incident:
                raise BatchError("INCIDENT_NOT_FOUND", "Sensor incident was not found.", 404)
            existing = connection.execute(
                "SELECT * FROM incident_actions WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if existing:
                if existing["incident_id"] != incident_id:
                    raise BatchError(
                        "IDEMPOTENCY_KEY_CONFLICT",
                        "Idempotency-Key is already associated with another incident.",
                        409,
                    )
                if (
                    bool(existing["demo_override"]) != bool(demo_override)
                    or existing["note"] != note
                ):
                    raise BatchError(
                        "IDEMPOTENCY_KEY_CONFLICT",
                        "Idempotency-Key is already associated with another confirmation payload.",
                        409,
                    )
                connection.commit()
                result = self.get_monitoring(incident["date"], idempotent=True)
                result["confirmation_idempotent"] = True
                return result

            health_row = connection.execute(
                "SELECT * FROM model_health_logs WHERE date = ?", (incident["date"],)
            ).fetchone()
            health = self._health_dict(health_row)
            audit = self._audit_dict(self._current_audit_row(connection))
            gate = evaluate_recovery_gate(health, audit, bool(demo_override))
            action_id = deterministic_id("action", incident_id, key)
            now = utc_now()
            connection.execute(
                """
                INSERT INTO incident_actions (
                    action_id, incident_id, action_type, actor_label, simulation,
                    demo_override, gate_status, gate_reasons_json, audit_id,
                    note, idempotency_key, created_at
                ) VALUES (?, ?, 'confirm_sensor_failure', 'simulated_technician_action', 1,
                          ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action_id, incident_id, int(bool(demo_override)), gate["status"],
                    stable_json(gate["reasons"]), audit["audit_id"], note, key, now,
                ),
            )
            connection.execute(
                """
                UPDATE sensor_incidents
                SET status = 'confirmed_simulation', updated_at = ?
                WHERE incident_id = ?
                """,
                (now, incident_id),
            )
            if gate["allowed"]:
                rows = connection.execute(
                    """
                    SELECT p.timestamp, p.post_3, o.physical_rh, r.run_id
                    FROM published_predictions p
                    JOIN runs r ON r.run_id = p.run_id
                    JOIN demo_sensor_observations o ON o.timestamp = p.timestamp
                    WHERE r.date = ? ORDER BY p.timestamp
                    """,
                    (incident["date"],),
                ).fetchall()
                for row in rows:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO recovery_estimates (
                            timestamp, date, incident_id, run_id, raw_physical_rh,
                            recovery_rh, recovery_source, policy_basis, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, 'model_estimate', ?, ?)
                        """,
                        (
                            row["timestamp"], incident["date"], incident_id,
                            row["run_id"], row["physical_rh"], row["post_3"],
                            gate["policy_basis"], now,
                        ),
                    )
            connection.commit()
        result = self.get_monitoring(incident["date"], idempotent=False)
        result["confirmation_idempotent"] = False
        return result

    def status(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            monitored = connection.execute("SELECT COUNT(*) FROM monitoring_runs").fetchone()[0]
            incidents = connection.execute("SELECT COUNT(*) FROM sensor_incidents").fetchone()[0]
            recovery = connection.execute("SELECT COUNT(*) FROM recovery_estimates").fetchone()[0]
        return {
            "monitoring_ready": True,
            "monitoring_mode": "synthetic_demo",
            "monitoring_simulation": True,
            "synthetic_fixture_sha256": self.fixture_sha256,
            "monitoring_config_version": MONITORING_CONFIG_VERSION,
            "monitored_days": monitored,
            "incident_count": incidents,
            "recovery_estimate_rows": recovery,
        }
