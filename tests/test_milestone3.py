from __future__ import annotations

import json
import shutil
import sqlite3
import unittest
import uuid
from contextlib import closing
from pathlib import Path

import pandas as pd

import inference as inf
from daily_batch import BatchError, DailyBatchService, FrozenRuntime
from monitoring import (
    MONITORING_CONFIG_VERSION,
    SYNTHETIC_FIXTURE_SHA256,
    SYNTHETIC_SOURCE,
    MonitoringService,
    evaluate_recovery_gate,
)


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROTOTYPE_DIR = BACKEND_DIR.parent
WORKSPACE_DIR = PROTOTYPE_DIR.parent
OFFICIAL_DIR = PROTOTYPE_DIR / "reference" / "official"
FIXTURE_PATH = (
    PROTOTYPE_DIR
    / "demo"
    / "fixtures"
    / "synthetic_physical_humidity_SENSOR_SIMULATION.csv"
)
PROVENANCE_PATH = PROTOTYPE_DIR / "demo" / "fixtures" / "SOURCE_PROVENANCE.json"


class RecoveryGateTests(unittest.TestCase):
    def test_audit_and_health_are_independent_hard_gates(self):
        health = {
            "critical_input_healthy": True,
            "pipeline_healthy": True,
            "band": "good",
        }
        simulated = {
            "status": "simulated_demo",
            "outcome": "pass",
            "independent_reference": False,
        }
        without_override = evaluate_recovery_gate(health, simulated, False)
        self.assertFalse(without_override["allowed"])
        self.assertIn(
            "independent_audit_or_explicit_demo_override_required",
            without_override["reasons"],
        )
        with_override = evaluate_recovery_gate(health, simulated, True)
        self.assertTrue(with_override["allowed"])
        self.assertEqual(
            with_override["policy_basis"],
            "explicit_simulated_demo_override",
        )

        for outcome, reason in (
            ("fail", "accuracy_audit_failed"),
            ("expired", "accuracy_audit_expired"),
        ):
            audit = {**simulated, "outcome": outcome}
            result = evaluate_recovery_gate(health, audit, True)
            self.assertFalse(result["allowed"])
            self.assertIn(reason, result["reasons"])

        unhealthy = {**health, "critical_input_healthy": False}
        result = evaluate_recovery_gate(unhealthy, simulated, True)
        self.assertFalse(result["allowed"])
        self.assertIn("critical_input_unhealthy", result["reasons"])


class MonitoringScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = FrozenRuntime(
            WORKSPACE_DIR / "NSC26-DS011_HBD_File Data Science.pth",
            BACKEND_DIR / "artifacts" / "inference_state.json",
            OFFICIAL_DIR / "train.csv",
            OFFICIAL_DIR / "test.csv",
        )
        cls.test = pd.read_csv(OFFICIAL_DIR / "test.csv", parse_dates=["datetime"])
        cls.test_dates = cls.test["datetime"].dt.strftime("%Y-%m-%d")

    def setUp(self):
        token = uuid.uuid4().hex
        self.database = BACKEND_DIR / "runtime" / f"test-m3-{token}.sqlite3"
        self.corrupt_fixture = BACKEND_DIR / "runtime" / f"corrupt-m3-{token}.csv"
        self.corrupt_provenance = BACKEND_DIR / "runtime" / f"corrupt-m3-{token}.json"
        self.daily = DailyBatchService(self.runtime, self.database)
        self.monitoring = MonitoringService(
            self.daily, FIXTURE_PATH, PROVENANCE_PATH
        )

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.database) + suffix).unlink(missing_ok=True)
        self.corrupt_fixture.unlink(missing_ok=True)
        self.corrupt_provenance.unlink(missing_ok=True)

    def _day(self, day: str) -> pd.DataFrame:
        return self.test.loc[self.test_dates == day, inf.INPUT_COLUMNS].copy()

    def test_full_scenario_provenance_recovery_gates_and_restart(self):
        results: dict[str, dict] = {}
        first_prediction_snapshot = None
        first_monitor_row = None

        for index, day in enumerate(self.runtime.ordered_dates):
            self.daily.ingest(day, self._day(day), f"m3-input-{day}")
            prediction = self.daily.run(day)
            if index == 0:
                first_prediction_snapshot = inf.stable_json_bytes(prediction)
            result = self.monitoring.monitor_day(day)
            results[day] = result
            self.assertTrue(result["simulation"])
            self.assertEqual(result["mode"], "synthetic_demo")
            self.assertEqual(result["provenance"]["fixture_sha256"], SYNTHETIC_FIXTURE_SHA256)
            self.assertFalse(result["provenance"]["model_input"])
            self.assertFalse(result["provenance"]["accuracy_ground_truth"])
            self.assertEqual(result["rintik"]["source"], "real_original_pth")
            self.assertEqual(result["health"]["trust_score"], 96.0)
            self.assertEqual(result["health"]["band"], "good")
            self.assertEqual(
                result["assessment"]["thresholds"]["label"],
                "illustrative_unvalidated",
            )
            self.assertNotIn("rmse", json.dumps(result).lower())

            if index == 0:
                with closing(sqlite3.connect(self.database)) as connection:
                    first_monitor_row = connection.execute(
                        "SELECT * FROM monitoring_runs WHERE date = ?", (day,)
                    ).fetchone()
            if index == 7:
                self.daily = DailyBatchService(self.runtime, self.database)
                self.monitoring = MonitoringService(
                    self.daily, FIXTURE_PATH, PROVENANCE_PATH
                )

        for day in self.runtime.ordered_dates[:6]:
            self.assertEqual(results[day]["assessment"]["verdict"], "monitoring")
            self.assertIsNone(results[day]["incident"])
        for day in ("2023-09-04", "2023-09-05"):
            self.assertEqual(
                results[day]["assessment"]["verdict"],
                "suspected_sensor_malfunction",
            )
            self.assertEqual(results[day]["incident"]["status"], "suspected")
            self.assertEqual(results[day]["sensor"]["state"], "stuck_simulated")
        outage = results["2023-09-06"]
        self.assertEqual(outage["assessment"]["verdict"], "suspected_sensor_outage")
        self.assertEqual(outage["sensor"]["state"], "missing_simulated")
        self.assertTrue(
            all(row["physical_rh"] is None for row in outage["sensor"]["values"])
        )
        repaired = results["2023-09-07"]
        self.assertEqual(repaired["sensor"]["state"], "simulated_repaired")
        self.assertEqual(
            repaired["assessment"]["verdict"],
            "monitoring_repaired_simulation",
        )

        repeated = self.monitoring.monitor_day(self.runtime.ordered_dates[-1])
        self.assertTrue(repeated["idempotent"])
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM demo_sensor_observations").fetchone()[0],
                734,
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM monitoring_runs").fetchone()[0],
                16,
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM sensor_incidents").fetchone()[0],
                3,
            )
            model_payloads = connection.execute(
                "SELECT csv_payload FROM daily_inputs"
            ).fetchall()
            self.assertTrue(
                all("simulated_sensor_air_humidity" not in row[0] for row in model_payloads)
            )
            current_first_monitor_row = connection.execute(
                "SELECT * FROM monitoring_runs WHERE date = ?",
                (self.runtime.ordered_dates[0],),
            ).fetchone()
            self.assertEqual(first_monitor_row, current_first_monitor_row)

        self.assertEqual(
            first_prediction_snapshot,
            inf.stable_json_bytes(self.daily.get_result(self.runtime.ordered_dates[0])),
        )

        outage_incident = results["2023-09-06"]["incident"]["incident_id"]
        recovered = self.monitoring.confirm_incident(
            outage_incident,
            demo_override=True,
            note="Explicit fictional technician confirmation for the demo.",
            idempotency_key="confirm-outage-once",
        )
        self.assertEqual(recovered["incident"]["status"], "confirmed_simulation")
        self.assertEqual(
            recovered["incident"]["actions"][0]["actor_label"],
            "simulated_technician_action",
        )
        self.assertEqual(recovered["recovery"]["status"], "active_estimated")
        self.assertEqual(recovered["recovery"]["source"], "model_estimate")
        self.assertEqual(len(recovered["recovery"]["values"]), 48)
        self.assertTrue(
            all(row["raw_physical_rh"] is None for row in recovered["recovery"]["values"])
        )
        recovery_by_timestamp = {
            row["timestamp"]: row["recovery_rh"]
            for row in recovered["recovery"]["values"]
        }
        self.assertEqual(
            recovery_by_timestamp,
            {row["timestamp"]: row["p3"] for row in recovered["rintik"]["values"]},
        )
        retry = self.monitoring.confirm_incident(
            outage_incident,
            demo_override=True,
            note="Explicit fictional technician confirmation for the demo.",
            idempotency_key="confirm-outage-once",
        )
        self.assertTrue(retry["confirmation_idempotent"])
        with self.assertRaises(BatchError) as caught:
            self.monitoring.confirm_incident(
                outage_incident,
                demo_override=False,
                note="Changed retry payload must be rejected.",
                idempotency_key="confirm-outage-once",
            )
        self.assertEqual(caught.exception.code, "IDEMPOTENCY_KEY_CONFLICT")

        failed = self.monitoring.create_demo_audit("fail", "audit-fail-once")
        self.assertEqual(failed["outcome"], "fail")
        with self.assertRaises(BatchError) as caught:
            self.monitoring.create_demo_audit("pass", "audit-fail-once")
        self.assertEqual(caught.exception.code, "IDEMPOTENCY_KEY_CONFLICT")
        restricted = self.monitoring.get_monitoring("2023-09-06")
        self.assertEqual(restricted["health"]["trust_score"], 96.0)
        self.assertFalse(restricted["recovery_gate"]["allowed"])
        self.assertIn(
            "accuracy_audit_failed", restricted["recovery_gate"]["reasons"]
        )
        self.assertEqual(
            restricted["recovery"]["status"], "historical_estimate_restricted"
        )

        expired = self.monitoring.create_demo_audit("expired", "audit-expired-once")
        self.assertEqual(expired["outcome"], "expired")
        restricted = self.monitoring.get_monitoring("2023-09-06")
        self.assertIn(
            "accuracy_audit_expired", restricted["recovery_gate"]["reasons"]
        )

        self.monitoring.create_demo_audit("pass", "audit-pass-again")
        stuck_incident = results["2023-09-04"]["incident"]["incident_id"]
        no_override = self.monitoring.confirm_incident(
            stuck_incident,
            demo_override=False,
            note=None,
            idempotency_key="confirm-stuck-no-override",
        )
        self.assertFalse(no_override["recovery_gate"]["allowed"])
        self.assertEqual(no_override["recovery"]["values"], [])
        allowed = self.monitoring.confirm_incident(
            stuck_incident,
            demo_override=True,
            note="Second explicit simulated action after reviewing the gate.",
            idempotency_key="confirm-stuck-with-override",
        )
        self.assertEqual(allowed["recovery"]["status"], "active_estimated")
        self.assertTrue(
            all(row["raw_physical_rh"] == 99.0 for row in allowed["recovery"]["values"])
        )
        with closing(sqlite3.connect(self.database)) as connection:
            physical = connection.execute(
                "SELECT DISTINCT physical_rh FROM demo_sensor_observations WHERE date = '2023-09-04'"
            ).fetchall()
            self.assertEqual(physical, [(99.0,)])

        restarted = MonitoringService(
            DailyBatchService(self.runtime, self.database),
            FIXTURE_PATH,
            PROVENANCE_PATH,
        )
        after_restart = restarted.get_monitoring("2023-09-06")
        self.assertEqual(after_restart["recovery"], self.monitoring.get_monitoring("2023-09-06")["recovery"])
        self.assertEqual(restarted.status()["monitored_days"], 16)

    def test_fixture_integrity_and_preconditions_fail_closed(self):
        first = self.runtime.ordered_dates[0]
        with self.assertRaises(BatchError) as caught:
            self.monitoring.monitor_day(first)
        self.assertEqual(caught.exception.code, "RUN_NOT_FOUND")

        unexpected = self._day(first)
        unexpected["simulated_sensor_air_humidity"] = 50.0
        with self.assertRaises(BatchError) as caught:
            self.daily.ingest(first, unexpected, "synthetic-must-not-enter-model")
        self.assertEqual(caught.exception.code, "UNEXPECTED_COLUMNS")

        shutil.copyfile(FIXTURE_PATH, self.corrupt_fixture)
        with self.corrupt_fixture.open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaises(inf.ArtifactIntegrityError):
            MonitoringService(self.daily, self.corrupt_fixture, PROVENANCE_PATH)

        provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
        provenance["accuracy_ground_truth"] = True
        self.corrupt_provenance.write_text(
            json.dumps(provenance), encoding="utf-8"
        )
        with self.assertRaises(inf.ArtifactIntegrityError):
            MonitoringService(self.daily, FIXTURE_PATH, self.corrupt_provenance)

        self.assertEqual(self.monitoring.fixture_sha256, SYNTHETIC_FIXTURE_SHA256)
        self.assertEqual(MONITORING_CONFIG_VERSION, "synthetic-monitoring-v1")
        self.assertEqual(set(self.monitoring.fixture["sensor_source"]), {SYNTHETIC_SOURCE})


if __name__ == "__main__":
    unittest.main(verbosity=2)
