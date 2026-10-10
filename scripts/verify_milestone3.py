"""Run the reproducible Milestone 3 monitoring/recovery scenario."""

from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from contextlib import closing
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROTOTYPE_DIR = BACKEND_DIR.parent
WORKSPACE_DIR = PROTOTYPE_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

import inference as inf  # noqa: E402
from daily_batch import DailyBatchService, FrozenRuntime  # noqa: E402
from monitoring import MonitoringService  # noqa: E402


def main() -> int:
    official = PROTOTYPE_DIR / "reference" / "official"
    fixtures = PROTOTYPE_DIR / "demo" / "fixtures"
    database = BACKEND_DIR / "runtime" / f"verify-m3-{uuid.uuid4().hex}.sqlite3"
    try:
        runtime = FrozenRuntime(
            WORKSPACE_DIR / "NSC26-DS011_HBD_File Data Science.pth",
            BACKEND_DIR / "artifacts" / "inference_state.json",
            official / "train.csv",
            official / "test.csv",
        )
        daily = DailyBatchService(runtime, database)
        monitoring = MonitoringService(
            daily,
            fixtures / "synthetic_physical_humidity_SENSOR_SIMULATION.csv",
            fixtures / "SOURCE_PROVENANCE.json",
        )
        test = pd.read_csv(official / "test.csv", parse_dates=["datetime"])
        test_dates = test["datetime"].dt.strftime("%Y-%m-%d")

        results = {}
        first_prediction = None
        for index, day in enumerate(runtime.ordered_dates):
            frame = test.loc[test_dates == day, inf.INPUT_COLUMNS]
            daily.ingest(day, frame, f"verify-m3-{day}")
            prediction = daily.run(day)
            if index == 0:
                first_prediction = inf.stable_json_bytes(prediction)
            results[day] = monitoring.monitor_day(day)
            if index == 7:
                daily = DailyBatchService(runtime, database)
                monitoring = MonitoringService(
                    daily,
                    fixtures / "synthetic_physical_humidity_SENSOR_SIMULATION.csv",
                    fixtures / "SOURCE_PROVENANCE.json",
                )

        outage = results["2023-09-06"]
        confirmed = monitoring.confirm_incident(
            outage["incident"]["incident_id"],
            demo_override=True,
            note="Reproducible fictional technician action.",
            idempotency_key="verify-m3-confirm-outage",
        )
        monitoring.create_demo_audit("fail", "verify-m3-audit-fail")
        restricted = monitoring.get_monitoring("2023-09-06")

        with closing(sqlite3.connect(database)) as connection:
            synthetic_rows = connection.execute(
                "SELECT COUNT(*) FROM demo_sensor_observations"
            ).fetchone()[0]
            incidents = connection.execute(
                "SELECT COUNT(*) FROM sensor_incidents"
            ).fetchone()[0]
            actions = connection.execute(
                "SELECT COUNT(*) FROM incident_actions"
            ).fetchone()[0]
            recovery_rows = connection.execute(
                "SELECT COUNT(*) FROM recovery_estimates WHERE date = '2023-09-06'"
            ).fetchone()[0]
            raw_outage_nonnull = connection.execute(
                """
                SELECT COUNT(*) FROM demo_sensor_observations
                WHERE date = '2023-09-06' AND physical_rh IS NOT NULL
                """
            ).fetchone()[0]
            model_payload_contamination = connection.execute(
                """
                SELECT COUNT(*) FROM daily_inputs
                WHERE csv_payload LIKE '%simulated_sensor_air_humidity%'
                """
            ).fetchone()[0]

        checks = {
            "model_hash": runtime.model_sha256 == inf.OFFICIAL_MODEL_SHA256,
            "m2_first_snapshot_immutable": first_prediction
            == inf.stable_json_bytes(daily.get_result(runtime.ordered_dates[0])),
            "all_dates_monitored": monitoring.status()["monitored_days"] == 16,
            "all_synthetic_rows_separate": synthetic_rows == 734
            and model_payload_contamination == 0,
            "stuck_days_suspected_only": all(
                results[day]["assessment"]["verdict"]
                == "suspected_sensor_malfunction"
                and results[day]["incident"]["status"] == "suspected"
                for day in ("2023-09-04", "2023-09-05")
            ),
            "outage_detected": outage["assessment"]["verdict"]
            == "suspected_sensor_outage",
            "repaired_labeled_simulation": results["2023-09-07"]["sensor"]["state"]
            == "simulated_repaired",
            "incident_trace_complete": incidents == 3 and actions == 1,
            "recovery_estimated_and_raw_preserved": recovery_rows == 48
            and raw_outage_nonnull == 0
            and confirmed["recovery"]["source"] == "model_estimate",
            "failed_audit_overrides_high_health": restricted["health"]["trust_score"]
            == 96.0
            and not restricted["recovery_gate"]["allowed"]
            and "accuracy_audit_failed" in restricted["recovery_gate"]["reasons"],
            "no_synthetic_accuracy_metric": "rmse" not in json.dumps(results).lower(),
        }
        report = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "model_sha256": runtime.model_sha256,
            "state_sha256": runtime.state_sha256,
            "synthetic_fixture_sha256": monitoring.fixture_sha256,
            "checks": checks,
            "evidence": {
                "monitored_days": monitoring.status()["monitored_days"],
                "synthetic_rows": synthetic_rows,
                "incident_count": incidents,
                "action_count": actions,
                "outage_recovery_rows": recovery_rows,
                "outage_raw_nonnull_rows": raw_outage_nonnull,
                "stuck_verdicts": {
                    day: results[day]["assessment"]["verdict"]
                    for day in ("2023-09-04", "2023-09-05")
                },
                "outage_verdict": outage["assessment"]["verdict"],
                "repaired_state": results["2023-09-07"]["sensor"]["state"],
                "failed_audit_gate": restricted["recovery_gate"],
            },
        }
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "PASS" else 1
    finally:
        for suffix in ("", "-wal", "-shm"):
            Path(str(database) + suffix).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
