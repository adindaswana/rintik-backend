from __future__ import annotations

import io
import json
import uuid
import unittest
from pathlib import Path

import httpx
import pandas as pd

import app as api
import inference as inf
from daily_batch import DailyBatchService, FrozenRuntime
from monitoring import MonitoringService
from operational_simulation import OperationalSimulationService


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROTOTYPE_DIR = BACKEND_DIR.parent
WORKSPACE_DIR = PROTOTYPE_DIR.parent
OFFICIAL_DIR = PROTOTYPE_DIR / "reference" / "official"
DEMO_FIXTURES = PROTOTYPE_DIR / "demo" / "fixtures"
OPERATIONAL_FIXTURES = BACKEND_DIR / "fixtures"


class ApiTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = FrozenRuntime(
            WORKSPACE_DIR / "NSC26-DS011_HBD_File Data Science.pth",
            BACKEND_DIR / "artifacts" / "inference_state.json",
            OFFICIAL_DIR / "train.csv",
            OFFICIAL_DIR / "test.csv",
        )
        cls.test = pd.read_csv(OFFICIAL_DIR / "test.csv", parse_dates=["datetime"])
        cls.operational_fixture = pd.read_csv(
            OPERATIONAL_FIXTURES / "operational_2026_exogenous.csv"
        )

    async def asyncSetUp(self):
        self.database = BACKEND_DIR / "runtime" / f"test-api-{uuid.uuid4().hex}.sqlite3"
        api.SERVICE = DailyBatchService(self.runtime, self.database)
        api.MONITORING = MonitoringService(
            api.SERVICE,
            DEMO_FIXTURES / "synthetic_physical_humidity_SENSOR_SIMULATION.csv",
            DEMO_FIXTURES / "SOURCE_PROVENANCE.json",
        )
        api.OPERATIONAL = OperationalSimulationService(
            self.runtime,
            self.database,
            OPERATIONAL_FIXTURES / "operational_2026_exogenous.csv",
            OPERATIONAL_FIXTURES / "operational_2026_provenance.json",
        )
        api.INITIALIZATION_ERROR = None
        api.MONITORING_INITIALIZATION_ERROR = None
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        api.SERVICE = None
        api.MONITORING = None
        api.OPERATIONAL = None
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.database) + suffix).unlink(missing_ok=True)

    async def test_operational_http_contract_and_cors_are_explicit(self):
        health = await self.client.get("/health")
        self.assertEqual(
            health.json()["operational_simulation"]["mode"],
            "synthetic-operational-2026-v1",
        )

        preflight = await self.client.options(
            "/v1/operational/days/2026-10-14/readings",
            headers={
                "Origin": "http://127.0.0.1:4174",
                "Access-Control-Request-Method": "PUT",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertEqual(preflight.status_code, 200, preflight.text)
        self.assertEqual(
            preflight.headers["access-control-allow-origin"],
            "http://127.0.0.1:4174",
        )

        invalid = await self.client.post(
            "/v1/operational/days/2026-10-14/readings",
            json={"unexpected": True},
        )
        self.assertEqual(invalid.status_code, 400, invalid.text)
        self.assertEqual(invalid.json()["error"]["code"], "INVALID_BODY")

        reading = self.operational_fixture.iloc[0].to_dict()
        created = await self.client.post(
            "/v1/operational/days/2026-10-14/readings",
            json={"reading": reading},
        )
        self.assertEqual(created.status_code, 200, created.text)
        self.assertEqual(created.json()["source"], "manual")

        filled = await self.client.post("/v1/operational/days/2026-10-14/fill")
        self.assertEqual(filled.status_code, 200, filled.text)
        self.assertEqual(filled.json()["progress"]["available"], 48)
        self.assertEqual(filled.json()["progress"]["manual"], 1)

        run = await self.client.post(
            "/v1/operational/days/2026-10-14/run",
            json={"humidity_scenario": "missing"},
        )
        self.assertEqual(run.status_code, 200, run.text)
        self.assertEqual(run.json()["monitoring"]["recovery"]["status"], "active_estimated")
        self.assertTrue(
            all(row["physical_rh"] is None for row in run.json()["monitoring"]["sensor"]["values"])
        )

    async def test_daily_api_returns_explicit_real_provenance(self):
        health = await self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["ready"])
        self.assertEqual(health.json()["source"], "real_original_pth")
        self.assertFalse(health.json()["simulation"])

        first_date = self.runtime.ordered_dates[0]
        dates = self.test["datetime"].dt.strftime("%Y-%m-%d")
        day = self.test.loc[dates == first_date, inf.INPUT_COLUMNS]
        ingest = await self.client.post(
            f"/v1/days/{first_date}/ingest",
            json={"csv": day.to_csv(index=False)},
            headers={"Idempotency-Key": "api-test-day-1"},
        )
        self.assertEqual(ingest.status_code, 200, ingest.text)
        self.assertEqual(ingest.json()["row_count"], 18)

        run = await self.client.post(f"/v1/days/{first_date}/run")
        self.assertEqual(run.status_code, 200, run.text)
        body = run.json()
        self.assertEqual(body["source"], "real_original_pth")
        self.assertFalse(body["simulation"])
        self.assertEqual(body["window"]["mode"], "expanding")
        self.assertEqual(len(body["predictions"]), 18)
        self.assertIn("pred_internal", body["predictions"][0])
        self.assertIn("post_3", body["predictions"][0])

    async def test_next_official_demo_input_is_verified_and_advances_with_state(self):
        first, second = self.runtime.ordered_dates[:2]
        fixture = await self.client.get("/v1/demo/official-input/next")
        self.assertEqual(fixture.status_code, 200, fixture.text)
        body = fixture.json()
        self.assertEqual(body["date"], first)
        self.assertEqual(body["row_count"], 18)
        self.assertEqual(body["source"], "official_test_csv")
        self.assertFalse(body["simulation"])
        self.assertFalse(body["prediction_produced"])
        self.assertEqual(
            body["official_test_sha256"],
            self.runtime.sidecar["sources"]["test"]["sha256"],
        )
        parsed = pd.read_csv(io.StringIO(body["csv"]))
        self.assertEqual(list(parsed.columns), inf.INPUT_COLUMNS)
        self.assertEqual(len(parsed), 18)

        ingest = await self.client.post(
            f"/v1/days/{first}/ingest",
            json={"csv": body["csv"]},
            headers={"Idempotency-Key": "demo-helper-first"},
        )
        self.assertEqual(ingest.status_code, 200, ingest.text)
        run = await self.client.post(f"/v1/days/{first}/run")
        self.assertEqual(run.status_code, 200, run.text)

        advanced = await self.client.get("/v1/demo/official-input/next")
        self.assertEqual(advanced.status_code, 200, advanced.text)
        self.assertEqual(advanced.json()["date"], second)
        self.assertEqual(advanced.json()["row_count"], 48)

    async def test_demo_reset_is_explicit_complete_and_preserves_frozen_identity(self):
        first = self.runtime.ordered_dates[0]
        fixture = (await self.client.get("/v1/demo/official-input/next")).json()
        ingested = await self.client.post(
            f"/v1/days/{first}/ingest",
            json={"csv": fixture["csv"]},
            headers={"Idempotency-Key": "reset-seed"},
        )
        self.assertEqual(ingested.status_code, 200, ingested.text)
        completed = await self.client.post(f"/v1/days/{first}/run")
        self.assertEqual(completed.status_code, 200, completed.text)
        monitored = await self.client.post(f"/v1/demo/days/{first}/monitor")
        self.assertEqual(monitored.status_code, 200, monitored.text)

        rejected = await self.client.post(
            "/v1/demo/reset", json={"confirmation": "reset"}
        )
        self.assertEqual(rejected.status_code, 400, rejected.text)
        self.assertEqual(
            rejected.json()["error"]["code"], "RESET_CONFIRMATION_REQUIRED"
        )
        self.assertEqual(api.SERVICE.status()["completed_days"], 1)

        reset = await self.client.post(
            "/v1/demo/reset", json={"confirmation": "RESET_DEMO_STATE"}
        )
        self.assertEqual(reset.status_code, 200, reset.text)
        body = reset.json()
        self.assertTrue(body["reset"])
        self.assertEqual(body["removed"]["runs"], 1)
        self.assertEqual(body["removed"]["monitoring_runs"], 1)
        self.assertEqual(body["model_sha256"], inf.OFFICIAL_MODEL_SHA256)
        self.assertEqual(body["state_sha256"], self.runtime.state_sha256)
        self.assertEqual(body["runtime"]["completed_days"], 0)
        self.assertEqual(body["runtime"]["next_expected_date"], first)
        self.assertEqual(body["monitoring"]["monitored_days"], 0)
        self.assertEqual(body["monitoring"]["incident_count"], 0)
        self.assertEqual(body["monitoring"]["recovery_estimate_rows"], 0)

        days = await self.client.get("/v1/days")
        self.assertEqual(days.status_code, 200, days.text)
        self.assertEqual(days.json()["days"], [])
        incidents = await self.client.get("/v1/demo/incidents")
        self.assertEqual(incidents.json()["incidents"], [])
        old_result = await self.client.get(f"/v1/days/{first}/result")
        self.assertEqual(old_result.status_code, 404, old_result.text)
        audit = await self.client.get("/v1/demo/audits/current")
        self.assertEqual(audit.json()["audit_id"], "audit_initial_simulated_demo")
        self.assertEqual(audit.json()["status"], "simulated_demo")

    async def test_full_compatibility_api_is_real_stateless_and_hash_gated(self):
        official = self.test.loc[:, inf.INPUT_COLUMNS]
        accepted = await self.client.post(
            "/predict",
            json={"csv": official.to_csv(index=False)},
        )
        self.assertEqual(accepted.status_code, 200, accepted.text)
        body = accepted.json()
        self.assertEqual(body["count"], 734)
        self.assertEqual(body["source"], "real_original_pth")
        self.assertFalse(body["simulation"])
        self.assertEqual(body["model_sha256"], inf.OFFICIAL_MODEL_SHA256)
        self.assertEqual(body["state_sha256"], self.runtime.state_sha256)
        self.assertEqual(body["pipeline_version"], inf.PIPELINE_VERSION)
        self.assertEqual(api.SERVICE.status()["completed_days"], 0)

        altered = official.copy()
        altered.loc[altered.index[0], "co2"] += 1
        rejected = await self.client.post(
            "/predict",
            json={"csv": altered.to_csv(index=False)},
        )
        self.assertEqual(rejected.status_code, 400, rejected.text)
        self.assertEqual(rejected.json()["error"]["code"], "OFFICIAL_FIXTURE_MISMATCH")
        self.assertNotIn("predictions", rejected.json())

    async def test_next_official_demo_input_reports_completed_scenario(self):
        real_service = api.SERVICE

        class CompletedService:
            @staticmethod
            def status():
                return {"next_expected_date": None}

        try:
            api.SERVICE = CompletedService()
            response = await self.client.get("/v1/demo/official-input/next")
        finally:
            api.SERVICE = real_service
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(response.json()["error"]["code"], "SCENARIO_COMPLETE")
        self.assertNotIn("predictions", response.json())

    async def test_unavailable_runtime_is_503_without_success_payload(self):
        api.SERVICE = None
        api.INITIALIZATION_ERROR = "test integrity failure"
        health = await self.client.get("/health")
        self.assertEqual(health.status_code, 503)
        self.assertFalse(health.json()["ready"])
        response = await self.client.post(
            "/v1/days/2023-08-29/run"
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "MODEL_UNAVAILABLE")
        self.assertNotIn("predictions", response.json())

    async def test_ingest_conflicts_use_explicit_http_409_envelopes(self):
        first, second = self.runtime.ordered_dates[:2]
        dates = self.test["datetime"].dt.strftime("%Y-%m-%d")
        first_day = self.test.loc[dates == first, inf.INPUT_COLUMNS].copy()
        second_day = self.test.loc[dates == second, inf.INPUT_COLUMNS].copy()

        accepted = await self.client.post(
            f"/v1/days/{first}/ingest",
            json={"csv": first_day.to_csv(index=False)},
            headers={"Idempotency-Key": "api-stable-key"},
        )
        self.assertEqual(accepted.status_code, 200, accepted.text)

        altered = first_day.copy()
        altered.loc[altered.index[0], "co2"] += 1
        conflict = await self.client.post(
            f"/v1/days/{first}/ingest",
            json={"csv": altered.to_csv(index=False)},
            headers={"Idempotency-Key": "api-stable-key"},
        )
        self.assertEqual(conflict.status_code, 409, conflict.text)
        self.assertEqual(conflict.json()["error"]["code"], "INPUT_CONFLICT")

        completed = await self.client.post(f"/v1/days/{first}/run")
        self.assertEqual(completed.status_code, 200, completed.text)

        key_conflict = await self.client.post(
            f"/v1/days/{second}/ingest",
            json={"csv": second_day.to_csv(index=False)},
            headers={"Idempotency-Key": "api-stable-key"},
        )
        self.assertEqual(key_conflict.status_code, 409, key_conflict.text)
        self.assertEqual(
            key_conflict.json()["error"]["code"],
            "IDEMPOTENCY_KEY_CONFLICT",
        )

    async def test_demo_monitoring_contract_is_separate_and_explicit(self):
        first = self.runtime.ordered_dates[0]
        dates = self.test["datetime"].dt.strftime("%Y-%m-%d")
        first_day = self.test.loc[dates == first, inf.INPUT_COLUMNS].copy()
        ingest = await self.client.post(
            f"/v1/days/{first}/ingest",
            json={"csv": first_day.to_csv(index=False)},
            headers={"Idempotency-Key": "api-monitor-first"},
        )
        self.assertEqual(ingest.status_code, 200, ingest.text)
        run = await self.client.post(f"/v1/days/{first}/run")
        self.assertEqual(run.status_code, 200, run.text)

        monitored = await self.client.post(f"/v1/demo/days/{first}/monitor")
        self.assertEqual(monitored.status_code, 200, monitored.text)
        body = monitored.json()
        self.assertTrue(body["simulation"])
        self.assertEqual(body["mode"], "synthetic_demo")
        self.assertFalse(body["provenance"]["model_input"])
        self.assertFalse(body["provenance"]["accuracy_ground_truth"])
        self.assertEqual(body["rintik"]["source"], "real_original_pth")
        self.assertEqual(len(body["sensor"]["values"]), 18)
        self.assertEqual(len(body["rintik"]["values"]), 18)
        self.assertEqual(body["health"]["trust_score"], 96.0)
        self.assertNotIn("rmse", json.dumps(body).lower())

        repeated = await self.client.post(f"/v1/demo/days/{first}/monitor")
        self.assertEqual(repeated.status_code, 200, repeated.text)
        self.assertTrue(repeated.json()["idempotent"])

        features = await self.client.get("/v1/features")
        self.assertEqual(features.status_code, 200, features.text)
        self.assertEqual(
            sum(len(group["features"]) for group in features.json()["groups"]),
            23,
        )
        series = await self.client.get(f"/v1/days/{first}/series")
        self.assertEqual(series.status_code, 200, series.text)
        self.assertEqual(series.json()["row_count"], 18)
        exported = await self.client.get(f"/v1/days/{first}/export.csv")
        self.assertEqual(exported.status_code, 200, exported.text)
        self.assertEqual(exported.headers["x-content-type-options"], "nosniff")
        self.assertEqual(len(exported.text.splitlines()), 19)
        self.assertIn("prediction_source", exported.text.splitlines()[0])

    async def test_monitoring_unavailable_is_503_without_fake_payload(self):
        api.MONITORING = None
        api.MONITORING_INITIALIZATION_ERROR = "test fixture integrity failure"
        health = await self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["ready"])
        self.assertFalse(health.json()["monitoring_ready"])
        self.assertEqual(health.json()["status"], "degraded")

        response = await self.client.post("/v1/demo/days/2023-08-29/monitor")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "MONITORING_UNAVAILABLE")
        self.assertNotIn("predictions", response.json())

    async def test_demo_audit_api_enforces_idempotency_payload(self):
        created = await self.client.post(
            "/v1/demo/audits",
            json={"scenario": "fail"},
            headers={"Idempotency-Key": "api-audit-conflict"},
        )
        self.assertEqual(created.status_code, 200, created.text)
        self.assertEqual(created.json()["outcome"], "fail")

        repeated = await self.client.post(
            "/v1/demo/audits",
            json={"scenario": "fail"},
            headers={"Idempotency-Key": "api-audit-conflict"},
        )
        self.assertEqual(repeated.status_code, 200, repeated.text)
        self.assertTrue(repeated.json()["idempotent"])

        conflict = await self.client.post(
            "/v1/demo/audits",
            json={"scenario": "pass"},
            headers={"Idempotency-Key": "api-audit-conflict"},
        )
        self.assertEqual(conflict.status_code, 409, conflict.text)
        self.assertEqual(conflict.json()["error"]["code"], "IDEMPOTENCY_KEY_CONFLICT")


if __name__ == "__main__":
    unittest.main(verbosity=2)
