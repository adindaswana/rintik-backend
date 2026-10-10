from __future__ import annotations

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


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROTOTYPE_DIR = BACKEND_DIR.parent
WORKSPACE_DIR = PROTOTYPE_DIR.parent
OFFICIAL_DIR = PROTOTYPE_DIR / "reference" / "official"
DEMO_FIXTURES = PROTOTYPE_DIR / "demo" / "fixtures"


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

    async def asyncSetUp(self):
        self.database = BACKEND_DIR / "runtime" / f"test-api-{uuid.uuid4().hex}.sqlite3"
        api.SERVICE = DailyBatchService(self.runtime, self.database)
        api.MONITORING = MonitoringService(
            api.SERVICE,
            DEMO_FIXTURES / "synthetic_physical_humidity_SENSOR_SIMULATION.csv",
            DEMO_FIXTURES / "SOURCE_PROVENANCE.json",
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
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.database) + suffix).unlink(missing_ok=True)

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
