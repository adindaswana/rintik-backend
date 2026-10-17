from __future__ import annotations

import hashlib
import shutil
import tempfile
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

import app as api
import inference as inf
from daily_batch import DailyBatchService, FrozenRuntime
from monitoring import MonitoringService, explain_recovery
from operational_simulation import OperationalSimulationService
from sessions import Session, SessionRegistry, parse_session_header


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROTOTYPE_DIR = BACKEND_DIR.parent
WORKSPACE_DIR = PROTOTYPE_DIR.parent
OFFICIAL_DIR = PROTOTYPE_DIR / "reference" / "official"
DEMO_FIXTURES = PROTOTYPE_DIR / "demo" / "fixtures"
OPERATIONAL_FIXTURES = BACKEND_DIR / "fixtures"
HEADER = "X-Rintik-Session"


def new_session_id() -> str:
    return uuid.uuid4().hex


class SessionIsolationApiTests(unittest.IsolatedAsyncioTestCase):
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
        self.tmp = Path(tempfile.mkdtemp(prefix="rintik-sessions-"))
        self.database = self.tmp / "default.sqlite3"
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
        self.previous_registry = api.SESSIONS
        api.SESSIONS = SessionRegistry(self.tmp / "sessions", max_sessions=3)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        api.SERVICE = None
        api.MONITORING = None
        api.OPERATIONAL = None
        api.SESSIONS = self.previous_registry
        shutil.rmtree(self.tmp, ignore_errors=True)

    def day_csv(self, day: str) -> str:
        dates = self.test["datetime"].dt.strftime("%Y-%m-%d")
        frame = self.test.loc[dates == day, inf.INPUT_COLUMNS]
        return frame.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S")

    async def process(self, session_id: str | None, day: str) -> dict:
        headers = {HEADER: session_id} if session_id else {}
        ingest = await self.client.post(
            f"/v1/days/{day}/ingest", json={"csv": self.day_csv(day)}, headers=headers
        )
        self.assertEqual(ingest.status_code, 200, ingest.text)
        run = await self.client.post(f"/v1/days/{day}/run", headers=headers)
        self.assertEqual(run.status_code, 200, run.text)
        return run.json()

    async def health(self, session_id: str | None) -> dict:
        headers = {HEADER: session_id} if session_id else {}
        response = await self.client.get("/health", headers=headers)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    async def test_two_sessions_do_not_see_each_other(self):
        a, b = new_session_id(), new_session_id()
        first = self.runtime.ordered_dates[0]
        result_a = await self.process(a, first)
        reading = self.operational_fixture.iloc[0].to_dict()
        saved = await self.client.post(
            "/v1/operational/days/2026-10-14/readings",
            json={"reading": reading},
            headers={HEADER: a},
        )
        self.assertEqual(saved.status_code, 200, saved.text)

        health_a, health_b, health_default = (
            await self.health(a), await self.health(b), await self.health(None)
        )
        self.assertEqual(health_a["completed_days"], 1)
        self.assertEqual(health_b["completed_days"], 0)
        self.assertEqual(health_default["completed_days"], 0)
        self.assertEqual(health_b["next_expected_date"], first)
        self.assertEqual(
            health_a["operational_simulation"]["days"][0]["reading_count"], 1
        )
        self.assertEqual(
            health_b["operational_simulation"]["days"][0]["reading_count"], 0
        )

        hidden = await self.client.get(f"/v1/days/{first}/result", headers={HEADER: b})
        self.assertEqual(hidden.status_code, 404)
        listed = await self.client.get("/v1/days", headers={HEADER: b})
        self.assertEqual(listed.json()["days"], [])
        own = await self.client.get(f"/v1/days/{first}/result", headers={HEADER: a})
        self.assertEqual(own.json()["run_id"], result_a["run_id"])

        # Same model, same input => same published values in an independent session.
        result_b = await self.process(b, first)
        self.assertEqual(result_b["predictions"], result_a["predictions"])

    async def test_reset_only_clears_the_calling_session(self):
        a, b = new_session_id(), new_session_id()
        first = self.runtime.ordered_dates[0]
        for session_id in (a, b, None):
            await self.process(session_id, first)
            saved = await self.client.post(
                "/v1/operational/days/2026-10-14/readings",
                json={"reading": self.operational_fixture.iloc[0].to_dict()},
                headers={HEADER: session_id} if session_id else {},
            )
            self.assertEqual(saved.status_code, 200, saved.text)

        demo_reset = await self.client.post(
            "/v1/demo/reset", json={"confirmation": "RESET_DEMO_STATE"}, headers={HEADER: b}
        )
        self.assertEqual(demo_reset.status_code, 200, demo_reset.text)
        self.assertEqual(demo_reset.json()["removed"]["runs"], 1)
        op_reset = await self.client.post(
            "/v1/operational/reset",
            json={"confirmation": "RESET_OPERATIONAL_SIMULATION"},
            headers={HEADER: b},
        )
        self.assertEqual(op_reset.status_code, 200, op_reset.text)
        self.assertEqual(op_reset.json()["deleted_rows"]["operational_readings"], 1)

        self.assertEqual((await self.health(b))["completed_days"], 0)
        for survivor in (a, None):
            health = await self.health(survivor)
            self.assertEqual(health["completed_days"], 1)
            self.assertEqual(
                health["operational_simulation"]["days"][0]["reading_count"], 1
            )

    async def test_missing_header_is_legacy_default_session(self):
        first = self.runtime.ordered_dates[0]
        await self.process(None, first)
        self.assertEqual(api.SERVICE.status()["completed_days"], 1)
        health = await self.health(None)
        self.assertEqual(health["session"], {"id_hash": None, "scoped": False})
        self.assertEqual(health["completed_days"], 1)
        self.assertFalse((self.tmp / "sessions").exists())

    async def test_health_reports_hashed_session_never_the_raw_id(self):
        session_id = new_session_id()
        response = await self.client.get("/health", headers={HEADER: session_id})
        self.assertEqual(
            response.json()["session"],
            {"id_hash": hashlib.sha256(session_id.encode()).hexdigest()[:8], "scoped": True},
        )
        self.assertNotIn(session_id, response.text)
        files = [path.name for path in (self.tmp / "sessions").iterdir()]
        self.assertTrue(files)
        self.assertFalse(any(session_id in name for name in files))

    async def test_invalid_header_is_rejected_without_echo(self):
        for bad in ("short", "x" * 65, "has space in it here", "semi;colon-0123456789", ""):
            for method, path in (("GET", "/health"), ("GET", "/v1/days"), ("POST", "/v1/demo/reset")):
                response = await self.client.request(
                    method, path, headers={HEADER: bad}, json={"confirmation": "RESET_DEMO_STATE"}
                )
                self.assertEqual(response.status_code, 400, (bad, path, response.text))
                self.assertEqual(response.json()["error"]["code"], "INVALID_SESSION")
                if bad:
                    self.assertNotIn(bad, response.json()["error"]["message"])
        # Stateless endpoints ignore the header entirely.
        features = await self.client.get("/v1/features", headers={HEADER: "short"})
        self.assertEqual(features.status_code, 200)

    async def test_cors_preflight_allows_session_header(self):
        preflight = await self.client.options(
            "/v1/days",
            headers={
                "Origin": "http://127.0.0.1:4173",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "x-rintik-session",
            },
        )
        self.assertEqual(preflight.status_code, 200, preflight.text)
        self.assertIn("x-rintik-session", preflight.headers["access-control-allow-headers"].lower())

    async def test_lru_eviction_deletes_oldest_session_database(self):
        first = self.runtime.ordered_dates[0]
        oldest = new_session_id()
        await self.process(oldest, first)
        for _ in range(3):  # cap is 3: the oldest session must be evicted
            await self.health(new_session_id())
        self.assertEqual(api.SESSIONS.active_count(), 3)
        self.assertEqual(len(list((self.tmp / "sessions").glob("*.sqlite3"))), 3)
        revived = await self.health(oldest)
        self.assertEqual(revived["completed_days"], 0)

    async def test_core_results_survive_monitoring_outage(self):
        api.MONITORING = None
        api.MONITORING_INITIALIZATION_ERROR = "test fixture integrity failure"
        session_id = new_session_id()
        first = self.runtime.ordered_dates[0]
        await self.process(session_id, first)
        result = await self.client.get(f"/v1/days/{first}/result", headers={HEADER: session_id})
        self.assertEqual(result.status_code, 200, result.text)
        series = await self.client.get(f"/v1/days/{first}/series", headers={HEADER: session_id})
        self.assertEqual(series.status_code, 200, series.text)
        health = await self.health(session_id)
        self.assertEqual(health["status"], "degraded")
        self.assertTrue(health["ready"])
        monitor = await self.client.post(
            f"/v1/demo/days/{first}/monitor", headers={HEADER: session_id}
        )
        self.assertEqual(monitor.status_code, 503)
        self.assertEqual(monitor.json()["error"]["code"], "MONITORING_UNAVAILABLE")

    async def test_outage_recovery_explanation_and_scoped_confirmation(self):
        session_id = new_session_id()
        headers = {HEADER: session_id}
        for day in self.runtime.ordered_dates:
            await self.process(session_id, day)
            monitored = await self.client.post(f"/v1/demo/days/{day}/monitor", headers=headers)
            self.assertEqual(monitored.status_code, 200, monitored.text)
            if day == "2023-09-06":
                break
        before = monitored.json()
        self.assertEqual(before["sensor"]["state"], "missing_simulated")
        self.assertEqual(before["recovery"]["status"], "inactive")
        explanation = before["recovery"]["explanation"]
        self.assertEqual(explanation["code"], "awaiting_incident_confirmation")
        self.assertEqual(explanation["next_steps"], ["confirm_incident_with_demo_override"])
        self.assertTrue(explanation["demo_override_would_allow"])
        self.assertTrue(explanation["estimate_available"])
        self.assertFalse(explanation["raw_physical_overwritten"])
        self.assertEqual(before["recovery_gate"]["status"], "restricted")

        confirmed = await self.client.post(
            explanation["confirm_endpoint"],
            json={"demo_override": True},
            headers={**headers, "Idempotency-Key": "session-confirm-0906"},
        )
        self.assertEqual(confirmed.status_code, 200, confirmed.text)
        self.assertEqual(confirmed.json()["recovery"]["status"], "active_estimated")
        self.assertEqual(
            confirmed.json()["recovery"]["explanation"]["code"], "recovery_active_estimated"
        )
        self.assertTrue(
            all(row["raw_physical_rh"] is None for row in confirmed.json()["recovery"]["values"])
        )
        # Another visitor never sees this incident.
        other = await self.client.get("/v1/demo/incidents", headers={HEADER: new_session_id()})
        self.assertEqual(other.json()["incidents"], [])


class SessionRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = FrozenRuntime(
            WORKSPACE_DIR / "NSC26-DS011_HBD_File Data Science.pth",
            BACKEND_DIR / "artifacts" / "inference_state.json",
        )

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rintik-registry-"))
        daily = DailyBatchService(self.runtime, self.tmp / "default.sqlite3")
        self.base = Session(scoped=False, id_hash=None, daily=daily, monitoring=None, operational=None)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_header_validation(self):
        self.assertIsNone(parse_session_header(None))
        self.assertEqual(parse_session_header("a" * 16), "a" * 16)
        self.assertEqual(parse_session_header("A-b_9" * 12 + "abcd"), "A-b_9" * 12 + "abcd")

    def test_concurrent_creation_returns_one_session(self):
        registry = SessionRegistry(self.tmp / "sessions", max_sessions=8)
        session_id = new_session_id()
        barrier = threading.Barrier(16)

        def grab(_):
            barrier.wait()
            session = registry.acquire(session_id, self.base)
            registry.release(session)
            return id(session)

        with ThreadPoolExecutor(max_workers=16) as pool:
            identities = set(pool.map(grab, range(16)))
        self.assertEqual(len(identities), 1)
        self.assertEqual(registry.active_count(), 1)

        ids = [new_session_id() for _ in range(24)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda sid: registry.release(registry.acquire(sid, self.base)), ids))
        self.assertEqual(registry.active_count(), 8)
        self.assertEqual(len(list((self.tmp / "sessions").glob("*.sqlite3"))), 8)

    def test_session_in_use_is_not_evicted_until_released(self):
        registry = SessionRegistry(self.tmp / "sessions", max_sessions=1)
        busy = registry.acquire(new_session_id(), self.base)
        other = registry.acquire(new_session_id(), self.base)
        self.assertTrue(busy.database_path.exists())
        self.assertEqual(registry.active_count(), 2)
        registry.release(other)
        registry.release(busy)
        self.assertTrue(busy.database_path.exists())
        newest = registry.acquire(new_session_id(), self.base)
        registry.release(newest)
        self.assertEqual(registry.active_count(), 1)
        self.assertFalse(busy.database_path.exists())
        self.assertFalse(other.database_path.exists())
        self.assertTrue(newest.database_path.exists())

    def test_existing_databases_are_adopted_after_restart(self):
        first = SessionRegistry(self.tmp / "sessions", max_sessions=2)
        kept = new_session_id()
        first.release(first.acquire(kept, self.base))
        restarted = SessionRegistry(self.tmp / "sessions", max_sessions=2)
        restarted.release(restarted.acquire(new_session_id(), self.base))
        restarted.release(restarted.acquire(new_session_id(), self.base))
        self.assertEqual(len(list((self.tmp / "sessions").glob("*.sqlite3"))), 2)


class RecoveryExplanationTests(unittest.TestCase):
    health_good = {"critical_input_healthy": True, "pipeline_healthy": True, "band": "good"}
    simulated_pass = {"status": "simulated_demo", "outcome": "pass", "independent_reference": False}
    incident = {"incident_id": "incident_x", "status": "suspected"}

    def test_no_incident_needs_no_action(self):
        result = explain_recovery(self.health_good, self.simulated_pass, None, False, "inactive", True)
        self.assertEqual(result["code"], "no_sensor_incident")
        self.assertEqual(result["next_steps"], [])
        self.assertIsNone(result["confirm_endpoint"])

    def test_failed_audit_requires_new_audit_then_override(self):
        failed = {"status": "simulated_demo", "outcome": "fail", "independent_reference": False}
        result = explain_recovery(self.health_good, failed, self.incident, False, "inactive", True)
        self.assertEqual(result["code"], "blocked_by_accuracy_audit")
        self.assertEqual(
            result["next_steps"],
            ["record_passing_demo_audit", "confirm_incident_with_demo_override"],
        )
        self.assertFalse(result["demo_override_would_allow"])
        self.assertIn("accuracy_audit_failed", result["blocking_reasons"])

    def test_unhealthy_pipeline_is_not_user_fixable(self):
        unhealthy = {"critical_input_healthy": False, "pipeline_healthy": True, "band": "poor"}
        result = explain_recovery(unhealthy, self.simulated_pass, self.incident, False, "inactive", True)
        self.assertEqual(result["code"], "blocked_by_operational_health")
        self.assertEqual(result["next_steps"], ["wait_for_healthy_pipeline"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
