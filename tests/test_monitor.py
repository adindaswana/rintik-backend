"""Uji Pemeriksaan Sensor: aturan demo-v1 (dari Adinda) dan endpoint /monitor.

Bagian A (aturan) tidak butuh model. Bagian B memakai runtime beku resmi
(.pth + sidecar + test.csv) lewat ASGI, sama seperti tests/test_api.py.
Data sensor pada bagian B SINTETIS (turunan estimasi model), bukan ground truth.
"""
from __future__ import annotations

import sys
import unittest
import uuid
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import app as api  # noqa: E402
import sensor_check  # noqa: E402
from daily_batch import DailyBatchService, FrozenRuntime  # noqa: E402

BACKEND_DIR = HERE.parent
PROTOTYPE_DIR = BACKEND_DIR.parent
WORKSPACE_DIR = PROTOTYPE_DIR.parent
OFFICIAL_DIR = PROTOTYPE_DIR / "reference" / "official"
CENTER = sensor_check.RULES["center"]["p3"]


# ---------------------------------------------------------------- A. aturan
def synth(days=6, shift_from=None, shift=-20.0, flat_day=None, stuck_day=None, jump_day=None,
          short_day=None, seed=0):
    """Tebakan sintetis berayun ~30 poin/hari; sensor = tebakan + center + derau kecil."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2023-08-01", periods=48 * days, freq="30min")
    slot = np.arange(len(t)) % 48
    model = 70 + 15 * np.cos(2 * np.pi * slot / 48)
    sensor = model + CENTER + rng.normal(0, 1.5, len(t))
    day = np.arange(len(t)) // 48
    if shift_from is not None:
        sensor[day >= shift_from] += shift
    if flat_day is not None:
        sensor[day == flat_day] = sensor[day == flat_day].mean()
    if stuck_day is not None:
        idx = np.flatnonzero(day == stuck_day)[:14]
        sensor[idx] = 99.0
    if jump_day is not None:
        sensor[np.flatnonzero(day == jump_day)[20]] += 45
    if short_day is not None:
        sensor[np.flatnonzero(day == short_day)[:10]] = np.nan
    return t, np.clip(sensor, 0, 100), model


def statuses(days):
    return [d["status"] for d in days]


def test_rules_sehat_impl():
    t, s, m = synth()
    days = sensor_check.evaluate_days(t, s, m, CENTER)
    assert statuses(days) == ["sehat"] * 6, days
    assert days[0]["diff3"] is None and days[1]["diff3"] is None and days[2]["diff3"] is not None


def test_rules_shift_minus20_impl():
    t, s, m = synth(days=8, shift_from=4)
    st = statuses(sensor_check.evaluate_days(t, s, m, CENTER))
    assert st[:4] == ["sehat"] * 4 and all(x == "dugaan_malfungsi" for x in st[4:]), st


def test_rules_shift_minus12_two_days_impl():
    t, s, m = synth(days=6, shift_from=3, shift=-12.5)
    days = sensor_check.evaluate_days(t, s, m, CENTER)
    st = statuses(days)
    assert st[3] == "waspada" and st[4] == "dugaan_malfungsi", st
    assert any("dua hari berturut-turut" in r for r in days[4]["reasons"])


def test_rules_flat_impl():
    t, s, m = synth(flat_day=2)
    days = sensor_check.evaluate_days(t, s, m, CENTER)
    assert days[2]["status"] == "dugaan_malfungsi" and any("sensor datar" in r for r in days[2]["reasons"])
    assert days[2]["corr"] is None


def test_rules_stuck_and_jump_impl():
    t, s, m = synth(stuck_day=1, jump_day=3)
    days = sensor_check.evaluate_days(t, s, m, CENTER)
    assert days[1]["status"] == "dugaan_malfungsi" and any("beruntun" in r for r in days[1]["reasons"])
    assert days[3]["status"] == "dugaan_malfungsi" and any("loncat" in r for r in days[3]["reasons"])


def test_rules_data_kurang_impl():
    t, s, m = synth(short_day=0)
    days = sensor_check.evaluate_days(t, s, m, CENTER)
    assert days[0]["status"] == "data_kurang" and days[0]["n"] == 38
    assert days[2]["diff3"] is None  # jendela 3 hari memuat hari tidak lengkap
    assert days[1]["status"] == "sehat"




class RuleTests(unittest.TestCase):
    def test_rules_sehat(self):
        test_rules_sehat_impl()

    def test_rules_shift_minus20(self):
        test_rules_shift_minus20_impl()

    def test_rules_shift_minus12_two_days(self):
        test_rules_shift_minus12_two_days_impl()

    def test_rules_flat(self):
        test_rules_flat_impl()

    def test_rules_stuck_and_jump(self):
        test_rules_stuck_and_jump_impl()

    def test_rules_data_kurang(self):
        test_rules_data_kurang_impl()



class MonitorEndpointTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = FrozenRuntime(
            WORKSPACE_DIR / "NSC26-DS011_HBD_File Data Science.pth",
            BACKEND_DIR / "artifacts" / "inference_state.json",
            OFFICIAL_DIR / "train.csv",
            OFFICIAL_DIR / "test.csv",
        )
        test = pd.read_csv(OFFICIAL_DIR / "test.csv", parse_dates=["datetime"])
        first_days = sorted(test["datetime"].dt.strftime("%Y-%m-%d").unique())[:6]
        cls.features = test[test["datetime"].dt.strftime("%Y-%m-%d").isin(first_days)].reset_index(drop=True)
        cls.days = first_days

    async def asyncSetUp(self):
        self.database = BACKEND_DIR / "runtime" / f"test-monitor-{uuid.uuid4().hex}.sqlite3"
        api.SERVICE = DailyBatchService(self.runtime, self.database)
        api.INITIALIZATION_ERROR = None
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://testserver"
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.database) + suffix).unlink(missing_ok=True)

    def rows(self, sensor):
        frame = self.features.copy()
        frame["datetime"] = frame["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        frame["air_humidity"] = sensor
        return [
            {k: (None if pd.isna(v) else v) for k, v in row.items()}
            for row in frame.to_dict(orient="records")
        ]

    async def estimate(self):
        response = await self.client.post("/monitor", json={"rows": self.rows(50.0)})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    async def test_healthy_and_faulty_sensor(self):
        base = await self.estimate()
        p3 = np.array([row["p3"] for row in base["predictions"]])
        stamps = pd.to_datetime([row["datetime"] for row in base["predictions"]])
        day_no = pd.Series(stamps.strftime("%Y-%m-%d")).map({d: i for i, d in enumerate(self.days)}).to_numpy()
        healthy = np.clip(p3 + CENTER, 0, 100)
        broken = np.where(day_no >= 3, np.clip(healthy - 25, 0, 100), healthy)
        ok = (await self.client.post("/monitor", json={"rows": self.rows(list(healthy))})).json()
        bad = (await self.client.post("/monitor", json={"rows": self.rows(list(broken))})).json()
        self.assertEqual(ok["count"], len(self.features))
        self.assertEqual([d["status"] for d in ok["days"]], ["data_kurang"] + ["sehat"] * 5)  # hari pertama resmi hanya 18 baris
        self.assertEqual([d["status"] for d in bad["days"]][3:], ["dugaan_malfungsi"] * 3)
        provenance = ok["provenance"]
        self.assertEqual(provenance["source"], "real_original_pth")
        self.assertFalse(provenance["simulation"])
        self.assertEqual(provenance["thresholds"], "illustrative_unvalidated")
        self.assertFalse(provenance["archived"])

    async def test_sensor_reading_does_not_change_estimate(self):
        a = await self.estimate()
        response = await self.client.post("/monitor", json={"rows": self.rows(5.0)})
        b = response.json()
        self.assertEqual([p["p3"] for p in a["predictions"]], [p["p3"] for p in b["predictions"]])
        self.assertEqual([p["inti"] for p in a["predictions"]], [p["inti"] for p in b["predictions"]])

    async def test_validation_errors(self):
        rows = self.rows(50.0)
        no_sensor = [{k: v for k, v in row.items() if k != "air_humidity"} for row in rows]
        r = await self.client.post("/monitor", json={"rows": no_sensor})
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (400, "INVALID_SENSOR_READINGS"))
        r = await self.client.post("/monitor", json={"rows": rows, "comparator": "x"})
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (400, "INVALID_COMPARATOR"))
        bad = [dict(row) for row in rows]
        bad[3]["air_humidity"] = "abc"
        r = await self.client.post("/monitor", json={"rows": bad})
        self.assertEqual(r.status_code, 400)
        self.assertIn("baris 4", r.json()["error"]["message"])
        dup = rows[:48] + rows[:1]
        r = await self.client.post("/monitor", json={"rows": dup})
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (400, "DUPLICATE_TIMESTAMP"))

    async def test_input_before_frozen_context_fails_closed(self):
        rows = self.rows(50.0)
        early = [dict(row, datetime="2023-08-10 " + row["datetime"][11:]) for row in rows[:48]]
        r = await self.client.post("/monitor", json={"rows": early})
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (400, "MONITOR_INPUT_OUT_OF_RANGE"))

    async def test_row_limit_matches_the_2000_rows_promised_by_the_ui(self):
        full = pd.read_csv(OFFICIAL_DIR / "test.csv")
        rows = [
            {k: (None if pd.isna(v) else v) for k, v in row.items()}
            for row in full.assign(air_humidity=60.0).to_dict(orient="records")
        ]
        self.assertGreater(len(rows), api.MAX_ROWS - 300)
        later = [dict(row, datetime=row["datetime"].replace("2023-", "2024-", 1)) for row in rows]
        # 1 468 rows: above the 1 000-row limit of other routes, accepted here.
        r = await self.client.post("/monitor", json={"rows": rows + later})
        self.assertNotEqual(r.json().get("error", {}).get("code"), "TOO_MANY_ROWS", r.text)
        too_many = (rows * 3)[: api.MONITOR_MAX_ROWS + 1]
        r = await self.client.post("/monitor", json={"rows": too_many})
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (413, "TOO_MANY_ROWS"))

    async def test_model_unavailable_has_no_fallback(self):
        api.SERVICE = None
        r = await self.client.post("/monitor", json={"rows": self.rows(50.0)})
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (503, "MODEL_UNAVAILABLE"))


if __name__ == "__main__":
    unittest.main()
