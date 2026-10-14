from __future__ import annotations

import json
import sqlite3
import unittest
import uuid
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

import inference as inf
from daily_batch import BatchError, DailyBatchService, FrozenRuntime
from operational_simulation import OperationalSimulationService


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROTOTYPE_DIR = BACKEND_DIR.parent
WORKSPACE_DIR = PROTOTYPE_DIR.parent
OFFICIAL_DIR = PROTOTYPE_DIR / "reference" / "official"
FIXTURES = BACKEND_DIR / "fixtures"


class OperationalSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = FrozenRuntime(
            WORKSPACE_DIR / "NSC26-DS011_HBD_File Data Science.pth",
            BACKEND_DIR / "artifacts" / "inference_state.json",
            OFFICIAL_DIR / "train.csv",
            OFFICIAL_DIR / "test.csv",
        )
        cls.fixture = pd.read_csv(FIXTURES / "operational_2026_exogenous.csv")

    def setUp(self):
        self.database = BACKEND_DIR / "runtime" / f"test-operational-{uuid.uuid4().hex}.sqlite3"
        self.official = DailyBatchService(self.runtime, self.database)
        self.service = OperationalSimulationService(
            self.runtime,
            self.database,
            FIXTURES / "operational_2026_exogenous.csv",
            FIXTURES / "operational_2026_provenance.json",
        )

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.database) + suffix).unlink(missing_ok=True)

    def _reading(self, timestamp: str) -> dict:
        row = self.fixture.loc[self.fixture["datetime"] == timestamp].iloc[0]
        return {column: row[column].item() if hasattr(row[column], "item") else row[column] for column in inf.INPUT_COLUMNS}

    def test_manual_slots_fill_process_restart_and_namespace_isolation(self):
        day = "2026-10-14"
        manual_slots = [f"{day} {hour:02d}:{minute:02d}:00" for hour in range(21, 24) for minute in (0, 30)]
        for timestamp in manual_slots:
            saved = self.service.save_manual(day, self._reading(timestamp))
            self.assertEqual(saved["source"], "manual")
        self.assertEqual(saved["progress"]["available"], 6)

        with self.assertRaises(BatchError) as duplicate:
            self.service.save_manual(day, self._reading(manual_slots[0]))
        self.assertEqual(duplicate.exception.code, "DUPLICATE_OPERATIONAL_TIMESTAMP")

        edited = self._reading(manual_slots[0])
        edited["co2"] = float(edited["co2"]) + 1.0
        update = self.service.save_manual(day, edited, replace=True)
        self.assertEqual(update["action"], "updated")

        completed = self.service.fill_missing(day)
        self.assertEqual(completed["inserted_fixture_rows"], 42)
        self.assertEqual(completed["progress"], {
            "available": 48, "required": 48, "manual": 6,
            "synthetic_fixture": 42, "missing": 0, "ready": True,
        })
        result = self.service.run(day, "normal")
        self.assertEqual(len(result["predictions"]), 48)
        self.assertTrue(result["simulation"])
        self.assertEqual(result["input_composition"], {"manual": 6, "synthetic_fixture": 42, "total": 48})
        self.assertEqual(result["window"]["mode"], "expanding")
        self.assertEqual(result["monitoring"]["provenance"]["model_input"], False)
        self.assertEqual(self.official.status()["completed_days"], 0)

        restarted = OperationalSimulationService(
            self.runtime, self.database,
            FIXTURES / "operational_2026_exogenous.csv",
            FIXTURES / "operational_2026_provenance.json",
        )
        self.assertEqual(restarted.get_result(day)["run_id"], result["run_id"])
        self.assertEqual(restarted.list_days()["next_expected_date"], "2026-10-15")
        with self.assertRaises(BatchError) as locked:
            restarted.save_manual(day, edited, replace=True)
        self.assertEqual(locked.exception.code, "OPERATIONAL_DAY_LOCKED")

    def test_missing_sensor_recovery_equals_model_and_reset_is_scoped(self):
        first, second = "2026-10-14", "2026-10-15"
        self.service.fill_missing(first)
        first_result = self.service.run(first, "normal")
        self.service.fill_missing(second)
        second_result = self.service.run(second, "missing")
        self.assertEqual(second_result["historical_pred_internal_max_abs_diff"], 0.0)
        model = [row["post_3"] for row in second_result["predictions"]]
        recovery = [row["recovery_rh"] for row in second_result["monitoring"]["recovery"]["values"]]
        self.assertTrue(np.array_equal(np.asarray(model), np.asarray(recovery)))
        self.assertTrue(all(row["physical_rh"] is None for row in second_result["monitoring"]["sensor"]["values"]))
        self.assertEqual(second_result["monitoring"]["recovery"]["status"], "active_estimated")

        reset = self.service.reset()
        self.assertEqual(reset["status"], "reset")
        self.assertEqual(self.service.list_days()["completed_days"], 0)
        self.assertEqual(self.official.status()["completed_days"], 0)
        self.assertEqual(reset["model_sha256"], inf.OFFICIAL_MODEL_SHA256)

    def test_invalid_readings_fail_closed(self):
        day = "2026-10-14"
        reading = self._reading(f"{day} 21:00:00")
        reading["air_temperature"] = 999
        with self.assertRaises(BatchError) as out_of_range:
            self.service.save_manual(day, reading)
        self.assertEqual(out_of_range.exception.code, "VALUE_OUT_OF_RANGE")

        reading = self._reading(f"{day} 21:00:00")
        reading["water_volume_line2"] = -1
        with self.assertRaises(BatchError) as invalid_volume:
            self.service.save_manual(day, reading)
        self.assertEqual(invalid_volume.exception.code, "VALUE_OUT_OF_RANGE")

        reading = self._reading(f"{day} 21:00:00")
        reading["env_id"] = 1.5
        with self.assertRaises(BatchError) as invalid_identifier:
            self.service.save_manual(day, reading)
        self.assertEqual(invalid_identifier.exception.code, "INVALID_IDENTIFIER")

        reading = self._reading(f"{day} 21:00:00")
        del reading["co2"]
        with self.assertRaises(BatchError) as missing:
            self.service.save_manual(day, reading)
        self.assertEqual(missing.exception.code, "MISSING_COLUMNS")

        self.service.fill_missing(day)
        with self.assertRaises(BatchError) as wrong_order:
            self.service.fill_missing("2026-10-16")
        self.assertEqual(wrong_order.exception.code, "OUT_OF_ORDER_OPERATIONAL_DATE")


if __name__ == "__main__":
    unittest.main()
