from __future__ import annotations

import csv
import io
import json
import unittest
import uuid
from pathlib import Path

import pandas as pd

import inference as inf
from daily_batch import BatchError, DailyBatchService, FrozenRuntime
from monitoring import MonitoringService
from operator_api import OperatorDataService, feature_catalog


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROTOTYPE_DIR = BACKEND_DIR.parent
WORKSPACE_DIR = PROTOTYPE_DIR.parent
OFFICIAL_DIR = PROTOTYPE_DIR / "reference" / "official"
FIXTURE_DIR = PROTOTYPE_DIR / "demo" / "fixtures"


class OperatorDataTests(unittest.TestCase):
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
        self.database = BACKEND_DIR / "runtime" / f"test-operator-{uuid.uuid4().hex}.sqlite3"
        self.extra_databases: list[Path] = []
        self.daily = DailyBatchService(self.runtime, self.database)
        self.monitoring = MonitoringService(
            self.daily,
            FIXTURE_DIR / "synthetic_physical_humidity_SENSOR_SIMULATION.csv",
            FIXTURE_DIR / "SOURCE_PROVENANCE.json",
        )
        self.operator = OperatorDataService(self.daily)

    def tearDown(self):
        for database in [self.database, *self.extra_databases]:
            for suffix in ("", "-wal", "-shm"):
                Path(str(database) + suffix).unlink(missing_ok=True)

    def _ingest_run_monitor(self, day: str) -> None:
        frame = self.test.loc[self.test_dates == day, inf.INPUT_COLUMNS].copy()
        self.daily.ingest(day, frame, f"operator-{day}")
        self.daily.run(day)
        self.monitoring.monitor_day(day)

    def test_feature_catalog_is_complete_unit_safe_and_excludes_identifier(self):
        catalog = feature_catalog()
        features = [feature for group in catalog["groups"] for feature in group["features"]]
        keys = [feature["key"] for feature in features]
        self.assertEqual(len(keys), 23)
        self.assertEqual(len(keys), len(set(keys)))
        self.assertNotIn("env_id", keys)
        self.assertEqual({item["key"] for item in catalog["excluded_fields"]}, {"env_id"})
        self.assertTrue(all(feature["unit"] for feature in features))
        self.assertIn("% tanah", {feature["unit"] for feature in features})
        self.assertNotIn("air_humidity", keys)

    def test_series_preserves_exact_values_nulls_and_partial_day(self):
        day = self.runtime.ordered_dates[0]
        self._ingest_run_monitor(day)
        result = self.operator.get_series(day)
        self.assertEqual(result["row_count"], 18)
        self.assertEqual(result["source"], "official_test_input")
        self.assertFalse(result["simulation"])
        self.assertEqual(result["run_status"], "completed")
        self.assertEqual(result["points"][0]["timestamp"], "2023-08-29 15:00:00")

        official = self.test.loc[self.test_dates == day].reset_index(drop=True)
        for index, point in enumerate(result["points"]):
            for key, value in point["values"].items():
                expected = official.loc[index, key]
                if pd.isna(expected):
                    self.assertIsNone(value)
                else:
                    self.assertAlmostEqual(value, float(expected), places=12)
        json.dumps(result, allow_nan=False)

    def test_export_has_lineage_sources_and_missing_raw_is_not_replaced(self):
        for day in self.runtime.ordered_dates[:9]:
            self._ingest_run_monitor(day)
        outage = "2023-09-06"
        monitored = self.monitoring.get_monitoring(outage)
        self.monitoring.confirm_incident(
            monitored["incident"]["incident_id"],
            demo_override=True,
            note="Explicit operator export test action.",
            idempotency_key="operator-export-confirm",
        )

        rows = list(csv.DictReader(io.StringIO(self.operator.export_day_csv(outage))))
        self.assertEqual(len(rows), 48)
        self.assertTrue(all(row["prediction_source"] == "real_original_pth" for row in rows))
        self.assertTrue(all(row["simulation"] == "false" for row in rows))
        self.assertTrue(all(row["simulated_physical_rh"] == "" for row in rows))
        self.assertTrue(all(row["physical_source"] == "SYNTHETIC_DEMO_NOT_GROUND_TRUTH" for row in rows))
        self.assertTrue(all(row["recovery_source"] == "model_estimate" for row in rows))
        self.assertTrue(all(row["recovery_rh"] == row["post_3"] for row in rows))
        self.assertTrue(all(row["model_sha256"] == inf.OFFICIAL_MODEL_SHA256 for row in rows))
        self.assertTrue(all(row["state_sha256"] == inf.OFFICIAL_STATE_SHA256 for row in rows))
        self.assertTrue(all(row["monitoring_verdict"] == "suspected_sensor_outage" for row in rows))

        self.assertEqual(OperatorDataService._safe_csv_text("=cmd"), "'=cmd")
        self.assertEqual(OperatorDataService._safe_csv_text("safe"), "safe")

    def test_missing_state_errors_are_explicit(self):
        first = self.runtime.ordered_dates[0]
        with self.assertRaises(BatchError) as caught:
            self.operator.get_series(first)
        self.assertEqual(caught.exception.code, "DATE_NOT_INGESTED")
        with self.assertRaises(BatchError) as caught:
            self.operator.export_day_csv(first)
        self.assertEqual(caught.exception.code, "RUN_NOT_FOUND")

    def test_export_remains_available_when_optional_monitoring_is_degraded(self):
        database = BACKEND_DIR / "runtime" / f"test-operator-bare-{uuid.uuid4().hex}.sqlite3"
        self.extra_databases.append(database)
        daily = DailyBatchService(self.runtime, database)
        first = self.runtime.ordered_dates[0]
        frame = self.test.loc[self.test_dates == first, inf.INPUT_COLUMNS].copy()
        daily.ingest(first, frame, "bare-export")
        daily.run(first)

        rows = list(csv.DictReader(io.StringIO(OperatorDataService(daily).export_day_csv(first))))
        self.assertEqual(len(rows), 18)
        self.assertTrue(all(row["prediction_source"] == "real_original_pth" for row in rows))
        self.assertTrue(all(row["physical_source"] == "" for row in rows))
        self.assertTrue(all(row["recovery_source"] == "" for row in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
