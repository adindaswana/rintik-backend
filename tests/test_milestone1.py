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


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROTOTYPE_DIR = BACKEND_DIR.parent
WORKSPACE_DIR = PROTOTYPE_DIR.parent
MODEL_PATH = WORKSPACE_DIR / "NSC26-DS011_HBD_File Data Science.pth"
STATE_PATH = BACKEND_DIR / "artifacts" / "inference_state.json"
OFFICIAL_DIR = PROTOTYPE_DIR / "reference" / "official"
TRAIN_PATH = OFFICIAL_DIR / "train.csv"
TEST_PATH = OFFICIAL_DIR / "test.csv"
SUBMISSION_PATH = OFFICIAL_DIR / "FINALIZATION_submission.csv"


class FrozenInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = FrozenRuntime(MODEL_PATH, STATE_PATH, TRAIN_PATH, TEST_PATH)
        cls.test = pd.read_csv(TEST_PATH, parse_dates=["datetime"])

    def test_protected_hashes_and_sidecar_provenance(self):
        self.assertEqual(inf.sha256_file(MODEL_PATH), inf.OFFICIAL_MODEL_SHA256)
        self.assertEqual(inf.sha256_file(TRAIN_PATH), inf.OFFICIAL_TRAIN_SHA256)
        self.assertEqual(inf.sha256_file(TEST_PATH), inf.OFFICIAL_TEST_SHA256)
        self.assertEqual(inf.sha256_file(SUBMISSION_PATH), inf.OFFICIAL_SUBMISSION_SHA256)
        self.assertEqual(
            self.runtime.sidecar["sources"]["model"]["sha256"],
            inf.OFFICIAL_MODEL_SHA256,
        )
        self.assertEqual(self.runtime.state["day_context"]["row_count"], 30)
        self.assertEqual(len(self.runtime.state["daily_median_features"]), 52)
        self.assertAlmostEqual(self.runtime.state["last10_mean"], 57.859083333333, places=11)
        self.assertAlmostEqual(self.runtime.state["carry_r0"], 4.268314490300, places=11)

    def test_full_horizon_matches_official_submission_at_published_precision(self):
        expected = pd.read_csv(SUBMISSION_PATH)
        actual = self.runtime.full_horizon(self.test)
        difference = np.abs(
            actual["air_humidity"].to_numpy()
            - expected["air_humidity"].to_numpy()
        )
        self.assertEqual(len(actual), 734)
        self.assertEqual(int(np.count_nonzero(difference)), 0)
        self.assertEqual(float(difference.max()), 0.0)

    def test_integrity_failures_are_closed(self):
        corrupt_model = BACKEND_DIR / "runtime" / f"corrupt-{uuid.uuid4().hex}.pth"
        corrupt_state = BACKEND_DIR / "runtime" / f"corrupt-{uuid.uuid4().hex}.json"
        try:
            corrupt_model.write_bytes(b"not the official model")
            with self.assertRaises(inf.ArtifactIntegrityError):
                inf.load_frozen_artifact(corrupt_model)

            sidecar = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            sidecar["sources"]["model"]["sha256"] = "0" * 64
            corrupt_state.write_bytes(inf.stable_json_bytes(sidecar))
            with self.assertRaises(inf.ArtifactIntegrityError):
                inf.load_state_sidecar(corrupt_state, inf.OFFICIAL_MODEL_SHA256)
        finally:
            corrupt_model.unlink(missing_ok=True)
            corrupt_state.unlink(missing_ok=True)


class DailyReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = FrozenRuntime(MODEL_PATH, STATE_PATH, TRAIN_PATH, TEST_PATH)
        cls.test = pd.read_csv(TEST_PATH, parse_dates=["datetime"])
        cls.test["date"] = cls.test["datetime"].dt.strftime("%Y-%m-%d")

    def setUp(self):
        self.database = BACKEND_DIR / "runtime" / f"test-replay-{uuid.uuid4().hex}.sqlite3"
        self.service = DailyBatchService(self.runtime, self.database)

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            Path(str(self.database) + suffix).unlink(missing_ok=True)

    def _day(self, value: str) -> pd.DataFrame:
        return self.test.loc[self.test["date"] == value, inf.INPUT_COLUMNS].copy()

    def test_full_chronological_replay_window_and_restart(self):
        counts = []
        final_result = None
        first_result = None
        for index, day in enumerate(self.runtime.ordered_dates):
            ingested = self.service.ingest(day, self._day(day), f"fixture-{day}")
            counts.append(ingested["row_count"])
            if index == 0:
                repeated_ingest = self.service.ingest(day, self._day(day), f"fixture-{day}")
                self.assertTrue(repeated_ingest["idempotent"])
            result = self.service.run(day)
            if index == 0:
                first_result = result
                repeated_run = self.service.run(day)
                self.assertTrue(repeated_run["idempotent"])
                self.assertEqual(repeated_run["predictions"], result["predictions"])
            if index < 15:
                self.assertEqual(result["window"]["mode"], "expanding")
                self.assertEqual(result["window"]["date_count"], index + 1)
            else:
                self.assertEqual(result["window"]["mode"], "rolling_15_calendar_days")
                self.assertEqual(result["window"]["date_count"], 15)
            self.assertEqual(result["historical_pred_internal_max_abs_diff"], 0.0)
            final_result = result

        self.assertEqual(counts, [18, *([48] * 14), 44])
        self.assertEqual(final_result["window"]["row_count"], 716)
        self.assertEqual(final_result["window"]["start_date"], self.runtime.ordered_dates[1])
        self.assertEqual(len(final_result["predictions"]), 44)
        self.assertEqual(first_result["window"]["row_count"], 18)

        status = self.service.status()
        self.assertEqual(status["ingested_days"], 16)
        self.assertEqual(status["completed_days"], 16)
        self.assertEqual(status["archived_pred_internal_rows"], 734)

        restarted = DailyBatchService(self.runtime, self.database)
        after_restart = restarted.get_result(self.runtime.ordered_dates[-1])
        self.assertEqual(after_restart["run_id"], final_result["run_id"])
        self.assertEqual(after_restart["predictions"], final_result["predictions"])

        with closing(sqlite3.connect(self.database)) as connection:
            archived = connection.execute(
                "SELECT timestamp, pred_internal FROM internal_predictions ORDER BY timestamp"
            ).fetchall()
            self.assertEqual(len(archived), 734)
        direct = self.runtime.predict_internal(self.test[inf.INPUT_COLUMNS])
        archived_values = np.asarray([row[1] for row in archived])
        self.assertLessEqual(
            float(np.max(np.abs(archived_values - direct["pred_internal"].to_numpy()))),
            1e-10,
        )
        direct_dates = direct["datetime"].dt.strftime("%Y-%m-%d")
        expected_window = direct.loc[
            direct_dates.isin(self.runtime.ordered_dates[-15:])
        ].reset_index(drop=True)
        expected_stages = self.runtime.postprocess(expected_window)
        expected_current = expected_stages["post_3"][
            expected_window["datetime"].dt.strftime("%Y-%m-%d").eq(
                self.runtime.ordered_dates[-1]
            )
        ]
        stored_current = np.asarray([
            row["post_3"] for row in final_result["predictions"]
        ])
        self.assertLessEqual(
            float(np.max(np.abs(stored_current - expected_current))),
            1e-12,
        )

    def test_validation_and_chronology_edges(self):
        first = self.runtime.ordered_dates[0]
        second = self.runtime.ordered_dates[1]

        with self.assertRaises(BatchError) as caught:
            self.service.ingest(second, self._day(second))
        self.assertEqual(caught.exception.code, "OUT_OF_ORDER_DATE")

        with self.assertRaises(BatchError) as caught:
            self.service.ingest(first, pd.DataFrame(columns=inf.INPUT_COLUMNS))
        self.assertEqual(caught.exception.code, "EMPTY_INPUT")

        missing_column = self._day(first).drop(columns=["pressure"])
        with self.assertRaises(BatchError) as caught:
            self.service.ingest(first, missing_column)
        self.assertEqual(caught.exception.code, "MISSING_COLUMNS")

        unexpected_target = self._day(first)
        unexpected_target["air_humidity"] = 50.0
        with self.assertRaises(BatchError) as caught:
            self.service.ingest(first, unexpected_target)
        self.assertEqual(caught.exception.code, "UNEXPECTED_COLUMNS")

        duplicate = pd.concat([self._day(first), self._day(first).iloc[[0]]], ignore_index=True)
        with self.assertRaises(BatchError) as caught:
            self.service.ingest(first, duplicate)
        self.assertEqual(caught.exception.code, "DUPLICATE_TIMESTAMP")

        unaligned = self._day(first)
        unaligned.loc[unaligned.index[0], "datetime"] += pd.Timedelta(minutes=1)
        with self.assertRaises(BatchError) as caught:
            self.service.ingest(first, unaligned)
        self.assertEqual(caught.exception.code, "TIMESTAMP_NOT_ALIGNED")

        critical_missing = self._day(first)
        critical_missing["co2"] = np.nan
        with self.assertRaises(BatchError) as caught:
            self.service.ingest(first, critical_missing)
        self.assertEqual(caught.exception.code, "CRITICAL_FEATURE_EMPTY")

        altered = self._day(first)
        altered.loc[altered.index[0], "co2"] += 1
        with self.assertRaises(BatchError) as caught:
            self.service.ingest(first, altered)
        self.assertEqual(caught.exception.code, "OFFICIAL_FIXTURE_MISMATCH")

        valid = self.service.ingest(first, self._day(first))
        self.assertEqual(valid["row_count"], 18)
        with self.assertRaises(BatchError) as caught:
            self.service.ingest(second, self._day(second))
        self.assertEqual(caught.exception.code, "PREVIOUS_DAY_NOT_RUN")


if __name__ == "__main__":
    unittest.main(verbosity=2)
