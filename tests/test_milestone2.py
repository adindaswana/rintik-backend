from __future__ import annotations

import sqlite3
import unittest
import uuid
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import inference as inf
from daily_batch import BatchError, DailyBatchService, FrozenRuntime


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROTOTYPE_DIR = BACKEND_DIR.parent
WORKSPACE_DIR = PROTOTYPE_DIR.parent
OFFICIAL_DIR = PROTOTYPE_DIR / "reference" / "official"
MODEL_PATH = WORKSPACE_DIR / "NSC26-DS011_HBD_File Data Science.pth"
STATE_PATH = BACKEND_DIR / "artifacts" / "inference_state.json"


class Milestone2TemporalOrchestrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = FrozenRuntime(
            MODEL_PATH,
            STATE_PATH,
            OFFICIAL_DIR / "train.csv",
            OFFICIAL_DIR / "test.csv",
        )
        cls.test = pd.read_csv(OFFICIAL_DIR / "test.csv", parse_dates=["datetime"])
        cls.test_dates = cls.test["datetime"].dt.strftime("%Y-%m-%d")

    def setUp(self):
        token = uuid.uuid4().hex
        self.database = BACKEND_DIR / "runtime" / f"test-m2-{token}.sqlite3"
        self.reference_database = (
            BACKEND_DIR / "runtime" / f"test-m2-reference-{token}.sqlite3"
        )
        self.service = DailyBatchService(self.runtime, self.database)

    def tearDown(self):
        for database in (self.database, self.reference_database):
            for suffix in ("", "-wal", "-shm"):
                Path(str(database) + suffix).unlink(missing_ok=True)

    def _day(self, value: str) -> pd.DataFrame:
        return self.test.loc[self.test_dates == value, inf.INPUT_COLUMNS].copy()

    @staticmethod
    def _snapshot_bytes(result: dict) -> bytes:
        return inf.stable_json_bytes(result)

    @staticmethod
    def _published_rows_bytes(database: Path, day: str) -> bytes:
        with closing(sqlite3.connect(database)) as connection:
            rows = connection.execute(
                """
                SELECT p.timestamp, p.pred_internal, p.post_1, p.post_2,
                       p.post_3, p.air_humidity
                FROM published_predictions p
                JOIN runs r ON r.run_id = p.run_id
                WHERE r.date = ? ORDER BY p.timestamp
                """,
                (day,),
            ).fetchall()
        return inf.stable_json_bytes([list(row) for row in rows])

    def test_replay_restart_causality_windows_and_snapshot_immutability(self):
        full_internal = self.runtime.predict_internal(self.test[inf.INPUT_COLUMNS])
        input_counts = []
        window_counts = []
        first_snapshot = None
        first_rows = None
        final_result = None

        for index, day in enumerate(self.runtime.ordered_dates):
            ingested = self.service.ingest(day, self._day(day), f"m2-{day}")
            result = self.service.run(day)
            input_counts.append(ingested["row_count"])
            window_counts.append(result["window"]["row_count"])

            prefix_rows = sum(input_counts)
            prefix_frame = self.test.iloc[:prefix_rows][inf.INPUT_COLUMNS]
            direct_prefix = self.runtime.predict_internal(prefix_frame)
            prefix_vs_full = np.abs(
                direct_prefix["pred_internal"].to_numpy()
                - full_internal["pred_internal"].to_numpy()[:prefix_rows]
            )
            self.assertLessEqual(float(prefix_vs_full.max()), 1e-10)

            with closing(sqlite3.connect(self.database)) as connection:
                archived = connection.execute(
                    "SELECT pred_internal FROM internal_predictions ORDER BY timestamp"
                ).fetchall()
            archived_values = np.asarray([row[0] for row in archived])
            self.assertEqual(len(archived_values), prefix_rows)
            self.assertLessEqual(
                float(
                    np.max(
                        np.abs(
                            archived_values
                            - direct_prefix["pred_internal"].to_numpy()
                        )
                    )
                ),
                1e-10,
            )

            active_dates = self.runtime.ordered_dates[max(0, index - 14): index + 1]
            direct_dates = direct_prefix["datetime"].dt.strftime("%Y-%m-%d")
            direct_window = direct_prefix.loc[
                direct_dates.isin(active_dates)
            ].reset_index(drop=True)
            expected_stages = self.runtime.postprocess(direct_window)
            current_mask = direct_window["datetime"].dt.strftime("%Y-%m-%d").eq(day)
            actual_rows = result["predictions"]

            self.assertEqual({row["timestamp"][:10] for row in actual_rows}, {day})
            self.assertEqual(len(actual_rows), ingested["row_count"])
            for stage in ("post_1", "post_2", "post_3"):
                actual = np.asarray([row[stage] for row in actual_rows])
                expected = expected_stages[stage][current_mask.to_numpy()]
                self.assertLessEqual(float(np.max(np.abs(actual - expected))), 1e-12)

            if index == 0:
                first_snapshot = self._snapshot_bytes(result)
                first_rows = self._published_rows_bytes(self.database, day)

            if index == 7:
                before_restart = self._snapshot_bytes(result)
                self.service = DailyBatchService(self.runtime, self.database)
                after_restart = self._snapshot_bytes(self.service.get_result(day))
                self.assertEqual(after_restart, before_restart)
                self.assertEqual(self.service.status()["next_expected_date"], self.runtime.ordered_dates[8])

            final_result = result

        self.assertEqual(input_counts, [18, *([48] * 14), 44])
        self.assertEqual(sum(input_counts), 734)
        self.assertEqual(
            window_counts,
            [18 + 48 * index for index in range(15)] + [716],
        )
        self.assertEqual(final_result["window"]["start_date"], self.runtime.ordered_dates[1])
        self.assertEqual(final_result["window"]["end_date"], self.runtime.ordered_dates[-1])
        self.assertEqual(final_result["window"]["mode"], "rolling_15_calendar_days")

        first_day = self.runtime.ordered_dates[0]
        self.assertEqual(
            self._snapshot_bytes(self.service.get_result(first_day)),
            first_snapshot,
        )
        self.assertEqual(
            self._published_rows_bytes(self.database, first_day),
            first_rows,
        )
        self.assertEqual(self.service.status()["archived_pred_internal_rows"], 734)

        reference = DailyBatchService(self.runtime, self.reference_database)
        reference_final = None
        for day in self.runtime.ordered_dates:
            reference.ingest(day, self._day(day), f"reference-{day}")
            reference_final = reference.run(day)
        self.assertEqual(
            self._snapshot_bytes(reference_final),
            self._snapshot_bytes(final_result),
        )

    def test_failure_restart_and_rejected_temporal_edges(self):
        first, second = self.runtime.ordered_dates[:2]

        with self.assertRaises(BatchError) as caught:
            self.service.ingest(second, self._day(second), "out-of-order")
        self.assertEqual(caught.exception.code, "OUT_OF_ORDER_DATE")

        mixed = pd.concat(
            [self._day(first), self._day(second).iloc[[0]]],
            ignore_index=True,
        )
        with self.assertRaises(BatchError) as caught:
            self.service.ingest(first, mixed, "mixed-date")
        self.assertEqual(caught.exception.code, "MULTIPLE_OR_WRONG_DATE")

        duplicate = pd.concat(
            [self._day(first), self._day(first).iloc[[0]]],
            ignore_index=True,
        )
        with self.assertRaises(BatchError) as caught:
            self.service.ingest(first, duplicate, "duplicate")
        self.assertEqual(caught.exception.code, "DUPLICATE_TIMESTAMP")

        altered = self._day(first)
        altered.loc[altered.index[0], "co2"] += 1
        with self.assertRaises(BatchError) as caught:
            self.service.ingest(first, altered, "altered-first-attempt")
        self.assertEqual(caught.exception.code, "OFFICIAL_FIXTURE_MISMATCH")

        for invalid_key in ("", "   ", "x" * 129):
            with self.assertRaises(BatchError) as caught:
                self.service.ingest(first, self._day(first), invalid_key)
            self.assertEqual(caught.exception.code, "INVALID_IDEMPOTENCY_KEY")

        accepted = self.service.ingest(first, self._day(first), "stable-key")
        self.assertFalse(accepted["idempotent"])
        repeated = self.service.ingest(first, self._day(first), "stable-key")
        self.assertTrue(repeated["idempotent"])

        with self.assertRaises(BatchError) as caught:
            self.service.ingest(first, altered, "stable-key")
        self.assertEqual(caught.exception.code, "INPUT_CONFLICT")
        self.assertEqual(caught.exception.status_code, 409)

        with patch.object(
            self.runtime,
            "predict_internal",
            side_effect=RuntimeError("injected inference failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected inference failure"):
                self.service.run(first)

        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 0)
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM internal_predictions").fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM published_predictions").fetchone()[0],
                0,
            )

        restarted = DailyBatchService(self.runtime, self.database)
        recovered = restarted.run(first)
        self.assertEqual(recovered["status"], "completed")
        self.assertEqual(len(recovered["predictions"]), 18)

        with self.assertRaises(BatchError) as caught:
            restarted.ingest(second, self._day(second), "stable-key")
        self.assertEqual(caught.exception.code, "IDEMPOTENCY_KEY_CONFLICT")
        self.assertEqual(caught.exception.status_code, 409)

        accepted_second = restarted.ingest(second, self._day(second), "second-key")
        self.assertEqual(accepted_second["row_count"], 48)

        legacy = DailyBatchService(self.runtime, self.reference_database)
        legacy.ingest(first, self._day(first), "legacy-reused-key")
        with closing(sqlite3.connect(self.reference_database)) as connection:
            connection.execute("DROP INDEX uq_daily_inputs_idempotency_key")
            connection.execute(
                """
                INSERT INTO daily_inputs (
                    date, payload_sha256, idempotency_key, row_count,
                    first_timestamp, last_timestamp, csv_payload, ingested_at
                )
                SELECT ?, payload_sha256, idempotency_key, row_count,
                       first_timestamp, last_timestamp, csv_payload, ingested_at
                FROM daily_inputs WHERE date = ?
                """,
                (second, first),
            )
            connection.commit()
        with self.assertRaises(BatchError) as caught:
            DailyBatchService(self.runtime, self.reference_database)
        self.assertEqual(caught.exception.code, "STATE_IDEMPOTENCY_CONFLICT")
        self.assertEqual(caught.exception.status_code, 500)


if __name__ == "__main__":
    unittest.main(verbosity=2)
