"""Produce reproducible evidence for Milestone 2 daily orchestration."""

from __future__ import annotations

import hashlib
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROTOTYPE_DIR = BACKEND_DIR.parent
WORKSPACE_DIR = PROTOTYPE_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

import inference as inf  # noqa: E402
from daily_batch import DailyBatchService, FrozenRuntime  # noqa: E402


def digest(value: object) -> str:
    return hashlib.sha256(inf.stable_json_bytes(value)).hexdigest()


def main() -> int:
    official = PROTOTYPE_DIR / "reference" / "official"
    database = BACKEND_DIR / "runtime" / f"verify-m2-{uuid.uuid4().hex}.sqlite3"
    try:
        runtime = FrozenRuntime(
            WORKSPACE_DIR / "NSC26-DS011_HBD_File Data Science.pth",
            BACKEND_DIR / "artifacts" / "inference_state.json",
            official / "train.csv",
            official / "test.csv",
        )
        test = pd.read_csv(official / "test.csv", parse_dates=["datetime"])
        test_dates = test["datetime"].dt.strftime("%Y-%m-%d")
        full_internal = runtime.predict_internal(test[inf.INPUT_COLUMNS])
        service = DailyBatchService(runtime, database)

        input_counts: list[int] = []
        window_rows: list[int] = []
        maximum_prefix_diff = 0.0
        maximum_post_diff = 0.0
        first_snapshot_before: str | None = None
        final_result: dict | None = None

        for index, day in enumerate(runtime.ordered_dates):
            frame = test.loc[test_dates == day, inf.INPUT_COLUMNS]
            ingest = service.ingest(day, frame, f"verify-m2-{day}")
            result = service.run(day)
            input_counts.append(ingest["row_count"])
            window_rows.append(result["window"]["row_count"])

            prefix_size = sum(input_counts)
            direct_prefix = runtime.predict_internal(
                test.iloc[:prefix_size][inf.INPUT_COLUMNS]
            )
            maximum_prefix_diff = max(
                maximum_prefix_diff,
                float(
                    np.max(
                        np.abs(
                            direct_prefix["pred_internal"].to_numpy()
                            - full_internal["pred_internal"].to_numpy()[:prefix_size]
                        )
                    )
                ),
            )

            active_dates = runtime.ordered_dates[max(0, index - 14): index + 1]
            prefix_dates = direct_prefix["datetime"].dt.strftime("%Y-%m-%d")
            active = direct_prefix.loc[prefix_dates.isin(active_dates)].reset_index(drop=True)
            expected = runtime.postprocess(active)
            current = active["datetime"].dt.strftime("%Y-%m-%d").eq(day).to_numpy()
            for stage in ("post_1", "post_2", "post_3"):
                actual = np.asarray([row[stage] for row in result["predictions"]])
                maximum_post_diff = max(
                    maximum_post_diff,
                    float(np.max(np.abs(actual - expected[stage][current]))),
                )

            if index == 0:
                first_snapshot_before = digest(result)
            if index == 7:
                service = DailyBatchService(runtime, database)
            final_result = result

        first_snapshot_after = digest(service.get_result(runtime.ordered_dates[0]))
        final_before_restart = digest(final_result)
        service = DailyBatchService(runtime, database)
        final_after_restart = digest(service.get_result(runtime.ordered_dates[-1]))
        repeated_final = service.run(runtime.ordered_dates[-1])
        status = service.status()

        expected_counts = [18, *([48] * 14), 44]
        expected_windows = [18 + 48 * index for index in range(15)] + [716]
        checks = {
            "model_hash": runtime.model_sha256 == inf.OFFICIAL_MODEL_SHA256,
            "input_counts": input_counts == expected_counts,
            "total_rows": sum(input_counts) == 734,
            "window_rows": window_rows == expected_windows,
            "prefix_no_lookahead": maximum_prefix_diff <= 1e-10,
            "postprocess_window_parity": maximum_post_diff <= 1e-12,
            "old_snapshot_immutable": first_snapshot_before == first_snapshot_after,
            "restart_snapshot_identical": final_before_restart == final_after_restart,
            "rerun_idempotent": repeated_final["idempotent"],
            "archive_complete": status["archived_pred_internal_rows"] == 734,
        }
        report = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "model_sha256": runtime.model_sha256,
            "state_sha256": runtime.state_sha256,
            "checks": checks,
            "evidence": {
                "dates": runtime.ordered_dates,
                "input_counts": input_counts,
                "window_rows": window_rows,
                "max_prefix_vs_full_pred_internal_abs_diff": maximum_prefix_diff,
                "max_direct_vs_stored_post_stage_abs_diff": maximum_post_diff,
                "first_snapshot_sha256": first_snapshot_after,
                "final_snapshot_sha256": final_after_restart,
                "status": status,
            },
        }
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "PASS" else 1
    finally:
        for suffix in ("", "-wal", "-shm"):
            Path(str(database) + suffix).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
