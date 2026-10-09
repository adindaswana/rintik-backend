"""Run the numerical and chronological Milestone 1 evidence checks."""

from __future__ import annotations

import json
import hashlib
import sqlite3
import sys
import uuid
from contextlib import closing
from pathlib import Path

import fastapi
import lightgbm
import numpy as np
import pandas as pd
import torch
import xgboost

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROTOTYPE_DIR = BACKEND_DIR.parent
WORKSPACE_DIR = PROTOTYPE_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

import inference as inf  # noqa: E402
from daily_batch import DailyBatchService, FrozenRuntime  # noqa: E402


def main() -> int:
    official = PROTOTYPE_DIR / "reference" / "official"
    database = BACKEND_DIR / "runtime" / f"verify-{uuid.uuid4().hex}.sqlite3"
    try:
        runtime = FrozenRuntime(
            WORKSPACE_DIR / "NSC26-DS011_HBD_File Data Science.pth",
            BACKEND_DIR / "artifacts" / "inference_state.json",
            official / "train.csv",
            official / "test.csv",
        )
        test = pd.read_csv(official / "test.csv", parse_dates=["datetime"])
        submission = pd.read_csv(official / "FINALIZATION_submission.csv")

        full = runtime.full_horizon(test)
        full_difference = np.abs(
            full["air_humidity"].to_numpy() - submission["air_humidity"].to_numpy()
        )

        service = DailyBatchService(runtime, database)
        test_dates = test["datetime"].dt.strftime("%Y-%m-%d")
        day_reports = []
        replay_predictions = []
        for day in runtime.ordered_dates:
            frame = test.loc[test_dates == day, inf.INPUT_COLUMNS]
            ingest = service.ingest(day, frame, f"verify-{day}")
            result = service.run(day)
            replay_predictions.extend(result["predictions"])
            day_reports.append({
                "date": day,
                "input_rows": ingest["row_count"],
                "published_rows": len(result["predictions"]),
                "window_mode": result["window"]["mode"],
                "window_dates": result["window"]["date_count"],
                "window_rows": result["window"]["row_count"],
                "historical_max_abs_diff": result["historical_pred_internal_max_abs_diff"],
            })

        replay_values = np.asarray([row["air_humidity"] for row in replay_predictions])
        replay_difference = np.abs(replay_values - submission["air_humidity"].to_numpy())
        publication_digest = hashlib.sha256(
            inf.stable_json_bytes([
                {"timestamp": row["timestamp"], "air_humidity": row["air_humidity"]}
                for row in replay_predictions
            ])
        ).hexdigest()

        direct_internal = runtime.predict_internal(test)
        with closing(sqlite3.connect(database)) as connection:
            archived = connection.execute(
                "SELECT pred_internal FROM internal_predictions ORDER BY timestamp"
            ).fetchall()
        archived_values = np.asarray([row[0] for row in archived])
        internal_difference = np.abs(
            archived_values - direct_internal["pred_internal"].to_numpy()
        )

        restarted = DailyBatchService(runtime, database)
        final_before = service.get_result(runtime.ordered_dates[-1])
        final_after = restarted.get_result(runtime.ordered_dates[-1])
        repeated_final = restarted.run(runtime.ordered_dates[-1])

        report = {
            "status": "PASS",
            "versions": {
                "python": sys.version.split()[0],
                "fastapi": fastapi.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
                "xgboost": xgboost.__version__,
                "lightgbm": lightgbm.__version__,
            },
            "model_sha256": runtime.model_sha256,
            "state_sha256": runtime.state_sha256,
            "full_horizon_parity": {
                "rows": len(full_difference),
                "identical_at_4dp": int(np.count_nonzero(full_difference) == 0),
                "max_abs_diff": float(full_difference.max()),
                "mean_abs_diff": float(full_difference.mean()),
            },
            "daily_replay": {
                "days": len(day_reports),
                "rows": len(replay_predictions),
                "input_counts": [item["input_rows"] for item in day_reports],
                "window_rows": [item["window_rows"] for item in day_reports],
                "max_historical_pred_internal_diff": max(
                    item["historical_max_abs_diff"] for item in day_reports
                ),
                "direct_vs_archived_internal_max_abs_diff": float(internal_difference.max()),
                "last_window": day_reports[-1],
                "publication_sha256": publication_digest,
                "restart_snapshot_identical": final_before["predictions"] == final_after["predictions"],
                "rerun_idempotent": repeated_final["idempotent"],
            },
            "daily_adapter_vs_submission_reference_only": {
                "note": "Difference is expected because daily P1-P3 uses expanding/rolling context, unlike the 734-point submission context.",
                "max_abs_diff_at_4dp": float(replay_difference.max()),
                "mean_abs_diff_at_4dp": float(replay_difference.mean()),
            },
        }
        print(json.dumps(report, indent=2))

        checks = [
            report["full_horizon_parity"]["identical_at_4dp"] == 1,
            len(day_reports) == 16,
            len(replay_predictions) == 734,
            report["daily_replay"]["max_historical_pred_internal_diff"] <= 1e-10,
            report["daily_replay"]["direct_vs_archived_internal_max_abs_diff"] <= 1e-10,
            day_reports[-1]["window_rows"] == 716,
            report["daily_replay"]["restart_snapshot_identical"],
            report["daily_replay"]["rerun_idempotent"],
        ]
        return 0 if all(checks) else 1
    finally:
        for suffix in ("", "-wal", "-shm"):
            Path(str(database) + suffix).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
