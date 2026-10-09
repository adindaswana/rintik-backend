"""Chronological, persistent daily-batch orchestration for frozen RINTIK."""

from __future__ import annotations

import hashlib
import io
import sqlite3
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import inference as inf


class BatchError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_iso_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise BatchError("INVALID_DATE", "Date must use YYYY-MM-DD format.") from exc
    if parsed.isoformat() != value:
        raise BatchError("INVALID_DATE", "Date must use YYYY-MM-DD format.")
    return parsed


class FrozenRuntime:
    def __init__(
        self,
        model_path: str | Path,
        sidecar_path: str | Path,
        train_path: str | Path | None = None,
        test_path: str | Path | None = None,
    ):
        self.model_path = Path(model_path)
        self.sidecar_path = Path(sidecar_path)
        self.artifact, self.model_sha256 = inf.load_frozen_artifact(self.model_path)
        self.sidecar, self.state_sha256 = inf.load_state_sidecar(
            self.sidecar_path, self.model_sha256
        )
        for key, path in (("train", train_path), ("test", test_path)):
            if path is None:
                continue
            source_path = Path(path)
            expected = self.sidecar["sources"][key]["sha256"]
            if not source_path.is_file() or inf.sha256_file(source_path) != expected:
                raise inf.ArtifactIntegrityError(
                    f"Official {key} source does not match the sidecar provenance"
                )
        self.boosters = inf.load_boosters(self.artifact)
        self.state = self.sidecar["inference_state"]
        self.manifests = {
            item["date"]: item for item in self.sidecar["official_test"]["days"]
        }
        self.ordered_dates = list(self.manifests)

    def predict_internal(self, frame: pd.DataFrame) -> pd.DataFrame:
        return inf.predict_internal(frame, self.artifact, self.boosters, self.state)

    def postprocess(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return inf.apply_postprocessing(
            frame["pred_internal"], frame["datetime"], self.artifact
        )

    def full_horizon(self, frame: pd.DataFrame) -> pd.DataFrame:
        return inf.predict_full_horizon(frame, self.artifact, self.boosters, self.state)


class DailyBatchService:
    def __init__(self, runtime: FrozenRuntime, database_path: str | Path):
        self.runtime = runtime
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize_database(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS daily_inputs (
                    date TEXT PRIMARY KEY,
                    payload_sha256 TEXT NOT NULL,
                    idempotency_key TEXT,
                    row_count INTEGER NOT NULL,
                    first_timestamp TEXT NOT NULL,
                    last_timestamp TEXT NOT NULL,
                    csv_payload TEXT NOT NULL,
                    ingested_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    date TEXT NOT NULL UNIQUE REFERENCES daily_inputs(date),
                    status TEXT NOT NULL CHECK(status = 'completed'),
                    payload_sha256 TEXT NOT NULL,
                    model_sha256 TEXT NOT NULL,
                    state_sha256 TEXT NOT NULL,
                    pipeline_version TEXT NOT NULL,
                    window_mode TEXT NOT NULL,
                    window_start_date TEXT NOT NULL,
                    window_end_date TEXT NOT NULL,
                    window_date_count INTEGER NOT NULL,
                    window_row_count INTEGER NOT NULL,
                    published_row_count INTEGER NOT NULL,
                    historical_max_abs_diff REAL NOT NULL,
                    completed_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS internal_predictions (
                    timestamp TEXT PRIMARY KEY,
                    date TEXT NOT NULL,
                    pred_internal REAL NOT NULL,
                    first_run_id TEXT NOT NULL REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS published_predictions (
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    timestamp TEXT NOT NULL,
                    date TEXT NOT NULL,
                    pred_internal REAL NOT NULL,
                    post_1 REAL NOT NULL,
                    post_2 REAL NOT NULL,
                    post_3 REAL NOT NULL,
                    air_humidity REAL NOT NULL,
                    PRIMARY KEY (run_id, timestamp)
                );

                CREATE INDEX IF NOT EXISTS idx_internal_date
                    ON internal_predictions(date);
                CREATE INDEX IF NOT EXISTS idx_published_date
                    ON published_predictions(date);
                """
            )
            connection.commit()

    @staticmethod
    def _csv_payload(frame: pd.DataFrame) -> str:
        normalized = inf.canonical_input_frame(frame)
        normalized["datetime"] = normalized["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return normalized.to_csv(
            index=False,
            lineterminator="\n",
            float_format="%.17g",
            na_rep="",
        )

    @staticmethod
    def _frame_from_payload(payload: str) -> pd.DataFrame:
        return pd.read_csv(io.StringIO(payload), parse_dates=["datetime"])

    def _validate_daily_frame(self, requested_date: str, frame: pd.DataFrame) -> pd.DataFrame:
        parse_iso_date(requested_date)
        if frame.empty:
            raise BatchError("EMPTY_INPUT", "Daily input contains no rows.")
        missing = [column for column in inf.INPUT_COLUMNS if column not in frame.columns]
        if missing:
            raise BatchError("MISSING_COLUMNS", f"Missing required columns: {', '.join(missing)}")
        extras = [column for column in frame.columns if column not in inf.INPUT_COLUMNS]
        if extras:
            raise BatchError(
                "UNEXPECTED_COLUMNS",
                f"Unexpected columns are not accepted in model input: {', '.join(extras)}",
            )
        try:
            normalized = inf.canonical_input_frame(frame)
        except (TypeError, ValueError) as exc:
            raise BatchError("INVALID_INPUT", str(exc)) from exc
        if normalized["datetime"].duplicated().any():
            raise BatchError("DUPLICATE_TIMESTAMP", "Daily input contains duplicate timestamps.")
        times = normalized["datetime"]
        aligned = (
            times.dt.minute.isin([0, 30])
            & times.dt.second.eq(0)
            & times.dt.microsecond.eq(0)
        )
        if not bool(aligned.all()):
            raise BatchError("TIMESTAMP_NOT_ALIGNED", "Timestamps must align to 30-minute slots.")
        present_dates = set(times.dt.strftime("%Y-%m-%d"))
        if present_dates != {requested_date}:
            raise BatchError("MULTIPLE_OR_WRONG_DATE", "Payload must contain only the requested date.")
        for column in inf.CRITICAL_FEATURES:
            if normalized[column].isna().all():
                raise BatchError(
                    "CRITICAL_FEATURE_EMPTY",
                    f"Critical feature {column} is missing for the entire day.",
                )
        manifest = self.runtime.manifests.get(requested_date)
        if manifest is None:
            raise BatchError(
                "UNSUPPORTED_SCENARIO_DATE",
                "This milestone accepts only dates from the official NSC test scenario.",
            )
        payload_sha = inf.canonical_frame_sha256(normalized)
        if payload_sha != manifest["payload_sha256"]:
            raise BatchError(
                "OFFICIAL_FIXTURE_MISMATCH",
                "Payload differs from the pinned official test-day fixture.",
            )
        return normalized

    def ingest(
        self,
        requested_date: str,
        frame: pd.DataFrame,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        normalized = self._validate_daily_frame(requested_date, frame)
        payload_sha = inf.canonical_frame_sha256(normalized)
        manifest = self.runtime.manifests[requested_date]
        payload = self._csv_payload(normalized)

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM daily_inputs WHERE date = ?", (requested_date,)
            ).fetchone()
            if existing:
                if existing["payload_sha256"] != payload_sha:
                    raise BatchError(
                        "INPUT_CONFLICT",
                        "A different payload is already stored for this date.",
                        409,
                    )
                connection.commit()
                return self._ingest_result(existing, idempotent=True)

            count = connection.execute("SELECT COUNT(*) FROM daily_inputs").fetchone()[0]
            if count >= len(self.runtime.ordered_dates):
                raise BatchError("SCENARIO_COMPLETE", "All official test dates are already ingested.", 409)
            expected_date = self.runtime.ordered_dates[count]
            if requested_date != expected_date:
                raise BatchError(
                    "OUT_OF_ORDER_DATE",
                    f"Expected next date {expected_date}, received {requested_date}.",
                    409,
                )
            if count:
                previous_date = self.runtime.ordered_dates[count - 1]
                previous_run = connection.execute(
                    "SELECT 1 FROM runs WHERE date = ?", (previous_date,)
                ).fetchone()
                if not previous_run:
                    raise BatchError(
                        "PREVIOUS_DAY_NOT_RUN",
                        f"Complete the batch for {previous_date} before ingesting the next date.",
                        409,
                    )

            connection.execute(
                """
                INSERT INTO daily_inputs (
                    date, payload_sha256, idempotency_key, row_count,
                    first_timestamp, last_timestamp, csv_payload, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    requested_date,
                    payload_sha,
                    idempotency_key,
                    len(normalized),
                    manifest["first_timestamp"],
                    manifest["last_timestamp"],
                    payload,
                    utc_now(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM daily_inputs WHERE date = ?", (requested_date,)
            ).fetchone()
            connection.commit()
            return self._ingest_result(row, idempotent=False)

    @staticmethod
    def _ingest_result(row: sqlite3.Row, idempotent: bool) -> dict[str, Any]:
        return {
            "date": row["date"],
            "row_count": row["row_count"],
            "first_timestamp": row["first_timestamp"],
            "last_timestamp": row["last_timestamp"],
            "payload_sha256": row["payload_sha256"],
            "idempotent": idempotent,
            "status": "ingested",
        }

    def _stored_prefix(self, connection: sqlite3.Connection, through_date: str) -> tuple[list[str], pd.DataFrame]:
        rows = connection.execute(
            "SELECT date, csv_payload FROM daily_inputs WHERE date <= ? ORDER BY date",
            (through_date,),
        ).fetchall()
        dates = [row["date"] for row in rows]
        if not dates or dates[-1] != through_date:
            raise BatchError("DATE_NOT_INGESTED", "Ingest this date before running inference.", 404)
        expected = self.runtime.ordered_dates[:len(dates)]
        if dates != expected:
            raise BatchError("STATE_CHRONOLOGY_ERROR", "Stored dates are not a valid chronological prefix.", 500)
        frames = [self._frame_from_payload(row["csv_payload"]) for row in rows]
        return dates, pd.concat(frames, ignore_index=True)

    def run(self, requested_date: str) -> dict[str, Any]:
        parse_iso_date(requested_date)
        with closing(self._connect()) as connection:
            existing = connection.execute(
                "SELECT run_id FROM runs WHERE date = ?", (requested_date,)
            ).fetchone()
            if existing:
                return self.get_result(requested_date, connection=connection, idempotent=True)
            dates, prefix = self._stored_prefix(connection, requested_date)
            payload_row = connection.execute(
                "SELECT payload_sha256 FROM daily_inputs WHERE date = ?", (requested_date,)
            ).fetchone()

        internal = self.runtime.predict_internal(prefix)
        timestamp_strings = internal["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        current_date_values = internal["datetime"].dt.strftime("%Y-%m-%d")

        historical_max_abs_diff = 0.0
        with closing(self._connect()) as connection:
            historical = connection.execute(
                "SELECT timestamp, pred_internal FROM internal_predictions ORDER BY timestamp"
            ).fetchall()
        if historical:
            current_by_timestamp = dict(zip(timestamp_strings, internal["pred_internal"]))
            differences = []
            for row in historical:
                if row["timestamp"] not in current_by_timestamp:
                    raise BatchError(
                        "TEMPORAL_STATE_DIVERGENCE",
                        "A previously archived timestamp disappeared from replay.",
                        500,
                    )
                differences.append(abs(row["pred_internal"] - current_by_timestamp[row["timestamp"]]))
            historical_max_abs_diff = float(max(differences, default=0.0))
            if historical_max_abs_diff > 1e-10:
                raise BatchError(
                    "TEMPORAL_STATE_DIVERGENCE",
                    f"Historical pred_internal changed by {historical_max_abs_diff:.12g}.",
                    500,
                )

        active_dates = dates[-15:]
        window = internal.loc[current_date_values.isin(active_dates)].reset_index(drop=True)
        stages = self.runtime.postprocess(window)
        for name, values in stages.items():
            window[name] = values
        window["date"] = window["datetime"].dt.strftime("%Y-%m-%d")
        published = window.loc[window["date"] == requested_date].copy()
        published["air_humidity"] = np.round(published["post_3"], 4)
        if len(published) != self.runtime.manifests[requested_date]["row_count"]:
            raise BatchError("PUBLICATION_ROW_MISMATCH", "Published row count differs from manifest.", 500)

        run_material = "|".join([
            requested_date,
            payload_row["payload_sha256"],
            self.runtime.model_sha256,
            self.runtime.state_sha256,
            inf.PIPELINE_VERSION,
        ])
        run_id = hashlib.sha256(run_material.encode("utf-8")).hexdigest()[:24]
        mode = "expanding" if len(dates) <= 15 else "rolling_15_calendar_days"

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            concurrent = connection.execute(
                "SELECT run_id FROM runs WHERE date = ?", (requested_date,)
            ).fetchone()
            if concurrent:
                connection.commit()
                return self.get_result(requested_date, idempotent=True)
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, date, status, payload_sha256, model_sha256, state_sha256,
                    pipeline_version, window_mode, window_start_date, window_end_date,
                    window_date_count, window_row_count, published_row_count,
                    historical_max_abs_diff, completed_at
                ) VALUES (?, ?, 'completed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, requested_date, payload_row["payload_sha256"],
                    self.runtime.model_sha256, self.runtime.state_sha256,
                    inf.PIPELINE_VERSION, mode, active_dates[0], active_dates[-1],
                    len(active_dates), len(window), len(published),
                    historical_max_abs_diff, utc_now(),
                ),
            )
            for timestamp, day_value, value in zip(
                timestamp_strings, current_date_values, internal["pred_internal"]
            ):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO internal_predictions
                        (timestamp, date, pred_internal, first_run_id)
                    VALUES (?, ?, ?, ?)
                    """,
                    (timestamp, day_value, float(value), run_id),
                )
            for row in published.itertuples(index=False):
                connection.execute(
                    """
                    INSERT INTO published_predictions (
                        run_id, timestamp, date, pred_internal,
                        post_1, post_2, post_3, air_humidity
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id, row.datetime.strftime("%Y-%m-%d %H:%M:%S"),
                        requested_date, float(row.pred_internal), float(row.post_1),
                        float(row.post_2), float(row.post_3), float(row.air_humidity),
                    ),
                )
            connection.commit()
        return self.get_result(requested_date, idempotent=False)

    def get_result(
        self,
        requested_date: str,
        connection: sqlite3.Connection | None = None,
        idempotent: bool = False,
    ) -> dict[str, Any]:
        owns_connection = connection is None
        connection = connection or self._connect()
        try:
            run = connection.execute(
                "SELECT * FROM runs WHERE date = ?", (requested_date,)
            ).fetchone()
            if not run:
                raise BatchError("RUN_NOT_FOUND", "No completed run exists for this date.", 404)
            rows = connection.execute(
                """
                SELECT timestamp, pred_internal, post_1, post_2, post_3, air_humidity
                FROM published_predictions WHERE run_id = ? ORDER BY timestamp
                """,
                (run["run_id"],),
            ).fetchall()
            return {
                "run_id": run["run_id"],
                "date": run["date"],
                "status": run["status"],
                "source": "real_original_pth",
                "simulation": False,
                "pipeline_version": run["pipeline_version"],
                "model_sha256": run["model_sha256"],
                "state_sha256": run["state_sha256"],
                "window": {
                    "mode": run["window_mode"],
                    "start_date": run["window_start_date"],
                    "end_date": run["window_end_date"],
                    "date_count": run["window_date_count"],
                    "row_count": run["window_row_count"],
                },
                "historical_pred_internal_max_abs_diff": run["historical_max_abs_diff"],
                "idempotent": idempotent,
                "predictions": [dict(row) for row in rows],
            }
        finally:
            if owns_connection:
                connection.close()

    def list_days(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT i.date, i.row_count, i.first_timestamp, i.last_timestamp,
                       i.payload_sha256, i.ingested_at,
                       CASE WHEN r.run_id IS NULL THEN 'ingested' ELSE 'completed' END AS status,
                       r.run_id
                FROM daily_inputs i LEFT JOIN runs r ON r.date = i.date
                ORDER BY i.date
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def status(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            ingested = connection.execute("SELECT COUNT(*) FROM daily_inputs").fetchone()[0]
            completed = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            archived = connection.execute("SELECT COUNT(*) FROM internal_predictions").fetchone()[0]
        next_date = (
            self.runtime.ordered_dates[ingested]
            if ingested < len(self.runtime.ordered_dates)
            else None
        )
        return {
            "ready": True,
            "source": "real_original_pth",
            "simulation": False,
            "model_sha256": self.runtime.model_sha256,
            "state_sha256": self.runtime.state_sha256,
            "pipeline_version": inf.PIPELINE_VERSION,
            "ingested_days": ingested,
            "completed_days": completed,
            "archived_pred_internal_rows": archived,
            "next_expected_date": next_date,
        }
