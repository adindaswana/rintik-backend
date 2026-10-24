"""Fail-closed FastAPI surface for frozen RINTIK daily inference."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import os
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

import inference as inf
import sensor_check
from daily_batch import BatchError, DailyBatchService, FrozenRuntime
from monitoring import MonitoringService
from operational_simulation import OperationalSimulationService
from operator_api import OperatorDataService, feature_catalog
from sessions import (
    SESSION_HEADER,
    Session,
    SessionRegistry,
    parse_session_header,
    session_digest,
)

LOGGER = logging.getLogger("rintik.backend")
BACKEND_DIR = Path(__file__).resolve().parent
PROTOTYPE_DIR = BACKEND_DIR.parent
WORKSPACE_DIR = PROTOTYPE_DIR.parent

MODEL_PATH = Path(os.environ.get(
    "RINTIK_MODEL_PATH",
    WORKSPACE_DIR / "NSC26-DS011_HBD_File Data Science.pth",
))
STATE_PATH = Path(os.environ.get(
    "RINTIK_STATE_PATH",
    BACKEND_DIR / "artifacts" / "inference_state.json",
))
DATABASE_PATH = Path(os.environ.get(
    "RINTIK_DATABASE_PATH",
    BACKEND_DIR / "runtime" / "rintik.sqlite3",
))
TRAIN_PATH = Path(os.environ.get(
    "RINTIK_TRAIN_PATH",
    PROTOTYPE_DIR / "reference" / "official" / "train.csv",
))
TEST_PATH = Path(os.environ.get(
    "RINTIK_TEST_PATH",
    PROTOTYPE_DIR / "reference" / "official" / "test.csv",
))
SYNTHETIC_FIXTURE_PATH = Path(os.environ.get(
    "RINTIK_SYNTHETIC_FIXTURE_PATH",
    PROTOTYPE_DIR / "demo" / "fixtures" / "synthetic_physical_humidity_SENSOR_SIMULATION.csv",
))
SYNTHETIC_PROVENANCE_PATH = Path(os.environ.get(
    "RINTIK_SYNTHETIC_PROVENANCE_PATH",
    PROTOTYPE_DIR / "demo" / "fixtures" / "SOURCE_PROVENANCE.json",
))
OPERATIONAL_FIXTURE_PATH = Path(os.environ.get(
    "RINTIK_OPERATIONAL_FIXTURE_PATH",
    BACKEND_DIR / "fixtures" / "operational_2026_exogenous.csv",
))
OPERATIONAL_PROVENANCE_PATH = Path(os.environ.get(
    "RINTIK_OPERATIONAL_PROVENANCE_PATH",
    BACKEND_DIR / "fixtures" / "operational_2026_provenance.json",
))
SESSION_DIR = Path(os.environ.get(
    "RINTIK_SESSION_DIR",
    DATABASE_PATH.parent / "sessions",
))
MAX_SESSIONS = int(os.environ.get("RINTIK_MAX_SESSIONS", "64"))
MAX_BODY_BYTES = 5 * 1024 * 1024
MAX_ROWS = 1000
# Pemeriksaan Sensor promises (UI, Vercel proxy, README) up to 2000 uploaded rows; it is stateless.
MONITOR_MAX_ROWS = 2000

SERVICE: DailyBatchService | None = None
MONITORING: MonitoringService | None = None
OPERATIONAL: OperationalSimulationService | None = None
INITIALIZATION_ERROR: str | None = None
MONITORING_INITIALIZATION_ERROR: str | None = None
# Visitor sessions: one SQLite file each, sharing the single frozen runtime.
SESSIONS = SessionRegistry(SESSION_DIR, MAX_SESSIONS)


def initialize_service() -> DailyBatchService:
    global SERVICE, MONITORING, OPERATIONAL, INITIALIZATION_ERROR, MONITORING_INITIALIZATION_ERROR
    try:
        runtime = FrozenRuntime(
            MODEL_PATH,
            STATE_PATH,
            train_path=TRAIN_PATH,
            test_path=TEST_PATH,
        )
        SERVICE = DailyBatchService(runtime, DATABASE_PATH)
        INITIALIZATION_ERROR = None
    except Exception as exc:
        SERVICE = None
        MONITORING = None
        INITIALIZATION_ERROR = "Frozen model runtime failed integrity or startup checks."
        MONITORING_INITIALIZATION_ERROR = "Model runtime is unavailable."
        LOGGER.exception("RINTIK runtime initialization failed")
        raise
    try:
        MONITORING = MonitoringService(
            SERVICE,
            SYNTHETIC_FIXTURE_PATH,
            SYNTHETIC_PROVENANCE_PATH,
        )
        MONITORING_INITIALIZATION_ERROR = None
    except Exception as exc:
        MONITORING = None
        MONITORING_INITIALIZATION_ERROR = (
            "Synthetic monitoring fixture failed integrity or startup checks."
        )
        LOGGER.exception("RINTIK synthetic monitoring initialization failed")
    try:
        OPERATIONAL = OperationalSimulationService(
            SERVICE.runtime,
            DATABASE_PATH,
            OPERATIONAL_FIXTURE_PATH,
            OPERATIONAL_PROVENANCE_PATH,
        )
    except Exception:
        OPERATIONAL = None
        LOGGER.exception("RINTIK operational simulation initialization failed")
    return SERVICE


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await asyncio.to_thread(initialize_service)
    except Exception:
        # Keep the API process alive so /health can expose an explicit unavailable
        # state. Prediction routes remain fail-closed with HTTP 503.
        pass
    yield


app = FastAPI(title="RINTIK Frozen Inference API", version="1.2.0", lifespan=lifespan)
allowed_origins = [
    item.strip()
    for item in os.environ.get(
        "RINTIK_CORS_ORIGINS",
        "http://127.0.0.1:4173,http://localhost:4173,http://127.0.0.1:4174,http://localhost:4174,http://127.0.0.1:5173,http://localhost:5173",
    ).split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["Content-Type", "Idempotency-Key", SESSION_HEADER],
)


def error_response(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def require_service() -> DailyBatchService:
    if SERVICE is None:
        raise BatchError(
            "MODEL_UNAVAILABLE",
            "Frozen model runtime is unavailable; no fallback prediction was produced.",
            503,
        )
    return SERVICE


def default_session() -> Session:
    """Legacy shared state, used when a request carries no session header."""
    return Session(
        scoped=False,
        id_hash=None,
        daily=require_service(),
        monitoring=MONITORING,
        operational=OPERATIONAL,
    )


def current_session(request: Request) -> Iterator[Session]:
    """Resolve X-Rintik-Session (absent => default); invalid => HTTP 400."""
    session_id = parse_session_header(request.headers.get(SESSION_HEADER))
    with SESSIONS.use(session_id, default_session()) as session:
        yield session


def require_monitoring(session: Session) -> MonitoringService:
    if session.monitoring is None:
        raise BatchError(
            "MONITORING_UNAVAILABLE",
            "Synthetic demo monitoring is unavailable; real model results were not replaced.",
            503,
        )
    return session.monitoring


def require_operational(session: Session) -> OperationalSimulationService:
    if session.operational is None:
        raise BatchError(
            "OPERATIONAL_SIMULATION_UNAVAILABLE",
            "The isolated operational simulation is unavailable; no fallback result was produced.",
            503,
        )
    return session.operational


async def request_json(request: Request) -> Any:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_BODY_BYTES:
                raise BatchError("PAYLOAD_TOO_LARGE", "Request body exceeds 5 MiB.", 413)
        except ValueError as exc:
            raise BatchError("INVALID_CONTENT_LENGTH", "Invalid Content-Length header.") from exc
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise BatchError("PAYLOAD_TOO_LARGE", "Request body exceeds 5 MiB.", 413)
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BatchError("INVALID_JSON", "Request body must be valid JSON.") from exc


def frame_from_body(body: Any, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    if not isinstance(body, dict):
        raise BatchError("INVALID_BODY", "Request body must be a JSON object.")
    if isinstance(body.get("rows"), list):
        rows = body["rows"]
        if len(rows) > max_rows:
            raise BatchError("TOO_MANY_ROWS", f"At most {max_rows} rows are accepted.", 413)
        return pd.DataFrame(rows)
    if isinstance(body.get("csv"), str):
        csv_text = body["csv"]
        if "\x00" in csv_text:
            raise BatchError("INVALID_CSV", "CSV text contains a NUL byte.")
        try:
            frame = pd.read_csv(io.StringIO(csv_text), on_bad_lines="error")
        except Exception as exc:
            raise BatchError("INVALID_CSV", "CSV could not be parsed safely.") from exc
        if len(frame) > max_rows:
            raise BatchError("TOO_MANY_ROWS", f"At most {max_rows} rows are accepted.", 413)
        return frame
    raise BatchError("INVALID_BODY", "Provide either a rows array or a csv string.")


@app.exception_handler(BatchError)
async def handle_batch_error(_: Request, exc: BatchError) -> JSONResponse:
    return error_response(exc.code, exc.message, exc.status_code)


@app.get("/health")
def health(request: Request) -> JSONResponse:
    session_id = parse_session_header(request.headers.get(SESSION_HEADER))
    session_info = {
        "id_hash": session_digest(session_id)[:8] if session_id else None,
        "scoped": session_id is not None,
    }
    if SERVICE is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "ready": False,
                "source": None,
                "simulation": False,
                "error": INITIALIZATION_ERROR or "Runtime has not initialized.",
                "session": session_info,
            },
        )
    with SESSIONS.use(session_id, default_session()) as session:
        monitoring_status = (
            session.monitoring.status()
            if session.monitoring is not None
            else {
                "monitoring_ready": False,
                "monitoring_mode": "synthetic_demo",
                "monitoring_simulation": True,
                "monitoring_error": MONITORING_INITIALIZATION_ERROR
                or "Synthetic monitoring is unavailable for this session.",
            }
        )
        state = "ok" if session.monitoring is not None else "degraded"
        operational_status = (
            session.operational.list_days()
            if session.operational is not None
            else {"mode": "synthetic-operational-2026-v1", "simulation": True, "ready": False}
        )
        return JSONResponse(
            status_code=200,
            content={
                "status": state,
                **session.daily.status(),
                **monitoring_status,
                "operational_simulation": operational_status,
                "session": session.describe(),
            },
        )


@app.get("/v1/operational/days")
def list_operational_days(session: Session = Depends(current_session)) -> dict[str, Any]:
    return require_operational(session).list_days()


@app.get("/v1/operational/days/{requested_date}")
def get_operational_day(requested_date: str, session: Session = Depends(current_session)) -> dict[str, Any]:
    return require_operational(session).get_day(requested_date)


@app.get("/v1/operational/days/{requested_date}/fixture")
def get_operational_fixture_reading(requested_date: str, timestamp: str, session: Session = Depends(current_session)) -> dict[str, Any]:
    return require_operational(session).get_fixture_reading(requested_date, timestamp)


@app.post("/v1/operational/days/{requested_date}/readings")
async def create_operational_reading(requested_date: str, request: Request, session: Session = Depends(current_session)) -> dict[str, Any]:
    body = await request_json(request)
    if not isinstance(body, dict) or set(body) != {"reading"}:
        raise BatchError("INVALID_BODY", "Provide exactly one reading object.")
    return await asyncio.to_thread(require_operational(session).save_manual, requested_date, body["reading"], False)


@app.put("/v1/operational/days/{requested_date}/readings")
async def update_operational_reading(requested_date: str, request: Request, session: Session = Depends(current_session)) -> dict[str, Any]:
    body = await request_json(request)
    if not isinstance(body, dict) or set(body) != {"reading"}:
        raise BatchError("INVALID_BODY", "Provide exactly one reading object.")
    return await asyncio.to_thread(require_operational(session).save_manual, requested_date, body["reading"], True)


@app.post("/v1/operational/days/{requested_date}/fill")
async def fill_operational_day(requested_date: str, session: Session = Depends(current_session)) -> dict[str, Any]:
    return await asyncio.to_thread(require_operational(session).fill_missing, requested_date)


@app.post("/v1/operational/days/{requested_date}/run")
async def run_operational_day(requested_date: str, request: Request, session: Session = Depends(current_session)) -> dict[str, Any]:
    body = await request_json(request)
    if not isinstance(body, dict) or set(body) != {"humidity_scenario"}:
        raise BatchError("INVALID_BODY", "Provide exactly one humidity_scenario.")
    return await asyncio.to_thread(require_operational(session).run, requested_date, body["humidity_scenario"])


@app.get("/v1/operational/days/{requested_date}/result")
def get_operational_result(requested_date: str, session: Session = Depends(current_session)) -> dict[str, Any]:
    return require_operational(session).get_result(requested_date)


@app.get("/v1/operational/days/{requested_date}/series")
def get_operational_series(requested_date: str, session: Session = Depends(current_session)) -> dict[str, Any]:
    return require_operational(session).get_series(requested_date)


@app.post("/v1/operational/reset")
async def reset_operational_simulation(request: Request, session: Session = Depends(current_session)) -> dict[str, Any]:
    body = await request_json(request)
    if body != {"confirmation": "RESET_OPERATIONAL_SIMULATION"}:
        raise BatchError(
            "RESET_CONFIRMATION_REQUIRED",
            "Operational reset requires RESET_OPERATIONAL_SIMULATION.",
        )
    return await asyncio.to_thread(require_operational(session).reset)


@app.post("/v1/days/{requested_date}/ingest")
async def ingest_day(requested_date: str, request: Request, session: Session = Depends(current_session)) -> dict[str, Any]:
    service = session.daily
    body = await request_json(request)
    frame = frame_from_body(body)
    idempotency_key = request.headers.get("idempotency-key")
    if idempotency_key and len(idempotency_key) > 128:
        raise BatchError("INVALID_IDEMPOTENCY_KEY", "Idempotency-Key is limited to 128 characters.")
    return await asyncio.to_thread(service.ingest, requested_date, frame, idempotency_key)


@app.post("/v1/days/{requested_date}/run")
async def run_day(requested_date: str, session: Session = Depends(current_session)) -> dict[str, Any]:
    service = session.daily
    return await asyncio.to_thread(service.run, requested_date)


@app.get("/v1/days")
def list_days(session: Session = Depends(current_session)) -> dict[str, Any]:
    service = session.daily
    return {"days": service.list_days(), "runtime": service.status()}


@app.get("/v1/demo/official-input/next")
def next_official_demo_input(session: Session = Depends(current_session)) -> dict[str, Any]:
    """Return the next pinned official day as input, never as a prediction."""
    service = session.daily
    next_date = service.status()["next_expected_date"]
    if next_date is None:
        raise BatchError(
            "SCENARIO_COMPLETE",
            "All official test dates are already processed; no next demo input exists.",
            409,
        )

    expected_source_sha256 = service.runtime.sidecar["sources"]["test"]["sha256"]
    if not TEST_PATH.is_file() or inf.sha256_file(TEST_PATH) != expected_source_sha256:
        raise BatchError(
            "OFFICIAL_SOURCE_UNAVAILABLE",
            "The pinned official test source failed its integrity check.",
            503,
        )
    try:
        frame = inf.canonical_input_frame(pd.read_csv(TEST_PATH))
    except Exception as exc:
        raise BatchError(
            "OFFICIAL_SOURCE_UNAVAILABLE",
            "The pinned official test source could not be read safely.",
            503,
        ) from exc
    dates = frame["datetime"].dt.strftime("%Y-%m-%d")
    day = frame.loc[dates == next_date, inf.INPUT_COLUMNS].copy()
    manifest = service.runtime.manifests[next_date]
    payload_sha256 = inf.canonical_frame_sha256(day)
    if len(day) != manifest["row_count"] or payload_sha256 != manifest["payload_sha256"]:
        raise BatchError(
            "OFFICIAL_SOURCE_UNAVAILABLE",
            "The next official daily fixture failed its integrity check.",
            503,
        )
    return {
        "mode": "official_test_fixture_demo",
        "source": "official_test_csv",
        "simulation": False,
        "prediction_produced": False,
        "date": next_date,
        "row_count": len(day),
        "payload_sha256": payload_sha256,
        "official_test_sha256": expected_source_sha256,
        "csv": day.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S"),
    }


@app.post("/v1/demo/reset")
async def reset_demo_state(request: Request, session: Session = Depends(current_session)) -> dict[str, Any]:
    """Explicitly reset local demo persistence; immutable artifacts are untouched."""
    monitoring = require_monitoring(session)
    body = await request_json(request)
    if body != {"confirmation": "RESET_DEMO_STATE"}:
        raise BatchError(
            "RESET_CONFIRMATION_REQUIRED",
            "Demo reset requires the exact confirmation RESET_DEMO_STATE.",
        )
    return await asyncio.to_thread(monitoring.reset_demo_state)


@app.get("/v1/days/{requested_date}/result")
def day_result(requested_date: str, session: Session = Depends(current_session)) -> dict[str, Any]:
    service = session.daily
    return service.get_result(requested_date)


@app.get("/v1/features")
def list_features() -> dict[str, Any]:
    require_service()
    return feature_catalog()


@app.get("/v1/days/{requested_date}/series")
def day_sensor_series(requested_date: str, session: Session = Depends(current_session)) -> dict[str, Any]:
    service = session.daily
    return OperatorDataService(service).get_series(requested_date)


@app.get("/v1/days/{requested_date}/export.csv")
def export_day(requested_date: str, session: Session = Depends(current_session)) -> Response:
    service = session.daily
    csv_text = OperatorDataService(service).export_day_csv(requested_date)
    return Response(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="rintik-{requested_date}.csv"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.post("/v1/demo/days/{requested_date}/monitor")
async def monitor_day(requested_date: str, session: Session = Depends(current_session)) -> dict[str, Any]:
    monitoring = require_monitoring(session)
    return await asyncio.to_thread(monitoring.monitor_day, requested_date)


@app.get("/v1/demo/days/{requested_date}/monitoring")
def monitoring_result(requested_date: str, session: Session = Depends(current_session)) -> dict[str, Any]:
    monitoring = require_monitoring(session)
    return monitoring.get_monitoring(requested_date)


@app.get("/v1/demo/incidents")
def list_demo_incidents(session: Session = Depends(current_session)) -> dict[str, Any]:
    monitoring = require_monitoring(session)
    return {"mode": "synthetic_demo", "simulation": True, "incidents": monitoring.list_incidents()}


@app.post("/v1/demo/audits")
async def create_demo_audit(request: Request, session: Session = Depends(current_session)) -> dict[str, Any]:
    monitoring = require_monitoring(session)
    body = await request_json(request)
    if not isinstance(body, dict) or set(body) != {"scenario"}:
        raise BatchError("INVALID_BODY", "Provide exactly one demo audit scenario.")
    return await asyncio.to_thread(
        monitoring.create_demo_audit,
        body["scenario"],
        request.headers.get("idempotency-key"),
    )


@app.get("/v1/demo/audits/current")
def current_demo_audit(session: Session = Depends(current_session)) -> dict[str, Any]:
    monitoring = require_monitoring(session)
    return monitoring.get_current_audit()


@app.post("/v1/demo/incidents/{incident_id}/confirm")
async def confirm_demo_incident(incident_id: str, request: Request, session: Session = Depends(current_session)) -> dict[str, Any]:
    monitoring = require_monitoring(session)
    body = await request_json(request)
    if not isinstance(body, dict) or set(body) - {"demo_override", "note"}:
        raise BatchError(
            "INVALID_BODY",
            "Confirmation accepts only demo_override and an optional note.",
        )
    if not isinstance(body.get("demo_override", False), bool):
        raise BatchError("INVALID_BODY", "demo_override must be a boolean.")
    note = body.get("note")
    if note is not None and not isinstance(note, str):
        raise BatchError("INVALID_BODY", "note must be a string when provided.")
    return await asyncio.to_thread(
        monitoring.confirm_incident,
        incident_id,
        body.get("demo_override", False),
        note,
        request.headers.get("idempotency-key"),
    )


@app.post("/predict")
async def predict_full_submission(request: Request) -> dict[str, Any]:
    """Compatibility endpoint, intentionally restricted to the exact 734-row fixture."""
    service = require_service()
    body = await request_json(request)
    frame = frame_from_body(body)
    try:
        normalized = inf.canonical_input_frame(frame)
    except (TypeError, ValueError) as exc:
        raise BatchError("INVALID_INPUT", str(exc)) from exc
    official = service.runtime.sidecar["official_test"]
    if (
        len(normalized) != official["total_rows"]
        or inf.canonical_frame_sha256(normalized)
        != official["payload_sha256"]
    ):
        raise BatchError(
            "OFFICIAL_FIXTURE_MISMATCH",
            "The compatibility endpoint accepts only the exact full official test fixture.",
        )
    output = await asyncio.to_thread(service.runtime.full_horizon, normalized)
    predictions = [
        {
            "datetime": row.datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "air_humidity": float(row.air_humidity),
        }
        for row in output.itertuples(index=False)
    ]
    values = output["post_3"].to_numpy()
    return {
        "count": len(predictions),
        "source": "real_original_pth",
        "simulation": False,
        "model_sha256": service.runtime.model_sha256,
        "state_sha256": service.runtime.state_sha256,
        "pipeline_version": inf.PIPELINE_VERSION,
        "predictions": predictions,
        "stats": {
            "mean": round(float(values.mean()), 2),
            "min": round(float(values.min()), 2),
            "max": round(float(values.max()), 2),
        },
    }


@app.post("/monitor")
async def monitor_uploaded_sensor(request: Request) -> dict[str, Any]:
    """Pemeriksaan Sensor (stateless): bandingkan air_humidity unggahan dengan estimasi model.

    Estimasi dibuat dari runtime beku yang sama dengan /predict; kolom air_humidity
    dibuang sebelum inferensi. Aturan demo-v1 belum dikalibrasi, hasilnya tidak
    diarsipkan, dan bacaan sensor unggahan bukan ground truth. Sinyal ini terpisah
    dari monitoring simulasi berarsip di /v1/demo/*.
    """
    service = require_service()
    body = await request_json(request)
    frame = frame_from_body(body, MONITOR_MAX_ROWS)
    comparator = body.get("comparator") or "p3"
    if comparator not in sensor_check.RULES["center"]:
        raise BatchError("INVALID_COMPARATOR", "comparator must be p3 or inti.")
    center = sensor_check.RULES["center"][comparator]
    if body.get("center") is not None:
        try:
            center = float(body["center"])
        except (TypeError, ValueError) as exc:
            raise BatchError("INVALID_CENTER", "center must be a number.") from exc
        if not math.isfinite(center):
            raise BatchError("INVALID_CENTER", "center must be a finite number.")
    sensor, message = sensor_check.parse_sensor(frame)
    if message:
        raise BatchError("INVALID_SENSOR_READINGS", message)
    try:
        features = frame.drop(columns=[sensor_check.SENSOR_COL])
        normalized = inf.canonical_input_frame(features)
    except (TypeError, ValueError) as exc:
        raise BatchError("INVALID_INPUT", str(exc)) from exc
    if normalized["datetime"].duplicated().any():
        raise BatchError("DUPLICATE_TIMESTAMP", "Each timestamp may appear only once.")
    sensor_by_time = pd.Series(
        sensor.to_numpy(),
        index=pd.to_datetime(frame["datetime"], errors="coerce"),
    )

    def run_estimate() -> tuple[pd.DataFrame, dict[str, Any]]:
        internal = service.runtime.predict_internal(normalized)
        return internal, service.runtime.postprocess(internal)

    try:
        internal, post = await asyncio.to_thread(run_estimate)
    except ValueError as exc:
        raise BatchError("MONITOR_INPUT_OUT_OF_RANGE", str(exc)) from exc
    inti = internal["pred_internal"].to_numpy()
    p3 = np.asarray(post["post_3"], dtype=float)
    observed = sensor_by_time.reindex(pd.to_datetime(internal["datetime"])).to_numpy()
    reference = p3 if comparator == "p3" else inti
    days = sensor_check.evaluate_days(internal["datetime"], observed, reference, center)
    summary = {status: 0 for status in sensor_check.STATUSES}
    for day in days:
        summary[day["status"]] += 1
    stamps = pd.to_datetime(internal["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    predictions = [
        {
            "datetime": stamp,
            "sensor": sensor_check._rnd(reading, 4),
            "p3": round(float(final), 4),
            "inti": round(float(core), 4),
        }
        for stamp, reading, final, core in zip(stamps, observed, p3, inti)
    ]
    return {
        "count": len(predictions),
        "comparator": comparator,
        "rule_version": sensor_check.RULES["version"],
        "center": center,
        "predictions": predictions,
        "days": days,
        "summary": summary,
        "rules": sensor_check.RULES,
        "provenance": {
            "source": "real_original_pth",
            "simulation": False,
            "model_sha256": service.runtime.model_sha256,
            "state_sha256": service.runtime.state_sha256,
            "pipeline_version": inf.PIPELINE_VERSION,
            "thresholds": "illustrative_unvalidated",
            "sensor_reading_role": "user_uploaded_comparator_not_ground_truth",
            "archived": False,
        },
    }
