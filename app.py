"""Fail-closed FastAPI surface for frozen RINTIK daily inference."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import inference as inf
from daily_batch import BatchError, DailyBatchService, FrozenRuntime

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
MAX_BODY_BYTES = 5 * 1024 * 1024
MAX_ROWS = 1000

SERVICE: DailyBatchService | None = None
INITIALIZATION_ERROR: str | None = None


def initialize_service() -> DailyBatchService:
    global SERVICE, INITIALIZATION_ERROR
    try:
        runtime = FrozenRuntime(
            MODEL_PATH,
            STATE_PATH,
            train_path=TRAIN_PATH,
            test_path=TEST_PATH,
        )
        SERVICE = DailyBatchService(runtime, DATABASE_PATH)
        INITIALIZATION_ERROR = None
        return SERVICE
    except Exception as exc:
        SERVICE = None
        INITIALIZATION_ERROR = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("RINTIK runtime initialization failed")
        raise


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await asyncio.to_thread(initialize_service)
    except Exception:
        # Keep the API process alive so /health can expose an explicit unavailable
        # state. Prediction routes remain fail-closed with HTTP 503.
        pass
    yield


app = FastAPI(title="RINTIK Frozen Inference API", version="1.0.0", lifespan=lifespan)
allowed_origins = [
    item.strip()
    for item in os.environ.get(
        "RINTIK_CORS_ORIGINS",
        "http://127.0.0.1:4173,http://localhost:4173,http://127.0.0.1:5173,http://localhost:5173",
    ).split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Idempotency-Key"],
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


def frame_from_body(body: Any) -> pd.DataFrame:
    if not isinstance(body, dict):
        raise BatchError("INVALID_BODY", "Request body must be a JSON object.")
    if isinstance(body.get("rows"), list):
        rows = body["rows"]
        if len(rows) > MAX_ROWS:
            raise BatchError("TOO_MANY_ROWS", f"At most {MAX_ROWS} rows are accepted.", 413)
        return pd.DataFrame(rows)
    if isinstance(body.get("csv"), str):
        csv_text = body["csv"]
        if "\x00" in csv_text:
            raise BatchError("INVALID_CSV", "CSV text contains a NUL byte.")
        try:
            frame = pd.read_csv(io.StringIO(csv_text), on_bad_lines="error")
        except Exception as exc:
            raise BatchError("INVALID_CSV", "CSV could not be parsed safely.") from exc
        if len(frame) > MAX_ROWS:
            raise BatchError("TOO_MANY_ROWS", f"At most {MAX_ROWS} rows are accepted.", 413)
        return frame
    raise BatchError("INVALID_BODY", "Provide either a rows array or a csv string.")


@app.exception_handler(BatchError)
async def handle_batch_error(_: Request, exc: BatchError) -> JSONResponse:
    return error_response(exc.code, exc.message, exc.status_code)


@app.get("/health")
def health() -> JSONResponse:
    if SERVICE is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "ready": False,
                "source": None,
                "simulation": False,
                "error": INITIALIZATION_ERROR or "Runtime has not initialized.",
            },
        )
    return JSONResponse(status_code=200, content={"status": "ok", **SERVICE.status()})


@app.post("/v1/days/{requested_date}/ingest")
async def ingest_day(requested_date: str, request: Request) -> dict[str, Any]:
    service = require_service()
    body = await request_json(request)
    frame = frame_from_body(body)
    idempotency_key = request.headers.get("idempotency-key")
    if idempotency_key and len(idempotency_key) > 128:
        raise BatchError("INVALID_IDEMPOTENCY_KEY", "Idempotency-Key is limited to 128 characters.")
    return await asyncio.to_thread(service.ingest, requested_date, frame, idempotency_key)


@app.post("/v1/days/{requested_date}/run")
async def run_day(requested_date: str) -> dict[str, Any]:
    service = require_service()
    return await asyncio.to_thread(service.run, requested_date)


@app.get("/v1/days")
def list_days() -> dict[str, Any]:
    service = require_service()
    return {"days": service.list_days(), "runtime": service.status()}


@app.get("/v1/days/{requested_date}/result")
def day_result(requested_date: str) -> dict[str, Any]:
    service = require_service()
    return service.get_result(requested_date)


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
        "predictions": predictions,
        "stats": {
            "mean": round(float(values.mean()), 2),
            "min": round(float(values.min()), 2),
            "max": round(float(values.max()), 2),
        },
    }
