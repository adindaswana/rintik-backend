"""Backend prediksi kelembapan udara (FastAPI).

Endpoint:
  GET  /health   -> {"status": "ok"}
  POST /predict  -> JSON {"rows": [{datetime, sensor...}]} atau {"csv": "teks csv"}
                    balas {"count", "predictions": [{datetime, air_humidity}], "stats"}
Model dimuat sekali saat startup lalu dipakai ulang.
"""
import asyncio
import io
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import inference as inf

ART_PATH = os.environ.get("MODEL_PATH", str(Path(__file__).resolve().parent / "artifact_model.pkl"))
MAX_ROWS = 2000

ART = None
BOOSTERS = None


def load_artifact():
    global ART, BOOSTERS
    if ART is None:
        with open(ART_PATH, "rb") as f:
            ART = pickle.load(f)
        BOOSTERS = inf.load_boosters(ART)
    return ART, BOOSTERS


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifact()
    yield


app = FastAPI(title="Prediksi Kelembapan Udara", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    ok = ART is not None
    return {"status": "ok" if ok else "loading"}


def error(msg, code=400):
    return JSONResponse(status_code=code, content={"error": msg})


def parse_csv_text(text):
    return pd.read_csv(io.StringIO(text), parse_dates=["datetime"])


@app.post("/predict")
async def predict(req: Request):
    try:
        body = await req.json()
    except Exception:
        return error("Isi permintaan tidak valid.")
    try:
        if isinstance(body, dict) and isinstance(body.get("rows"), list):
            rows = body["rows"]
            if not rows:
                return error("Data kosong. Periksa kembali file atau form Anda.")
            if len(rows) > MAX_ROWS:
                return error("Data terlalu besar. Gunakan file yang lebih kecil.")
            df = pd.DataFrame(rows)
        elif isinstance(body, dict) and isinstance(body.get("csv"), str):
            if len(body["csv"]) > 5 * 1024 * 1024:
                return error("Ukuran file terlalu besar.")
            df = parse_csv_text(body["csv"])
        else:
            return error("Isi permintaan tidak valid.")
        if "datetime" not in df.columns:
            return error("Kolom waktu tidak ditemukan. Gunakan file contoh sebagai acuan.")
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        if df["datetime"].isna().any():
            return error("Format waktu belum sesuai. Gunakan format YYYY-MM-DD HH:MM:SS.")
        for c in inf.REQUIRED_SENSOR_COLS:
            if c not in df.columns:
                df[c] = float("nan")
        for c in inf.REQUIRED_SENSOR_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        art, boosters = load_artifact()
        preds, stats = await asyncio.to_thread(inf.predict_df, df, art, boosters)
        return {"count": len(preds), "predictions": preds, "stats": stats}
    except Exception:
        return error("Terjadi kendala saat memproses. Periksa data Anda lalu coba lagi.", 500)
