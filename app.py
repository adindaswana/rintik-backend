"""Backend prediksi kelembapan udara (FastAPI).

Endpoint:
  GET  /health   -> {"status": "ok"}
  POST /predict  -> JSON {"rows": [{datetime, sensor...}]} atau {"csv": "teks csv"}
                    balas {"count", "predictions": [{datetime, air_humidity, inti}], "stats"}
                    (air_humidity = P3/final, sama seperti submission; inti = tebakan inti)
  POST /monitor  -> JSON {"rows": [{datetime, sensor..., air_humidity}], "comparator": "p3"|"inti"}
                    bandingkan bacaan sensor kelembapan dengan tebakan model per hari,
                    balas status sehat / waspada / dugaan_malfungsi / data_kurang + alasan.
Model dimuat sekali saat startup lalu dipakai ulang.
"""
import asyncio
import io
import json
import math
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import inference as inf

ART_PATH = os.environ.get("MODEL_PATH", str(Path(__file__).resolve().parent / "artifact_model.pkl"))
MAX_ROWS = 2000
MAX_BODY_BYTES = 5 * 1024 * 1024

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


def fmt_dt(values):
    return pd.Series(values).dt.strftime("%Y-%m-%d %H:%M:%S").tolist()


def rows_to_frame(body, cap_csv_rows=True):
    """Validasi bersama /predict dan /monitor. Return (df, pesan_error).

    Kolom sensor wajib yang tidak ada diisi NaN, semua dijadikan numerik.
    Kolom lain (mis. air_humidity untuk /monitor) dibiarkan apa adanya.
    """
    if isinstance(body, dict) and isinstance(body.get("rows"), list):
        rows = body["rows"]
        if not rows:
            return None, "Data kosong. Periksa kembali file atau form Anda."
        if len(rows) > MAX_ROWS:
            return None, "Data terlalu besar. Gunakan file yang lebih kecil."
        if not all(isinstance(r, dict) for r in rows):
            return None, "Isi permintaan tidak valid."
        df = pd.DataFrame(rows)
    elif isinstance(body, dict) and isinstance(body.get("csv"), str):
        if len(body["csv"]) > MAX_BODY_BYTES:
            return None, "Ukuran file terlalu besar."
        df = parse_csv_text(body["csv"])
        if df.empty:
            return None, "Data kosong. Periksa kembali file atau form Anda."
        if cap_csv_rows and len(df) > MAX_ROWS:
            return None, "Data terlalu besar. Gunakan file yang lebih kecil."
    else:
        return None, "Isi permintaan tidak valid."
    if "datetime" not in df.columns:
        return None, "Kolom waktu tidak ditemukan. Gunakan file contoh sebagai acuan."
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if df["datetime"].isna().any():
        return None, "Format waktu belum sesuai. Gunakan format YYYY-MM-DD HH:MM:SS."
    for c in inf.REQUIRED_SENSOR_COLS:
        if c not in df.columns:
            df[c] = float("nan")
    for c in inf.REQUIRED_SENSOR_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, None


@app.post(
    "/predict",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "rows": {
                                "type": "array",
                                "items": {"type": "object"},
                                "description": "Daftar baris dengan kolom datetime YYYY-MM-DD HH:MM:SS + sensor",
                            },
                            "csv": {
                                "type": "string",
                                "description": "Alternatif: teks CSV dengan kolom datetime",
                            },
                        },
                        "example": {
                            "rows": [
                                {
                                    "datetime": "2023-08-29 15:00:00",
                                    "air_temperature": 26.8,
                                    "co2": 468.9,
                                    "pressure": 1009.6,
                                }
                            ]
                        },
                    }
                }
            },
            "required": True,
        }
    },
)
async def predict(req: Request):
    try:
        body = await req.json()
    except Exception:
        return error("Isi permintaan tidak valid.")
    try:
        df, err = rows_to_frame(body, cap_csv_rows=False)
        if err:
            return error(err)
        art, boosters = load_artifact()
        det = await asyncio.to_thread(inf.predict_detail, df, art, boosters)
        p3, inti = det["p3"], det["inti"]
        preds = [{"datetime": d, "air_humidity": float(v), "inti": float(w)} for d, v, w in zip(
            fmt_dt(det["datetime"]), np.round(p3, 4), np.round(inti, 4))]
        stats = {"mean": round(float(p3.mean()), 2), "min": round(float(p3.min()), 2), "max": round(float(p3.max()), 2)}
        return {"count": len(preds), "predictions": preds, "stats": stats}
    except Exception:
        return error("Terjadi kendala saat memproses. Periksa data Anda lalu coba lagi.", 500)


# ---------------------------------------------------------------------------
# /monitor: bandingkan bacaan sensor kelembapan dengan tebakan model per hari.
# ---------------------------------------------------------------------------
SENSOR_COL = "air_humidity"

# Semua ambang di satu tempat. BELUM DIKALIBRASI: diambil dari simulasi 23 hari
# validasi (6-28 Agu 2023, 21 jendela 3 hari; deck appendix A8/A9 dan slide
# prototipe "Pengaman"). Wajib dikalibrasi ulang dengan ~30 hari data lapangan.
# selisih = sensor - tebakan (poin %RH); dev = |selisih rata-rata harian - center|.
RULES = {
    "version": "demo-v1",
    "center": {"p3": -2.36, "inti": -2.80},  # selisih "normal" saat sensor sehat
    "min_rows": 48,               # < 48 bacaan sensor sehari -> data_kurang
    "level_after_days": 3,        # aturan diff3 baru aktif mulai hari ke-3 unggahan
    "waspada_dev": 10.0,          # dev > 10 -> waspada
    "dev_two_days": 12.0,         # dev > 12 dua hari berturut-turut -> dugaan
    "dev_one_day": 16.3,          # dev > 16,3 satu hari (~3 simpangan baku) -> dugaan
    "diff3_dev": 16.3,            # |rerata selisih 3 hari - center| > 16,3 -> dugaan
    "corr_min": 0.5,              # cek bentuk: korelasi harian < 0,5 -> dugaan
    "corr_min_model_range": 5.0,  # ...hanya bila tebakan berayun >= 5 poin (hindari bising)
    "flat_sensor_range": 1.0,     # sensor datar: rentang <= 1 ...
    "flat_model_range": 20.0,     # ...padahal tebakan berayun >= 20
    "stuck_high_value": 98.0,     # aturan biasa: bacaan >= 98% ...
    "stuck_high_slots": 12,       # ...12 slot 30 menit beruntun (6 jam)
    "jump_points": 30.0,          # aturan biasa: loncat > 30 poin antar slot 30 menit
    "slot_minutes": 30,
}
STATUSES = ("sehat", "waspada", "dugaan_malfungsi", "data_kurang")


def _id(x, nd=1):
    """Angka gaya Indonesia: koma desimal, tanda minus asli."""
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def _rnd(x, nd):
    return None if x is None or not math.isfinite(x) else round(float(x), nd)


def _runs(ts, vals, slot, cond):
    """Panjang run terpanjang (jumlah slot) dengan cond benar dan jarak tepat satu slot."""
    best = cur = 0
    prev_t = None
    for t, v in zip(ts, vals):
        if cond(v):
            cur = cur + 1 if (cur and prev_t is not None and t - prev_t == slot) else 1
            best = max(best, cur)
        else:
            cur = 0
        prev_t = t
    return best


def evaluate_days(dts, sensor, model, center, rules=RULES):
    """Hitung statistik dan status per hari kalender.

    dts: datetime64 terurut; sensor: bacaan fisik (boleh NaN); model: tebakan pembanding.
    """
    R = rules
    slot = pd.Timedelta(minutes=R["slot_minutes"])
    frame = pd.DataFrame({"datetime": pd.to_datetime(dts), "sensor": np.asarray(sensor, float),
                          "model": np.asarray(model, float)})
    frame["date"] = frame["datetime"].dt.floor("D")
    days = []
    for i, (date, g) in enumerate(frame.groupby("date", sort=True)):
        g = g.sort_values("datetime", kind="mergesort")
        ok = g["sensor"].notna() & g["model"].notna()
        s = g.loc[ok, "sensor"].to_numpy()
        m = g.loc[ok, "model"].to_numpy()
        t = g.loc[ok, "datetime"].tolist()
        n = int(ok.sum())
        mean_diff = float(np.mean(s - m)) if n else None
        s_rng = float(np.ptp(s)) if n else None
        m_rng = float(np.ptp(m)) if n else None
        corr = None
        if n >= 3 and np.std(s) > 0 and np.std(m) > 0:
            corr = float(np.corrcoef(s, m)[0, 1])
        stuck = _runs(t, s, slot, lambda v: v >= R["stuck_high_value"]) if n else 0
        jump = 0.0
        for k in range(1, n):
            if t[k] - t[k - 1] == slot:
                jump = max(jump, abs(float(s[k] - s[k - 1])))
        days.append({"date": date, "i": i, "n": n, "n_rows": int(len(g)), "mean_diff": mean_diff,
                     "corr": corr, "sensor_range": s_rng, "model_range": m_rng,
                     "stuck": stuck, "jump": jump})

    out = []
    full = lambda d: d is not None and d["n"] >= R["min_rows"]  # noqa: E731
    for j, d in enumerate(days):
        prev = days[j - 1] if j >= 1 else None
        diff3 = None
        if j + 1 >= R["level_after_days"]:
            win = days[j + 1 - R["level_after_days"]: j + 1]
            if all(full(x) for x in win):
                diff3 = float(np.mean([x["mean_diff"] for x in win]))
        reasons = []
        if d["n"] < R["min_rows"]:
            status = "data_kurang"
            if d["n"] == 0:
                reasons.append(f"tidak ada bacaan sensor terisi (dari {d['n_rows']} baris)")
            else:
                reasons.append(f"hanya {d['n']} bacaan sensor (minimal {R['min_rows']}), status belum dinilai")
        else:
            md, dev = d["mean_diff"], abs(d["mean_diff"] - center)
            red = []
            if dev > R["dev_one_day"]:
                red.append(f"selisih rata-rata {_id(md)} poin, menyimpang {_id(dev)} dari normal "
                           f"{_id(center, 2)} (lebih dari {_id(R['dev_one_day'])})")
            elif dev > R["dev_two_days"] and full(prev) and abs(prev["mean_diff"] - center) > R["dev_two_days"]:
                red.append(f"selisih menyimpang lebih dari {_id(R['dev_two_days'], 0)} poin dua hari berturut-turut "
                           f"(hari ini {_id(dev)}, kemarin {_id(abs(prev['mean_diff'] - center))})")
            if diff3 is not None and abs(diff3 - center) > R["diff3_dev"]:
                red.append(f"rata-rata selisih 3 hari {_id(diff3)} poin, menyimpang {_id(abs(diff3 - center))} "
                           f"dari normal (lebih dari {_id(R['diff3_dev'])})")
            if (d["corr"] is not None and d["model_range"] >= R["corr_min_model_range"]
                    and d["corr"] < R["corr_min"]):
                red.append(f"pola naik-turun sensor tidak searah tebakan (korelasi {_id(d['corr'], 2)}, "
                           f"minimal {_id(R['corr_min'], 1)})")
            if d["sensor_range"] <= R["flat_sensor_range"] and d["model_range"] >= R["flat_model_range"]:
                red.append(f"sensor datar seharian (rentang {_id(d['sensor_range'])}) padahal tebakan "
                           f"berayun {_id(d['model_range'], 0)} poin")
            if d["stuck"] >= R["stuck_high_slots"]:
                red.append(f"bacaan ≥{_id(R['stuck_high_value'], 0)}% selama {_id(d['stuck'] * R['slot_minutes'] / 60)} "
                           f"jam beruntun (batas {_id(R['stuck_high_slots'] * R['slot_minutes'] / 60, 0)} jam)")
            if d["jump"] > R["jump_points"]:
                red.append(f"bacaan loncat {_id(d['jump'])} poin dalam {R['slot_minutes']} menit "
                           f"(batas {_id(R['jump_points'], 0)})")
            if red:
                status, reasons = "dugaan_malfungsi", red
            elif dev > R["waspada_dev"]:
                status = "waspada"
                reasons.append(f"selisih rata-rata {_id(md)} poin, menyimpang {_id(dev)} dari normal "
                               f"{_id(center, 2)} (batas waspada {_id(R['waspada_dev'], 0)})")
            else:
                status = "sehat"
                txt = f"selisih rata-rata {_id(md)} poin, dekat normal {_id(center, 2)}"
                if d["corr"] is not None:
                    txt += f"; pola searah tebakan (korelasi {_id(d['corr'], 2)})"
                reasons.append(txt)
            if j + 1 < R["level_after_days"]:
                reasons.append(f"cek selisih 3 hari baru aktif mulai hari ke-{R['level_after_days']}")
        out.append({
            "date": d["date"].strftime("%Y-%m-%d"), "n": d["n"],
            "mean_diff": _rnd(d["mean_diff"], 2), "corr": _rnd(d["corr"], 3),
            "sensor_range": _rnd(d["sensor_range"], 2), "model_range": _rnd(d["model_range"], 2),
            "diff3": _rnd(diff3, 2), "status": status, "reasons": reasons,
        })
    return out


def parse_sensor(df):
    """Ambil kolom air_humidity dari unggahan. Return (Series float, pesan_error)."""
    if SENSOR_COL not in df.columns:
        return None, "Kolom bacaan sensor kelembapan (air_humidity) wajib ada."
    raw = df[SENSOR_COL]
    blank = raw.isna() | raw.astype(str).str.strip().isin(["", "nan", "NaN", "None"])
    val = pd.to_numeric(raw.where(~blank), errors="coerce")
    bad = (~blank) & val.isna()
    if bad.any():
        return None, f"Bacaan sensor kelembapan pada baris {int(np.flatnonzero(bad.to_numpy())[0]) + 1} bukan angka."
    if val.isna().all():
        return None, "Semua bacaan sensor kelembapan kosong. Tidak ada yang bisa diperiksa."
    return val.astype(float), None


MONITOR_SCHEMA = {
    "type": "object",
    "required": ["rows"],
    "properties": {
        "rows": {
            "type": "array",
            "items": {"type": "object"},
            "description": "Baris seperti /predict (datetime + kolom sensor) PLUS air_humidity = bacaan "
                           "sensor kelembapan fisik. air_humidity tidak dipakai model, hanya pembanding. "
                           f"Maks {MAX_ROWS} baris, badan maks 5 MB.",
        },
        "csv": {"type": "string", "description": "Alternatif: teks CSV dengan kolom datetime dan air_humidity"},
        "comparator": {"type": "string", "enum": ["p3", "inti"], "default": "p3",
                       "description": "Tebakan pembanding: p3 (final, sama dengan /predict) atau inti"},
        "center": {"type": "number", "description": "Opsional. Selisih normal; default p3 -2.36, inti -2.80"},
    },
    "example": {
        "comparator": "p3",
        "rows": [{"datetime": "2023-08-15 00:00:00", "air_temperature": 18.2, "co2": 470.1,
                  "pressure": 1009.6, "air_humidity": 88.4}],
    },
}


@app.post(
    "/monitor",
    summary="Periksa bacaan sensor kelembapan terhadap tebakan model",
    description=(
        "Tebakan dibuat dari kolom sensor lain (air_humidity dibuang sebelum prediksi), lalu dibandingkan "
        "dengan air_humidity per hari kalender. selisih = sensor - tebakan. Status: sehat / waspada / "
        "dugaan_malfungsi / data_kurang, disertai alasan. Ambang rule_version demo-v1, belum dikalibrasi."
    ),
    openapi_extra={"requestBody": {"content": {"application/json": {"schema": MONITOR_SCHEMA}}, "required": True}},
)
async def monitor(req: Request):
    try:
        raw = await req.body()
        if len(raw) > MAX_BODY_BYTES:
            return error("Ukuran file terlalu besar.")
        body = json.loads(raw)
    except Exception:
        return error("Isi permintaan tidak valid.")
    try:
        comparator = body.get("comparator") if isinstance(body, dict) else None
        if comparator in (None, ""):
            comparator = "p3"
        if comparator not in RULES["center"]:
            return error("Pembanding harus p3 atau inti.")
        center = RULES["center"][comparator]
        if isinstance(body, dict) and body.get("center") is not None:
            try:
                center = float(body["center"])
                if not math.isfinite(center):
                    raise ValueError
            except (TypeError, ValueError):
                return error("Nilai center harus berupa angka.")
        df, err = rows_to_frame(body)
        if err:
            return error(err)
        sensor, err = parse_sensor(df)
        if err:
            return error(err)
        feats = df.drop(columns=[SENSOR_COL])  # model tidak boleh melihat bacaan sensor kelembapan
        art, boosters = load_artifact()
        det = await asyncio.to_thread(inf.predict_detail, feats, art, boosters)
        s = sensor.loc[det["index"]].to_numpy()
        model = det["p3"] if comparator == "p3" else det["inti"]
        days = evaluate_days(det["datetime"], s, model, center)
        summary = {k: 0 for k in STATUSES}
        for d in days:
            summary[d["status"]] += 1
        preds = [{"datetime": t, "sensor": _rnd(a, 4), "p3": round(float(b), 4), "inti": round(float(c), 4)}
                 for t, a, b, c in zip(fmt_dt(det["datetime"]), s, det["p3"], det["inti"])]
        return {"count": len(preds), "comparator": comparator, "rule_version": RULES["version"],
                "center": center, "predictions": preds, "days": days, "summary": summary,
                "rules": RULES}
    except Exception:
        return error("Terjadi kendala saat memproses. Periksa data Anda lalu coba lagi.", 500)
