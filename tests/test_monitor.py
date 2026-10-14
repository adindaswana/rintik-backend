"""Uji /monitor, /predict, dan perbaikan konteks 29 Agu.

Jalankan (dari folder backend):
    MODEL_PATH=/path/artifact_model.pkl DATA_DIR=/path/dataset python tests/test_monitor.py
atau dengan pytest bila terpasang. Uji aturan (bagian A) tidak butuh artefak.
Uji integrasi (bagian B) dilewati bila artefak / train.csv / test.csv tidak ada.
DATA_DIR berisi train.csv dan test.csv kompetisi.
"""
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import app as appmod  # noqa: E402

DATA_DIR = Path(os.environ.get("DATA_DIR", "/Users/fransiskaadinda/Downloads/LOMBA/enesce/laporan/dataset"))
HAVE_ART = Path(appmod.ART_PATH).is_file()
HAVE_DATA = (DATA_DIR / "train.csv").is_file() and (DATA_DIR / "test.csv").is_file()
CENTER = appmod.RULES["center"]["p3"]


# ---------------------------------------------------------------- A. aturan
def synth(days=6, shift_from=None, shift=-20.0, flat_day=None, stuck_day=None, jump_day=None,
          short_day=None, seed=0):
    """Tebakan sintetis berayun ~30 poin/hari; sensor = tebakan + center + derau kecil."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2023-08-01", periods=48 * days, freq="30min")
    slot = np.arange(len(t)) % 48
    model = 70 + 15 * np.cos(2 * np.pi * slot / 48)
    sensor = model + CENTER + rng.normal(0, 1.5, len(t))
    day = np.arange(len(t)) // 48
    if shift_from is not None:
        sensor[day >= shift_from] += shift
    if flat_day is not None:
        sensor[day == flat_day] = sensor[day == flat_day].mean()
    if stuck_day is not None:
        idx = np.flatnonzero(day == stuck_day)[:14]
        sensor[idx] = 99.0
    if jump_day is not None:
        sensor[np.flatnonzero(day == jump_day)[20]] += 45
    if short_day is not None:
        sensor[np.flatnonzero(day == short_day)[:10]] = np.nan
    return t, np.clip(sensor, 0, 100), model


def statuses(days):
    return [d["status"] for d in days]


def test_rules_sehat():
    t, s, m = synth()
    days = appmod.evaluate_days(t, s, m, CENTER)
    assert statuses(days) == ["sehat"] * 6, days
    assert days[0]["diff3"] is None and days[1]["diff3"] is None and days[2]["diff3"] is not None


def test_rules_shift_minus20():
    t, s, m = synth(days=8, shift_from=4)
    st = statuses(appmod.evaluate_days(t, s, m, CENTER))
    assert st[:4] == ["sehat"] * 4 and all(x == "dugaan_malfungsi" for x in st[4:]), st


def test_rules_shift_minus12_two_days():
    t, s, m = synth(days=6, shift_from=3, shift=-12.5)
    days = appmod.evaluate_days(t, s, m, CENTER)
    st = statuses(days)
    assert st[3] == "waspada" and st[4] == "dugaan_malfungsi", st
    assert any("dua hari berturut-turut" in r for r in days[4]["reasons"])


def test_rules_flat():
    t, s, m = synth(flat_day=2)
    days = appmod.evaluate_days(t, s, m, CENTER)
    assert days[2]["status"] == "dugaan_malfungsi" and any("sensor datar" in r for r in days[2]["reasons"])
    assert days[2]["corr"] is None


def test_rules_stuck_and_jump():
    t, s, m = synth(stuck_day=1, jump_day=3)
    days = appmod.evaluate_days(t, s, m, CENTER)
    assert days[1]["status"] == "dugaan_malfungsi" and any("beruntun" in r for r in days[1]["reasons"])
    assert days[3]["status"] == "dugaan_malfungsi" and any("loncat" in r for r in days[3]["reasons"])


def test_rules_data_kurang():
    t, s, m = synth(short_day=0)
    days = appmod.evaluate_days(t, s, m, CENTER)
    assert days[0]["status"] == "data_kurang" and days[0]["n"] == 38
    assert days[2]["diff3"] is None  # jendela 3 hari memuat hari tidak lengkap
    assert days[1]["status"] == "sehat"


# ----------------------------------------------------------- B. integrasi
def _client():
    from fastapi.testclient import TestClient
    return TestClient(appmod.app)


def _rows(df):
    """Seperti frontend: semua nilai teks, sel kosong jadi ''."""
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return out.astype(object).where(out.notna(), "").astype(str).to_dict(orient="records")


def _last15():
    tr = pd.read_csv(DATA_DIR / "train.csv")
    days = sorted(tr["datetime"].str[:10].unique())[-15:]
    return tr[tr["datetime"].str[:10].isin(days)].reset_index(drop=True), days


def _post(c, path, payload):
    t0 = time.time()
    r = c.post(path, json=payload)
    return r, time.time() - t0


def integration_ready():
    return HAVE_ART and HAVE_DATA


def test_predict_test_csv_matches_submission():
    if not integration_ready():
        return "lewati"
    with _client() as c:
        rows = _rows(pd.read_csv(DATA_DIR / "test.csv"))
        r, dt = _post(c, "/predict", {"rows": rows})
        j = r.json()
        assert r.status_code == 200 and j["count"] == 734
        assert set(j["predictions"][0]) == {"datetime", "air_humidity", "inti"}
        assert j["predictions"][0]["datetime"] == "2023-08-29 15:00:00"
        # submission: 60.1435 (Linux); beda platform float <= 0.003
        assert abs(j["predictions"][0]["air_humidity"] - 60.1435) <= 0.003
        return "count=734, baris1=%.4f, %.2fs" % (j["predictions"][0]["air_humidity"], dt)


def test_context_bug_fixed():
    if not integration_ready():
        return "lewati"
    import inference as inf
    art, bo = appmod.load_artifact()
    tr = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["datetime"])
    a = tr[(tr.datetime >= "2023-08-22") & (tr.datetime < "2023-08-29")].reset_index(drop=True)
    b = tr[(tr.datetime >= "2023-08-22") & (tr.datetime < "2023-08-30")].reset_index(drop=True)
    da = inf.predict_detail(a.drop(columns=["air_humidity"]), art, bo)
    db = inf.predict_detail(b.drop(columns=["air_humidity"]), art, bo)
    k = len(a)
    rm = float(np.sqrt(np.mean((da["p3"] - a["air_humidity"].to_numpy()) ** 2)))
    assert np.array_equal(da["inti"], db["inti"][:k])  # inti per hari & kausal -> identik
    assert rm < 5.0, rm  # sebelum perbaikan 26,44
    return "RMSE 22-28 Agu = %.3f; inti identik dengan potongan unggahan 22-29" % rm


def test_monitor_sehat_rusak():
    if not integration_ready():
        return "lewati"
    sehat, days = _last15()
    rusak = sehat.copy()
    d = rusak["datetime"].str[:10]
    m = d.isin(days[7:]) & rusak["air_humidity"].notna()
    rusak.loc[m, "air_humidity"] = (rusak.loc[m, "air_humidity"] - 20).clip(0, 100)
    flat = d == days[11]
    rusak.loc[flat, "air_humidity"] = round(rusak.loc[flat, "air_humidity"].mean(), 2)
    out = []
    with _client() as c:
        for name, df in (("sehat", sehat), ("rusak", rusak)):
            for comp in ("p3", "inti"):
                r, dt = _post(c, "/monitor", {"rows": _rows(df), "comparator": comp})
                j = r.json()
                assert r.status_code == 200, j
                assert j["count"] == len(df) and len(j["days"]) == 15 and j["rule_version"] == "demo-v1"
                st = [x["status"] for x in j["days"]]
                assert st[-1] == "data_kurang"  # 29 Agu hanya 30 baris
                if name == "rusak":
                    assert all(x == "dugaan_malfungsi" for x in st[7:14]), st
                    assert any("sensor datar" in x for x in j["days"][11]["reasons"])
                out.append("%s/%s %.2fs %s" % (name, comp, dt, j["summary"]))
    return "; ".join(out)


def test_monitor_validation():
    if not integration_ready():
        return "lewati"
    sehat, _ = _last15()
    rows = _rows(sehat.head(60))
    with _client() as c:
        r = c.post("/monitor", json={"rows": [{k: v for k, v in x.items() if k != "air_humidity"} for x in rows]})
        assert r.status_code == 400 and "air_humidity" in r.json()["error"]
        r = c.post("/monitor", json={"rows": rows, "comparator": "x"})
        assert r.status_code == 400
        bad = [dict(x) for x in rows]
        bad[3]["air_humidity"] = "abc"
        r = c.post("/monitor", json={"rows": bad})
        assert r.status_code == 400 and "baris 4" in r.json()["error"]
        r = c.post("/monitor", json={"rows": rows * 40})
        assert r.status_code == 400  # > MAX_ROWS
        r = c.post("/monitor", content=b"{" + b" " * (6 * 1024 * 1024) + b"}",
                   headers={"content-type": "application/json"})
        assert r.status_code == 400
        # air_humidity tidak memengaruhi tebakan
        a = c.post("/monitor", json={"rows": rows}).json()
        rows2 = [dict(x, air_humidity="5") for x in rows]
        b = c.post("/monitor", json={"rows": rows2}).json()
        assert [p["p3"] for p in a["predictions"]] == [p["p3"] for p in b["predictions"]]
    return "ok"


TESTS = [test_rules_sehat, test_rules_shift_minus20, test_rules_shift_minus12_two_days, test_rules_flat,
         test_rules_stuck_and_jump, test_rules_data_kurang, test_predict_test_csv_matches_submission,
         test_context_bug_fixed, test_monitor_sehat_rusak, test_monitor_validation]

if __name__ == "__main__":
    fail = 0
    for fn in TESTS:
        try:
            res = fn()
            print("LULUS", fn.__name__, res or "")
        except AssertionError as e:
            fail += 1
            print("GAGAL", fn.__name__, str(e)[:500])
    sys.exit(1 if fail else 0)
