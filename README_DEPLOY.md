# Deploy Backend ke Render Free

## 1. Siapkan file model (sekali saja)

File `artifact_model.pkl` (±121 MB) tidak masuk git (limit GitHub 100 MB/file).
Unggah ke HuggingFace Hub sebagai repo model:

1. Login huggingface.co, buat New Model (public), mis. `username/hfm-humidity-model`.
2. Upload file `artifact_model.pkl` lewat web (Add file > Upload).
3. Salin URL langsung file: `https://huggingface.co/username/hfm-humidity-model/resolve/main/artifact_model.pkl`.
   Ini menjadi `MODEL_URL`.

Alternatif: GitHub Release pada repo backend (lampirkan file sebagai aset rilis),
lalu pakai URL aset sebagai `MODEL_URL`.

## 2. Buat repo GitHub backend

Push folder ini (tanpa file `.pkl`, sudah diabaikan `.gitignore`):

```
git init && git add . && git commit -m "Backend prediksi kelembapan udara"
```

## 3. Buat Web Service di Render

1. render.com > New > Web Service > hubungkan repo backend.
2. Runtime: Docker. Plan: Free. Health check path: `/health`.
3. Environment > tambah `MODEL_URL` = URL dari langkah 1.
4. Deploy. Boot pertama lambat (unduh model ±121 MB + muat ke memori),
   request berikutnya cepat sampai service tidur (15 menit idle).

## 4. Sambungkan Vercel

Di dashboard Vercel (repo frontend), isi Environment Variables:

- `MODEL_MODE=remote`
- `MODEL_API_URL=https://NAMA-SERVICE.onrender.com` (tanpa garis miring akhir)

Redeploy frontend. Uji lewat halaman tersembunyi `/status.html`:
tunggu status terjangkau, lalu uji prediksi dari `/predict.html`.

## 5. Sebelum demo

Buka `/status.html`, klik Periksa Sekarang 1-2 menit sebelum demo.
Bila belum terjangkau, tunggu sekitar satu menit dan periksa lagi.

## 6. Endpoint

| Endpoint | Isi |
|---|---|
| `GET /health` | `{"status":"ok"}` bila model sudah dimuat |
| `POST /predict` | `{"rows":[...]}` atau `{"csv":"..."}` → `{"count","predictions":[{datetime, air_humidity, inti}],"stats"}`. `air_humidity` = P3 (final, sama seperti submission), `inti` = tebakan inti (`pred_internal`) sebelum post-processing. Field `inti` tambahan; frontend lama hanya membaca `datetime` dan `air_humidity`. |
| `POST /monitor` | Bandingkan bacaan sensor kelembapan dengan tebakan model, per hari. |
| `GET /docs` | OpenAPI / Swagger, termasuk skema `/predict` dan `/monitor`. |

Konteks 29 Agu: artefak menyimpan 30 baris train 29 Agu 00:00–14:30. Baris ini hanya
digabung bila unggahan punya baris pada 29 Agu (melengkapi agregat harian hari itu);
datetime yang sama di unggahan menang. Indeks baris unggahan diambil lewat penanda asal,
jadi unggahan bertanggal sebelum 29 Agu tidak lagi bergeser (dulu RMSE 22–28 Agu 26,44,
sekarang 3,31). Prediksi `test.csv` tidak berubah.

### `POST /monitor`

Masukan sama dengan `/predict` (maks 2000 baris, badan maks 5 MB) **ditambah** kolom
`air_humidity` = bacaan sensor kelembapan fisik (boleh kosong per baris). Kolom ini
dibuang sebelum prediksi, jadi tebakan model tidak pernah melihatnya.

```
{"rows": [{"datetime": "2023-08-15 00:00:00", "air_temperature": 18.2, ..., "air_humidity": 88.4}],
 "comparator": "p3",      // atau "inti"; default "p3"
 "center": -2.36}         // opsional; default p3 -2.36, inti -2.80
```

Per hari kalender: `n` (jumlah bacaan sensor terisi), `mean_diff` = rerata
(sensor − tebakan), `corr` (Pearson sensor vs tebakan; `null` bila sensor konstan),
`sensor_range`, `model_range`, `diff3` = rerata `mean_diff` 3 hari terakhir (mulai hari
ke-3 dan hanya bila ketiga hari lengkap). `dev = |mean_diff − center|`.

Status (ambang di `RULES` pada `app.py`, `rule_version` `demo-v1`, **belum dikalibrasi**,
dari simulasi 23 hari validasi):

- `data_kurang`: n < 48.
- `dugaan_malfungsi`: dev > 16,3; atau dev > 12 dua hari berturut-turut; atau
  |diff3 − center| > 16,3; atau korelasi < 0,5 (hanya bila tebakan berayun ≥ 5); atau sensor
  datar (rentang ≤ 1) padahal tebakan berayun ≥ 20; atau bacaan ≥ 98% selama 12 slot
  beruntun (6 jam); atau loncat > 30 poin antar slot 30 menit.
- `waspada`: dev > 10.
- `sehat`: selain itu.

Respons:

```
{"count", "comparator", "rule_version", "center",
 "predictions": [{"datetime", "sensor", "p3", "inti"}],
 "days": [{"date", "n", "mean_diff", "corr", "sensor_range", "model_range", "diff3", "status", "reasons": [...]}],
 "summary": {"sehat", "waspada", "dugaan_malfungsi", "data_kurang"},
 "rules": {...}}
```

Contoh curl (file `rows.json` berisi `{"rows":[...],"comparator":"p3"}`):

```
curl -s -X POST https://rintik-backend.onrender.com/monitor \
  -H 'Content-Type: application/json' --data @rows.json
```

Contoh satu hari dari respons (sensor digeser −20):

```
{"date": "2023-08-22", "n": 48, "mean_diff": -20.53, "corr": 0.996, "sensor_range": 44.39,
 "model_range": 41.46, "diff3": -6.86, "status": "dugaan_malfungsi",
 "reasons": ["selisih rata-rata −20,5 poin, menyimpang 18,2 dari normal −2,36 (lebih dari 16,3)"]}
```

Galat validasi (HTTP 400, `{"error": "..."}`): kolom `air_humidity` tidak ada, nilainya
bukan angka, semua kosong, `comparator` bukan `p3`/`inti`, lebih dari 2000 baris, badan > 5 MB.

### Uji

```
MODEL_PATH=/path/artifact_model.pkl DATA_DIR=/path/dataset python tests/test_monitor.py
```

Uji aturan memakai tebakan sintetis (tanpa artefak). Uji integrasi (regresi `test.csv`,
perbaikan konteks, `/monitor` 15 hari sehat/rusak, validasi) dilewati bila artefak atau
`train.csv`/`test.csv` tidak ada. Butuh `httpx` untuk `TestClient` (hanya untuk uji, tidak
perlu di image).
