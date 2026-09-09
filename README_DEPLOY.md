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
