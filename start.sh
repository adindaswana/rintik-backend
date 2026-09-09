#!/bin/sh
# Unduh artifact bila belum ada (Render Free tanpa disk persisten),
# lalu jalankan API. MODEL_URL diisi di dashboard Render.
set -e
if [ ! -f /app/artifact_model.pkl ] && [ -n "$MODEL_URL" ]; then
  echo "Mengunduh model..."
  python -c "import urllib.request,os; urllib.request.urlretrieve(os.environ['MODEL_URL'], '/app/artifact_model.pkl')"
  ls -la /app/artifact_model.pkl
fi
if [ ! -f /app/artifact_model.pkl ]; then
  echo "artifact_model.pkl tidak ditemukan dan MODEL_URL kosong."
  exit 1
fi
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
