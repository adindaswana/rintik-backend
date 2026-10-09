FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    RINTIK_MODEL_PATH="/official/NSC26-DS011_HBD_File Data Science.pth" \
    RINTIK_STATE_PATH=/app/artifacts/inference_state.json \
    RINTIK_TRAIN_PATH=/official/train.csv \
    RINTIK_TEST_PATH=/official/test.csv \
    RINTIK_DATABASE_PATH=/runtime/rintik.sqlite3

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py daily_batch.py inference.py start.sh ./
COPY artifacts ./artifacts
RUN chmod +x start.sh

EXPOSE 8000
CMD ["./start.sh"]
