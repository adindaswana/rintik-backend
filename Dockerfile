FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MODEL_PATH=/app/artifact_model.pkl

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py inference.py start.sh ./
RUN chmod +x start.sh

EXPOSE 8000
CMD ["./start.sh"]
