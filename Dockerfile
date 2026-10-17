# RINTIK frozen backend, siap untuk Hugging Face Spaces (Docker SDK) maupun Docker biasa.
# Checkpoint dan data resmi TIDAK ikut image: diunduh saat start dari repo privat
# (RINTIK_ASSET_REPO + secret HF_TOKEN), lalu tetap diverifikasi SHA256 oleh inference.py.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HOME=/home/user \
    PORT=7860 \
    RINTIK_ASSET_DIR=/home/user/assets \
    RINTIK_MODEL_PATH="/home/user/assets/NSC26-DS011_HBD_File Data Science.pth" \
    RINTIK_STATE_PATH=/home/user/app/artifacts/inference_state.json \
    RINTIK_TRAIN_PATH=/home/user/assets/train.csv \
    RINTIK_TEST_PATH=/home/user/assets/test.csv \
    RINTIK_SYNTHETIC_FIXTURE_PATH=/home/user/assets/synthetic_physical_humidity_SENSOR_SIMULATION.csv \
    RINTIK_SYNTHETIC_PROVENANCE_PATH=/home/user/assets/SOURCE_PROVENANCE.json \
    RINTIK_DATABASE_PATH=/home/user/runtime/rintik.sqlite3 \
    RINTIK_LOW_MEMORY=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user

WORKDIR /home/user/app
COPY requirements.txt .
# Server memakai RINTIK_LOW_MEMORY=1 (checkpoint dibuka tanpa torch), jadi torch tidak dipasang.
RUN grep -v '^torch' requirements.txt > /tmp/requirements-server.txt \
    && pip install --no-cache-dir -r /tmp/requirements-server.txt huggingface_hub==0.26.2

COPY --chown=user . .
RUN chmod +x start.sh
USER user
RUN mkdir -p /home/user/assets /home/user/runtime

EXPOSE 7860
CMD ["./start.sh"]
