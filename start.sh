#!/bin/sh
# Start only after the protected artifacts pass runtime verification.
# Jika RINTIK_ASSET_REPO diisi (mis. di Hugging Face Spaces), checkpoint dan data resmi
# diunduh dulu dari repo privat; hash SHA256 tetap diverifikasi sebelum API melayani.
# No substitute model is permitted.
set -e
if [ -n "$RINTIK_ASSET_REPO" ]; then
  python scripts/fetch_private_assets.py
fi
python -c "import os,inference as i; a,h=i.load_frozen_artifact(os.environ['RINTIK_MODEL_PATH']); i.load_state_sidecar(os.environ['RINTIK_STATE_PATH'],h); print('verified frozen model',h)"
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
