#!/bin/sh
# Start only after the protected local artifacts pass runtime verification.
# No network download and no substitute model are permitted.
set -e
python -c "import os,inference as i; a,h=i.load_frozen_artifact(os.environ['RINTIK_MODEL_PATH']); i.load_state_sidecar(os.environ['RINTIK_STATE_PATH'],h); print('verified frozen model',h)"
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
