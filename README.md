# RINTIK backend — frozen inference and daily orchestration

FastAPI backend for the official RINTIK checkpoint and chronological NSC test replay. This branch does not download a model, train a model, or fall back to generated predictions.

## Protected inputs

Default paths assume the repository remains at `FINAL/Prototipe/rintik-backend`:

- checkpoint: `FINAL/NSC26-DS011_HBD_File Data Science.pth`;
- official sources: `Prototipe/reference/official/train.csv` and `test.csv`;
- committed sidecar: `artifacts/inference_state.json`;
- local runtime database: `runtime/rintik.sqlite3` (ignored by Git).

The checkpoint must match SHA256 `adce8fda2f465679ea6e8fe31124822a45a555b9dc154e12076ef5b6140699b0` or the API stays unavailable.

## Reproduce and verify

From this directory with Python 3.12:

```powershell
python -m pip install -r requirements-dev.txt
python scripts\build_inference_state.py --check
python -m unittest discover -s tests -v
python scripts\verify_milestone2.py
```

`verify_milestone2.py` performs the complete 16-date replay, restarts the
service mid-replay, checks prefix causality and P1→P2→P3 window parity, and
verifies old-day/final snapshot immutability. It uses a temporary SQLite file
that is deleted after the run.

The executed M2 decisions and evidence are recorded in
[`docs/MILESTONE_2_QA.md`](docs/MILESTONE_2_QA.md).

To rebuild the deterministic sidecar from the pinned official sources:

```powershell
python scripts\build_inference_state.py
python scripts\build_inference_state.py --check
```

The builder validates model/train/test hashes before writing. It never edits those sources.

## Run locally

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Then inspect `GET http://127.0.0.1:8000/health`. A missing or mismatched artifact returns HTTP 503 and prediction routes remain fail-closed.

The stable daily contract is documented in `../contracts/API_CONTRACT.md`. Use:

1. `POST /v1/days/{date}/ingest` with one exact official test day;
2. `POST /v1/days/{date}/run`;
3. `GET /v1/days/{date}/result`.

Repeat in chronological order. The first/last dates contain 18/44 rows; the 14 dates between contain 48 rows each.
Retries of an already stored identical payload are idempotent. A different
payload for the same date, or reuse of an `Idempotency-Key` for another date,
is rejected with HTTP 409.

## Environment overrides

- `RINTIK_MODEL_PATH`
- `RINTIK_STATE_PATH`
- `RINTIK_TRAIN_PATH`
- `RINTIK_TEST_PATH`
- `RINTIK_DATABASE_PATH`
- `RINTIK_CORS_ORIGINS`

## Deployment status

Remote deployment is intentionally disabled through Milestone 2. `render.yaml` declares no services. The Docker image requires the checkpoint plus official train/test files to be mounted read-only at `/official` and a writable `/runtime`; it does not download remote artifacts. No push or deployment has been performed.
