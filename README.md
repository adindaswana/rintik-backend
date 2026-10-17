# RINTIK backend — frozen inference, daily orchestration, and operator data

FastAPI backend for the official RINTIK checkpoint, chronological NSC test replay, and explicitly synthetic monitoring/recovery scenario. It does not download or train a model, and it never falls back to generated predictions.

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
python scripts\verify_milestone3.py
python scripts\preflight_demo.py
```

`preflight_demo.py` is the M5 fail-closed venue check. It verifies exact pinned
dependency versions, model/state/source hashes, all 16 daily fixture hashes,
synthetic provenance, and a disposable clean SQLite schema without inference.
Pass `--database <path> --api-url http://127.0.0.1:8140` to additionally check
an active demo database and the exact live readiness response. See
[`docs/MILESTONE_5_PREFLIGHT.md`](docs/MILESTONE_5_PREFLIGHT.md).

`verify_milestone2.py` performs the complete 16-date replay, restarts the
service mid-replay, checks prefix causality and P1→P2→P3 window parity, and
verifies old-day/final snapshot immutability. It uses a temporary SQLite file
that is deleted after the run.

The executed milestone decisions and evidence are recorded in
[`docs/MILESTONE_2_QA.md`](docs/MILESTONE_2_QA.md) and
[`docs/MILESTONE_3_QA.md`](docs/MILESTONE_3_QA.md). M4 operator API evidence is
in [`docs/MILESTONE_4_QA.md`](docs/MILESTONE_4_QA.md).

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

The read-only operator surface also provides:

- `GET /v1/features` for the grouped, unit-bearing feature catalog;
- `GET /v1/days/{date}/series` for canonical stored input values, with missing
  values serialized as JSON `null`;
- `GET /v1/days/{date}/export.csv` for current-day model stages, hashes,
  window lineage, synthetic sensor provenance, incident state, and recovery
  source in one CSV. The export is spreadsheet-formula-safe and carries
  `X-Content-Type-Options: nosniff`.

For the local public-page example button,
`GET /v1/demo/official-input/next` returns only the next pinned official input
CSV plus source/day hashes and `prediction_produced=false`. The route rechecks
the official source and daily manifest on every request. It does not infer,
bypass chronological ingest, or mutate state. The exact 734-row compatibility
route remains `POST /predict`; it is hash-gated, stateless, and returns frozen
model/state/pipeline provenance.

`POST /v1/demo/reset` accepts only the exact body
`{"confirmation":"RESET_DEMO_STATE"}`. It transactionally clears the local
demo replay and monitoring tables, then restores the initial simulated audit.
The response reports removed row counts and unchanged frozen model/state
hashes. It never deletes or edits the checkpoint, sidecar, or official
fixtures.

Repeat in chronological order. The first/last dates contain 18/44 rows; the 14 dates between contain 48 rows each.
Retries of an already stored identical payload are idempotent. A different
payload for the same date, or reuse of an `Idempotency-Key` for another date,
is rejected with HTTP 409.

After a daily run, the separate `/v1/demo` namespace can replay the pinned
synthetic RH scenario:

1. `POST /v1/demo/days/{date}/monitor`;
2. `GET /v1/demo/days/{date}/monitoring`;
3. `GET /v1/demo/incidents`;
4. `POST /v1/demo/audits` with an explicit simulated scenario;
5. `POST /v1/demo/incidents/{incident_id}/confirm` with an idempotency key.

Every monitoring response declares `mode=synthetic_demo`, `simulation=true`,
`model_input=false`, and `accuracy_ground_truth=false`. Recovery values are
stored separately as `source=model_estimate`; raw simulated sensor values are
not overwritten. Thresholds and Model Health are illustrative operational
heuristics, not calibrated accuracy evidence.

## Isolated operational simulation (14-25 October 2026)

The `/v1/operational` namespace supports the product's manual-reading workflow
without weakening the hash-gated official replay. It uses a deterministic,
documented synthetic exogenous fixture with 48 half-hour slots per date. The
fixture contains no physical-humidity input and is labeled as simulation, not
October 2026 field data or accuracy evidence.

Typical flow:

1. `GET /v1/operational/days` and select `next_expected_date`;
2. `POST /v1/operational/days/{date}/readings` for each manual slot;
3. explicitly call `POST /v1/operational/days/{date}/fill` to fill remaining
   slots from the synthetic fixture while preserving manual rows;
4. call `POST /v1/operational/days/{date}/run` only after all 48 slots exist;
5. read `/result` and `/series`, or reset only this namespace with
   `POST /v1/operational/reset` and the exact confirmation token.

Manual duplicates are rejected unless the explicit `PUT` edit route is used.
Runs remain chronological and archive `pred_internal` separately. The same
frozen runtime and Post 1-3 implementation are reused, while operational state,
provenance, published rows, and simulated humidity observations live in
separate tables from the official 2023 demo. A missing-humidity scenario keeps
the raw simulated sensor value null and stores the P3 substitute separately as
`estimated` recovery.

Regenerate and verify the fixture deterministically with:

```powershell
python scripts\build_operational_fixture.py
```

The committed CSV SHA256 is
`d311ff980a0043b14311b6d7e20fc04ca69ef09c34127a6594bee3fa48953da6`.

## Environment overrides

- `RINTIK_MODEL_PATH`
- `RINTIK_STATE_PATH`
- `RINTIK_TRAIN_PATH`
- `RINTIK_TEST_PATH`
- `RINTIK_DATABASE_PATH`
- `RINTIK_SYNTHETIC_FIXTURE_PATH`
- `RINTIK_SYNTHETIC_PROVENANCE_PATH`
- `RINTIK_OPERATIONAL_FIXTURE_PATH`
- `RINTIK_OPERATIONAL_PROVENANCE_PATH`
- `RINTIK_CORS_ORIGINS`

## Pemeriksaan Sensor (`POST /monitor`)

Endpoint stateless dari fitur Periksa Sensor (aturan `demo-v1` di `sensor_check.py`, ambang
`illustrative_unvalidated`). Masukan sama seperti `/predict` ditambah kolom `air_humidity`
(bacaan sensor unggahan, dibuang sebelum inferensi). Estimasi berasal dari runtime beku yang
sama; unggahan sebelum konteks beku ditolak dengan `MONITOR_INPUT_OUT_OF_RANGE`, dan bila model
tidak tersedia respons 503 tanpa fallback. Hasil tidak diarsipkan dan bacaan sensor unggahan bukan
ground truth. Ini terpisah dari monitoring simulasi berarsip `/v1/demo/*` (sinyal konflik pada
fixture sintetis); keduanya sama-sama ilustratif dan tidak boleh dibaca sebagai akurasi.

## Deployment status

Remote deployment remains intentionally disabled through Milestone 5. `render.yaml` declares no services. The Docker image requires the checkpoint plus official train/test files to be mounted read-only at `/official` and a writable `/runtime`; it does not download remote artifacts. No push or deployment has been performed.
