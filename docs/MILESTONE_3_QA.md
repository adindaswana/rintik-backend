# Milestone 3 — monitoring, recovery, and integration reliability

Date: 10 October 2026
Branch: `feature/nsc-final-daily-inference`
Starting commit: `600f621`
Scope: additive M3 backend/API integration; no model or formula changes

## Implemented boundary

M3 adds a separate, pinned synthetic-RH monitoring stream, persisted health,
audit, incident/action, and recovery records, plus `/v1/demo` routes. The real
model routes retain `source=real_original_pth` and `simulation=false`.

The synthetic fixture is SHA256-pinned to
`891070499559c984b0761ba04d5a705e7a52bd181fb58d033e18a519eccf63f3`.
Startup validates its schema, timestamps, 734-row chronology, scenario labels,
source label, provenance linkage to the official test hash, and declarations
`model_input=false` and `accuracy_ground_truth=false`.

## Reliability policy

- 4–5 September: suspected stuck sensor only; no automatic diagnosis.
- 6 September: suspected outage; physical RH remains null.
- 7 September onward: explicitly `simulated_repaired`.
- Recovery requires healthy operational gates plus an explicit simulated
  technician action and demo override while the simulated audit passes.
- A failed, expired, or not-yet-verified audit restricts recovery independently
  of a high health score.
- Recovery is persisted separately with `source=model_estimate`; raw sensor
  observations are immutable.
- Detection thresholds and Trust Score are labeled illustrative/unvalidated.
- No RMSE or other synthetic-as-ground-truth accuracy metric is produced.

## Reproducible commands

```powershell
python scripts\build_inference_state.py --check
python -m unittest discover -s tests -v
python scripts\verify_milestone2.py
python scripts\verify_milestone3.py
```

The frontend integration smoke test is executed against a clean local uvicorn
instance using `npm run integration:test` from `rintik-frontend`.

## Evidence expected from the final M3 run

- all M1/M2 regression tests remain PASS;
- 16 days and 734 timestamps replay through real HTTP;
- exact checkpoint and sidecar hashes remain unchanged;
- three suspected incidents are persisted (two stuck, one outage);
- explicit outage confirmation yields 48 separately stored estimates;
- retry/restart/idempotency and payload-conflict behavior pass;
- corrupt synthetic fixture/provenance fails closed;
- backend outage never yields a success-shaped fake prediction.

Final observed durations and commit hash are recorded in the workspace
`worklogs/QA_EVIDENCE.md`, because that file is outside this Git repository.

## Not claimed

This milestone does not claim calibrated anomaly detection, independent field
accuracy, production concurrency, deployment readiness, or a completed M4
dashboard. The simulated audit is not a real accuracy audit.
