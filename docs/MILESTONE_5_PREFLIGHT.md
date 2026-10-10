# Milestone 5 backend preflight

`scripts/preflight_demo.py` is a read-only/fail-closed verification layer for
the final local demonstration. It does not run predictions, retrain, rewrite a
sidecar, or mutate an existing runtime database.

## Clean-schema preflight

```powershell
python scripts\preflight_demo.py --output ..\qa\evidence\m5-final-readiness\preflight.json
```

The script checks:

- every runtime requirement is exactly pinned and the installed public version
  matches (a local suffix such as `+cpu` is allowed);
- checkpoint, sidecar, train, and test provenance through `FrozenRuntime`;
- the 16 fixture files and canonical `18 + 14×48 + 44 = 734` row pattern;
- the synthetic fixture hash and explicit `model_input=false` /
  `accuracy_ground_truth=false` declarations;
- all required SQLite tables, `PRAGMA integrity_check`, and foreign keys on a
  disposable database under `Prototipe/qa/runtime`.

## Live preflight

```powershell
python scripts\preflight_demo.py `
  --database ..\qa\runtime\m5-demo.sqlite3 `
  --api-url http://127.0.0.1:8140 `
  --output ..\qa\evidence\m5-final-readiness\preflight-live.json
```

The live response must be HTTP 200 and report `ready=true`,
`source=real_original_pth`, `simulation=false`, exact model/state hashes, and
separately ready synthetic-demo monitoring. A mismatch is a non-zero exit.

This preflight proves the tested host is internally consistent. It is not a
substitute for executing the full procedure on a separate clean Windows
machine and is not a production deployment certification.
