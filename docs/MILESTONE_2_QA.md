# Milestone 2 — implementation and QA record

Date: 10 October 2026

Branch: `feature/nsc-final-daily-inference`

Scope: daily temporal orchestration and persistence only

## Engineering decision

Milestone 1 already supplied the frozen `.pth` runtime, deterministic state sidecar, SQLite tables, complete-prefix inference, and expanding/rolling post-processing. M2 therefore hardens and proves that design instead of replacing it.

The adapter keeps two distinct operations:

1. `pred_internal` is recomputed from the complete chronological prefix and compared with every previously archived value at tolerance `1e-10`.
2. Frozen P1→P2→P3 is recomputed from `pred_internal` over the active calendar-date window; only the current date is stored as the publication snapshot.

Dates 1–15 use an expanding date set. Date 16 uses dates 2–16, exactly 15 calendar dates and 716 observed rows. No row is fabricated for the partial first or final day.

Ingestion is serialized with `BEGIN IMMEDIATE`. Identical date/payload retries are idempotent. A different payload for an existing date and cross-date reuse of an idempotency key return explicit HTTP 409 errors. A partial unique SQLite index enforces key ownership; legacy state containing duplicate key ownership fails closed before index creation and is never rewritten automatically.

## Executed evidence

```powershell
python scripts\build_inference_state.py --check
python -m unittest discover -s tests -v
python scripts\verify_milestone2.py
```

Results:

- sidecar reproducibility: PASS, SHA256 `a24da3473bf42a10fb9bde05e37495e5b69adffde9193dd4648daa9e0898996e`;
- complete regression: 10 tests, PASS;
- M2 verifier: PASS;
- protected model SHA256: `adce8fda2f465679ea6e8fe31124822a45a555b9dc154e12076ef5b6140699b0`;
- input counts: `18, 14×48, 44`, total 734;
- active-window rows: `18, 66, 114, 162, 210, 258, 306, 354, 402, 450, 498, 546, 594, 642, 690, 716`;
- maximum prefix-vs-full `pred_internal` difference: `0.0`;
- maximum direct-vs-stored P1/P2/P3 difference: `0.0`;
- old-day and restart snapshot digests: identical;
- injected inference failure: no partial rows in `runs`, `internal_predictions`, or `published_predictions`;
- restart after day 8 and a separate uninterrupted replay: identical deterministic terminal snapshot.

The regression also preserves M1's 734/734 four-decimal equality with the official submission and corrupt-artifact fail-closed behavior.

## Limits retained intentionally

- The daily route accepts only the pinned official test scenario; arbitrary live field inputs are not validated.
- The rolling daily adapter and the official one-shot 734-row submission are distinct, documented execution profiles.
- SQLite is justified for this local single-operator prototype, not claimed as production-scale storage.
- Synthetic RH, monitoring, Recovery Mode, trust/audit scoring, frontend integration, and demo-readiness work remain outside M2.
- No remote push, merge, deployment, collaborator, permission, checkpoint, or formula change is part of this milestone.
