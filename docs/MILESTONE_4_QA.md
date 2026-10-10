# Milestone 4 backend QA — operator read models and export

Test date: 10 October 2026

Baseline: `c2e7474` on `feature/nsc-final-daily-inference`

## Scope and decision

M3 already owned inference, persistence, synthetic monitoring, incidents,
audit gates, and recovery. M4 therefore adds only read-only operator views over
that state. It does not duplicate inference, mutate snapshots, or introduce a
second persistence layer.

The feature catalog deliberately excludes `env_id` (identifier) and
`air_humidity` (model target/output). All 23 physical or engineered input
features have explicit groups and units. Stored source nulls remain JSON nulls;
the UI is responsible for rendering gaps, not imputing them.

The CSV export is generated from completed immutable publication state and
optional monitoring tables. A degraded/unavailable monitoring subsystem does
not hide real model results: monitoring columns remain empty while model
lineage stays present. Strings that can trigger spreadsheet formula execution
are prefixed with an apostrophe.

## Executed evidence

| Command/scenario | Result | Evidence |
|---|---|---|
| `python -m unittest tests.test_operator_api tests.test_api -v` | PASS | 11 tests; exact series/nulls, 18-row partial day, 23-feature catalog, export lineage, missing-state errors, degraded monitoring, HTTP headers |
| `python -m unittest discover -s tests -v` | PASS | 21 tests in 105.974 s; all M1–M4 regression tests |
| `python scripts/build_inference_state.py --check` | PASS | sidecar SHA256 `a24da3473bf42a10fb9bde05e37495e5b69adffde9193dd4648daa9e0898996e`, 36,245 bytes |
| `python scripts/verify_milestone1.py` | PASS | 734/734 parity at 4 dp; max and mean absolute difference 0 |
| `python scripts/verify_milestone2.py` | PASS | prefix and P1/P2/P3 max difference 0; snapshots/restart/idempotency unchanged |
| `python scripts/verify_milestone3.py` | PASS | 734 separate synthetic rows; 3 incidents; 48 estimated recovery rows; raw outage preserved null |
| Fresh-database real HTTP replay from the frontend verifier | PASS | 16 dates, 734 rows, 23 features, 48-row outage export, correct stuck/outage/repaired labels |
| Restart with the same HTTP database | PASS | ready; 16 completed and 16 monitored days; series/export each returned 48 outage rows; `nosniff` present |

Protected artifact hashes remained:

- model: `adce8fda2f465679ea6e8fe31124822a45a555b9dc154e12076ef5b6140699b0`;
- sidecar: `a24da3473bf42a10fb9bde05e37495e5b69adffde9193dd4648daa9e0898996e`;
- synthetic fixture: `891070499559c984b0761ba04d5a705e7a52bd181fb58d033e18a519eccf63f3`.

No checkpoint, sidecar, official source, inference formula, or database schema
from M1–M3 was modified by M4.

## Remaining boundary

These routes are for the pinned local NSC fixture and single-operator SQLite
prototype. They are not evidence of arbitrary field-data support or production
concurrency. Thresholds and audit states remain explicitly illustrative.
