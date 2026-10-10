"""Fail-closed preflight for the local RINTIK Grand Final demonstration.

This script intentionally performs no inference and never mutates an existing
runtime database.  It validates the frozen runtime, pinned dependencies,
official daily fixtures, synthetic-demo provenance, and the SQLite schema that
would be created for a new demo run.  With ``--api-url`` it also validates the
live readiness response produced by the exact database supplied with
``--database``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROTOTYPE_DIR = BACKEND_DIR.parent
WORKSPACE_DIR = PROTOTYPE_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

import inference as inf  # noqa: E402
from daily_batch import DailyBatchService, FrozenRuntime  # noqa: E402
from monitoring import MonitoringService, SYNTHETIC_FIXTURE_SHA256  # noqa: E402


REQUIRED_TABLES = {
    "daily_inputs",
    "runs",
    "internal_predictions",
    "published_predictions",
    "demo_sensor_observations",
    "model_health_logs",
    "demo_accuracy_audits",
    "monitoring_runs",
    "sensor_incidents",
    "incident_actions",
    "recovery_estimates",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def requirement_versions(path: Path) -> dict[str, dict[str, str]]:
    observed: dict[str, dict[str, str]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "-r ")):
            continue
        if "==" not in line:
            raise RuntimeError(f"Unpinned runtime requirement: {line}")
        package, expected = line.split("==", 1)
        actual = importlib.metadata.version(package)
        # A local build suffix such as torch 2.2.1+cpu has the same pinned
        # public version and is acceptable for this CPU-only prototype.
        if actual.split("+", 1)[0] != expected:
            raise RuntimeError(
                f"Dependency mismatch for {package}: expected {expected}, got {actual}"
            )
        observed[package] = {"expected": expected, "actual": actual}
    return observed


def fixture_report(runtime: FrozenRuntime) -> dict[str, Any]:
    fixture_root = PROTOTYPE_DIR / "demo" / "fixtures"
    manifest_path = fixture_root / "day_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    days = manifest.get("calendar_dates", [])
    counts = [item.get("rows") for item in days]
    if len(days) != 16 or counts != [18, *([48] * 14), 44]:
        raise RuntimeError(f"Daily fixture count pattern is not canonical: {counts}")
    if manifest.get("total_rows") != 734:
        raise RuntimeError("Daily fixture manifest total is not 734")
    if manifest.get("source_test_sha256") != inf.OFFICIAL_TEST_SHA256:
        raise RuntimeError("Daily fixture manifest points to a non-official test source")

    actual_days = []
    fixture_dir = fixture_root / "inputs_original_test_by_day"
    for item in days:
        fixture_path = fixture_dir / item["file"]
        actual_hash = sha256_file(fixture_path)
        if actual_hash != item["sha256"]:
            raise RuntimeError(f"Fixture hash mismatch: {fixture_path.name}")
        sidecar_day = runtime.manifests.get(item["date"])
        if sidecar_day is None or sidecar_day["row_count"] != item["rows"]:
            raise RuntimeError(f"Fixture/sidecar mismatch for {item['date']}")
        actual_days.append(
            {"date": item["date"], "rows": item["rows"], "sha256": actual_hash}
        )

    synthetic_path = fixture_root / "synthetic_physical_humidity_SENSOR_SIMULATION.csv"
    provenance_path = fixture_root / "SOURCE_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    synthetic_hash = sha256_file(synthetic_path)
    if synthetic_hash != SYNTHETIC_FIXTURE_SHA256:
        raise RuntimeError("Synthetic sensor fixture hash mismatch")
    if provenance.get("model_input") is not False:
        raise RuntimeError("Synthetic fixture provenance must declare model_input=false")
    if provenance.get("accuracy_ground_truth") is not False:
        raise RuntimeError(
            "Synthetic fixture provenance must declare accuracy_ground_truth=false"
        )
    return {
        "manifest": str(manifest_path),
        "dates": len(days),
        "rows": sum(counts),
        "count_pattern": counts,
        "days": actual_days,
        "synthetic_fixture_sha256": synthetic_hash,
        "synthetic_model_input": False,
        "synthetic_accuracy_ground_truth": False,
    }


def schema_report(database_path: Path) -> dict[str, Any]:
    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            if not row[0].startswith("sqlite_")
        }
        missing = sorted(REQUIRED_TABLES - tables)
        if missing:
            raise RuntimeError(f"Runtime database is missing tables: {missing}")
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        if integrity != "ok" or foreign_key_errors:
            raise RuntimeError(
                f"SQLite integrity failed: integrity={integrity}, foreign_keys={foreign_key_errors}"
            )
        counts = {
            table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in sorted(REQUIRED_TABLES)
        }
    return {
        "database": str(database_path),
        "tables": sorted(tables),
        "required_tables": sorted(REQUIRED_TABLES),
        "integrity_check": integrity,
        "foreign_key_errors": 0,
        "row_counts": counts,
    }


def live_readiness(api_url: str) -> dict[str, Any]:
    url = api_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            status = response.status
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Live readiness request failed: {exc}") from exc
    required = {
        "ready": True,
        "source": "real_original_pth",
        "simulation": False,
        "model_sha256": inf.OFFICIAL_MODEL_SHA256,
        "state_sha256": inf.OFFICIAL_STATE_SHA256,
        "monitoring_ready": True,
        "monitoring_mode": "synthetic_demo",
        "monitoring_simulation": True,
    }
    mismatches = {
        key: {"expected": expected, "actual": body.get(key)}
        for key, expected in required.items()
        if body.get(key) != expected
    }
    if status != 200 or mismatches:
        raise RuntimeError(
            f"Live readiness mismatch: status={status}, fields={mismatches}"
        )
    return {"url": url, "http_status": status, "body": body}


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_path = Path(args.model).resolve()
    state_path = Path(args.state).resolve()
    train_path = Path(args.train).resolve()
    test_path = Path(args.test).resolve()
    started = time.perf_counter()
    checks: dict[str, Any] = {}

    checks["dependencies"] = requirement_versions(BACKEND_DIR / "requirements.txt")
    runtime_started = time.perf_counter()
    runtime = FrozenRuntime(model_path, state_path, train_path, test_path)
    checks["frozen_runtime"] = {
        "model_path": str(model_path),
        "model_sha256": runtime.model_sha256,
        "state_path": str(state_path),
        "state_sha256": runtime.state_sha256,
        "pipeline_version": inf.PIPELINE_VERSION,
        "ordered_dates": runtime.ordered_dates,
        "load_seconds": round(time.perf_counter() - runtime_started, 3),
    }
    checks["fixtures"] = fixture_report(runtime)

    temporary_database = args.database is None
    if temporary_database:
        temp_dir = (
            PROTOTYPE_DIR
            / "qa"
            / "runtime"
            / f"preflight-{uuid.uuid4().hex}"
        )
        temp_dir.mkdir(parents=True, exist_ok=False)
        database_path = temp_dir / "preflight.sqlite3"
    else:
        temp_dir = None
        database_path = Path(args.database).resolve()

    try:
        service = DailyBatchService(runtime, database_path)
        monitoring = MonitoringService(
            service,
            PROTOTYPE_DIR
            / "demo"
            / "fixtures"
            / "synthetic_physical_humidity_SENSOR_SIMULATION.csv",
            PROTOTYPE_DIR / "demo" / "fixtures" / "SOURCE_PROVENANCE.json",
        )
        checks["database_schema"] = schema_report(database_path)
        checks["database_schema"]["temporary"] = temporary_database
        if args.api_url:
            checks["live_readiness"] = live_readiness(args.api_url)
    finally:
        if temp_dir is not None:
            # Windows can retain SQLite/WAL handles briefly after the final
            # connection closes. Drop service references and retry boundedly;
            # never broaden deletion beyond this UUID-named preflight folder.
            if "monitoring" in locals():
                del monitoring
            if "service" in locals():
                del service
            gc.collect()
            cleanup_error: OSError | None = None
            for _ in range(20):
                try:
                    shutil.rmtree(temp_dir)
                    cleanup_error = None
                    break
                except OSError as exc:
                    cleanup_error = exc
                    time.sleep(0.1)
            if cleanup_error is not None:
                raise cleanup_error

    return {
        "status": "PASS",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "executable": sys.executable,
            "network_required": False,
        },
        "checks": checks,
        "total_seconds": round(time.perf_counter() - started, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=WORKSPACE_DIR / "NSC26-DS011_HBD_File Data Science.pth",
    )
    parser.add_argument("--state", default=BACKEND_DIR / "artifacts" / "inference_state.json")
    parser.add_argument("--train", default=PROTOTYPE_DIR / "reference" / "official" / "train.csv")
    parser.add_argument("--test", default=PROTOTYPE_DIR / "reference" / "official" / "test.csv")
    parser.add_argument(
        "--database",
        help="Existing/live demo database to inspect. Omit to validate a disposable clean schema.",
    )
    parser.add_argument("--api-url", help="Also require an exact live /health response.")
    parser.add_argument("--output", help="Write the JSON report to this path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except Exception as exc:
        report = {
            "status": "FAIL",
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error": f"{type(exc).__name__}: {exc}",
        }
        exit_code = 1
    else:
        exit_code = 0
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
