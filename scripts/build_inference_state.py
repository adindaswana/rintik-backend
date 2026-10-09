"""Build or verify the deterministic RINTIK inference-state sidecar."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROTOTYPE_DIR = BACKEND_DIR.parent
WORKSPACE_DIR = PROTOTYPE_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

import inference as inf  # noqa: E402


def checked_source(path: Path, expected_sha256: str) -> dict[str, object]:
    if not path.is_file():
        raise inf.ArtifactIntegrityError(f"Required official source not found: {path}")
    actual = inf.sha256_file(path)
    if actual != expected_sha256:
        raise inf.ArtifactIntegrityError(
            f"Official source SHA256 mismatch for {path.name}: expected {expected_sha256}, got {actual}"
        )
    return {"filename": path.name, "sha256": actual, "size_bytes": path.stat().st_size}


def build_bytes(model_path: Path, train_path: Path, test_path: Path) -> bytes:
    artifact, model_sha = inf.load_frozen_artifact(model_path)
    sources = {
        "model": {
            "filename": model_path.name,
            "sha256": model_sha,
            "size_bytes": model_path.stat().st_size,
        },
        "train": checked_source(train_path, inf.OFFICIAL_TRAIN_SHA256),
        "test": checked_source(test_path, inf.OFFICIAL_TEST_SHA256),
    }
    boosters = inf.load_boosters(artifact)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sidecar = inf.build_state_sidecar(artifact, boosters, train, test, sources)
    return inf.stable_json_bytes(sidecar)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=WORKSPACE_DIR / "NSC26-DS011_HBD_File Data Science.pth",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=PROTOTYPE_DIR / "reference" / "official" / "train.csv",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=PROTOTYPE_DIR / "reference" / "official" / "test.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BACKEND_DIR / "artifacts" / "inference_state.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Rebuild in memory and fail if output is absent or differs byte-for-byte.",
    )
    args = parser.parse_args()

    payload = build_bytes(args.model.resolve(), args.train.resolve(), args.test.resolve())
    if args.check:
        if not args.output.is_file():
            raise SystemExit(f"FAIL: sidecar does not exist: {args.output}")
        if args.output.read_bytes() != payload:
            raise SystemExit("FAIL: sidecar is not reproducible from the pinned sources")
        print(f"PASS sidecar reproducible sha256={inf.sha256_file(args.output)} bytes={len(payload)}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"WROTE {args.output}")
    print(f"sha256={inf.sha256_file(args.output)} bytes={len(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
