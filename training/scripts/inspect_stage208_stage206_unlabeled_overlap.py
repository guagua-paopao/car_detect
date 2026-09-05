#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage206", required=True, type=Path)
    parser.add_argument("--existing-unlabeled", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.stage206, args.existing_unlabeled):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or unsafe input: {path}")
    if args.output.exists():
        raise FileExistsError(args.output)

    with args.stage206.open("r", encoding="utf-8-sig", newline="") as stream:
        stage206 = list(csv.DictReader(stream))
    with args.existing_unlabeled.open("r", encoding="utf-8-sig", newline="") as stream:
        existing = list(csv.DictReader(stream))
    existing_by_sha = {
        row.get("sha256", "").strip().lower(): row
        for row in existing
        if row.get("sha256", "").strip()
    }
    train_rows = [row for row in stage206 if row.get("split", "").strip().lower() == "train"]
    validation_rows = [row for row in stage206 if row.get("split", "").strip().lower() == "validation"]
    overlap = [row for row in train_rows if row.get("sha256", "").strip().lower() in existing_by_sha]
    missing = [row for row in train_rows if row.get("sha256", "").strip().lower() not in existing_by_sha]
    overlap_existing = [existing_by_sha[row.get("sha256", "").strip().lower()] for row in overlap]
    report = {
        "schema_version": "stage208-stage206-existing-unlabeled-overlap-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_read_only_overlap_audit",
        "inputs": {
            "stage206": str(args.stage206.resolve()),
            "stage206_sha256": sha256_path(args.stage206),
            "existing_unlabeled": str(args.existing_unlabeled.resolve()),
            "existing_unlabeled_sha256": sha256_path(args.existing_unlabeled),
        },
        "counts": {
            "stage206_rows": len(stage206),
            "stage206_train_rows": len(train_rows),
            "stage206_validation_rows_isolated": len(validation_rows),
            "existing_unlabeled_rows": len(existing),
            "stage206_train_rows_already_in_existing_unlabeled": len(overlap),
            "stage206_train_rows_absent_from_existing_unlabeled": len(missing),
            "overlap_existing_source_dataset": dict(sorted(Counter(row.get("source_dataset", "") for row in overlap_existing).items())),
            "overlap_existing_weather": dict(sorted(Counter(row.get("weather", "") for row in overlap_existing).items())),
            "overlap_existing_lighting": dict(sorted(Counter(row.get("lighting", "") for row in overlap_existing).items())),
            "overlap_existing_stage70_origin": dict(sorted(Counter(row.get("stage70_origin", "") for row in overlap_existing).items())),
        },
        "interpretation": (
            "all Stage206 train rows were already consumed by the Stage167 unlabeled consistency manifest; do not append or duplicate"
            if len(overlap) == len(train_rows)
            else "some Stage206 train rows are new to the Stage167 unlabeled consistency manifest"
        ),
        "policy": {
            "read_only": True,
            "validation_pixels_opened": False,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "training_started": False,
            "deployment_performed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{sha256_path(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
