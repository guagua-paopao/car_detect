#!/usr/bin/env python3
"""Safely apply a completed VCoR double-review queue to train rows.

The command is deliberately fail-closed. It refuses pending, malformed, or
duplicate reviews and never changes validation/test labels, preserving the
independent evaluation split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = ROOT / "config" / "vehicle_labels.v1.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"review merge rejected: {message}")


def load_allowed_colors() -> set[str]:
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    colors = set(labels["colors"])
    if "unknown" not in colors:
        fail("label contract has no color unknown")
    return colors


def main() -> int:
    args = parse_args()
    allowed = load_allowed_colors()
    with args.candidate.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        candidate_fields = reader.fieldnames or []
        candidate = list(reader)
    with args.queue.open("r", encoding="utf-8", newline="") as fh:
        queue_reader = csv.DictReader(fh)
        queue = list(queue_reader)
        queue_fields = queue_reader.fieldnames or []
    required_queue_fields = {"review_id", "split", "image_path", "sha256", "reviewer_1_color", "reviewer_2_color", "adjudicated_color", "disagreement", "review_status", "reviewer_1_id", "reviewer_2_id"}
    missing = required_queue_fields - set(queue_fields)
    if missing:
        fail(f"queue missing columns: {sorted(missing)}")

    candidate_vcor = {
        (row["split"], row["image_path"], row["sha256"]): row
        for row in candidate
        if row.get("source_dataset") == "VCoR" and row.get("color_supervised") == "true"
    }
    if len(candidate_vcor) != sum(1 for row in candidate if row.get("source_dataset") == "VCoR" and row.get("color_supervised") == "true"):
        fail("candidate VCoR keys are not unique")
    queue_keys = [(row.get("split", ""), row.get("image_path", ""), row.get("sha256", "")) for row in queue]
    if len(set(queue_keys)) != len(queue_keys):
        fail("queue contains duplicate crop keys")
    if set(queue_keys) != set(candidate_vcor):
        fail("queue does not exactly cover candidate VCoR color rows")

    updates: dict[tuple[str, str, str], str] = {}
    pending = 0
    disagreements = 0
    for row in queue:
        key = (row["split"], row["image_path"], row["sha256"])
        status = row["review_status"]
        first = row["reviewer_1_color"]
        second = row["reviewer_2_color"]
        adjudicated = row["adjudicated_color"]
        if status != "double_review_approved":
            pending += 1
            continue
        if not row["reviewer_1_id"] or not row["reviewer_2_id"]:
            fail(f"approved row {row['review_id']} lacks reviewer IDs")
        if first not in allowed or second not in allowed:
            fail(f"approved row {row['review_id']} has an invalid reviewer color")
        if first != second and not adjudicated:
            fail(f"approved disagreement {row['review_id']} lacks adjudication")
        chosen = adjudicated or first
        if chosen not in allowed:
            fail(f"row {row['review_id']} has invalid adjudicated color")
        if first != second:
            disagreements += 1
        updates[key] = chosen
    if pending:
        fail(f"{pending} rows are not double_review_approved")

    output_rows = []
    train_updated = 0
    eval_preserved = 0
    for row in candidate:
        key = (row.get("split", ""), row.get("image_path", ""), row.get("sha256", ""))
        if key in updates and row.get("split") == "train":
            row = dict(row)
            row["color"] = updates[key]
            row["color_review_status"] = "double_review_approved"
            row["label_confidence"] = "high" if updates[key] != "unknown" else "medium"
            train_updated += 1
        elif key in updates:
            # Frozen validation/test labels remain byte-for-byte unchanged.
            eval_preserved += 1
        output_rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=candidate_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    report = {
        "schema_version": "1.0",
        "status": "merged_train_only",
        "candidate_input": str(args.candidate),
        "candidate_input_sha256": sha256(args.candidate),
        "queue_input": str(args.queue),
        "queue_input_sha256": sha256(args.queue),
        "labels_contract": str(LABELS_PATH),
        "labels_contract_sha256": sha256(LABELS_PATH),
        "output_manifest": str(args.output),
        "output_manifest_sha256": sha256(args.output),
        "queue_rows": len(queue),
        "train_rows_updated": train_updated,
        "validation_test_rows_preserved": eval_preserved,
        "adjudicated_disagreements": disagreements,
        "evaluation_labels_modified": False,
        "next_gate": "run validate_training_inputs.py and independent validation before any model training",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
