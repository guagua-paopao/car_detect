#!/usr/bin/env python3
"""Build a blinded, auditable two-person review packet for VCoR colors.

The packet intentionally does not alter the candidate manifest or any labels. A
reviewer-facing queue is blinded to the source color; a separate audit key keeps
the source label available for later reconciliation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "training" / "artifacts" / "attribute-domain-v2"
INPUT = ART / "attribute_manifest.formal-candidate.csv"
OUT = ART / "vcor-color-double-review"
REMOTE_DATASET_ROOT = "/root/autodl-tmp/vcas/datasets/dataset-domain-bmd45-attributes-merged-v1"
LABELS_PATH = ROOT / "config" / "vehicle_labels.v1.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_allowed_colors() -> list[str]:
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    colors = list(labels["colors"])
    if "unknown" not in colors:
        raise RuntimeError("label contract must include color unknown")
    return colors


def main() -> int:
    with INPUT.open("r", encoding="utf-8", newline="") as fh:
        source_rows = list(csv.DictReader(fh))
    allowed_colors = load_allowed_colors()
    vcor = [
        row for row in source_rows
        if row.get("source_dataset") == "VCoR"
        and row.get("color_supervised") == "true"
    ]
    vcor.sort(key=lambda row: (row.get("sha256", ""), row.get("image_path", "")))
    if not vcor:
        raise RuntimeError("no VCoR color rows found in formal candidate")

    queue_rows: list[dict[str, str]] = []
    key_rows: list[dict[str, str]] = []
    for index, row in enumerate(vcor, start=1):
        review_id = f"VCOR-COLOR-{index:05d}"
        relative_crop = row["image_path"].replace("../dataset-domain-bmd45-attributes-merged-v1/", "")
        remote_image_path = f"{REMOTE_DATASET_ROOT}/{relative_crop}"
        common = {
            "review_id": review_id,
            "split": row["split"],
            "image_path": row["image_path"],
            "remote_image_path": remote_image_path,
            "source_frame_id": row["source_frame_id"],
            "sha256": row["sha256"],
            "crop_quality": row["crop_quality"],
            "lighting": row["lighting"],
            "vehicle_size": row["vehicle_size"],
            "occlusion_level": row["occlusion_level"],
        }
        queue_rows.append({
            **common,
            "reviewer_1_color": "",
            "reviewer_2_color": "",
            "adjudicated_color": "",
            "disagreement": "",
            "review_status": "pending",
            "reviewer_1_id": "",
            "reviewer_2_id": "",
            "reviewed_at_utc": "",
        })
        key_rows.append({
            **common,
            "source_color": row["color"],
            "source_review_status": row["color_review_status"],
            "source_manifest": row["source_manifest"],
        })

    queue_fields = list(queue_rows[0])
    key_fields = list(key_rows[0])
    OUT.mkdir(parents=True, exist_ok=True)
    queue_path = OUT / "vcor-color-double-review-queue.csv"
    key_path = OUT / "vcor-color-double-review-audit-key.csv"
    write_csv(queue_path, queue_rows, queue_fields)
    write_csv(key_path, key_rows, key_fields)

    train_count = sum(1 for row in vcor if row.get("split") == "train")
    validation_count = sum(1 for row in vcor if row.get("split") == "validation")
    test_count = sum(1 for row in vcor if row.get("split") == "test")
    protocol = f"""# VCoR color double-review packet

This packet contains `{len(vcor):,}` VCoR color crops from the formal candidate
(`{train_count:,}` train, `{validation_count:,}` validation, `{test_count:,}` test).
It is a review input only: no labels in the candidate manifest have been changed.

## Review protocol

1. Each crop is reviewed independently by two reviewers, blinded to the source label.
2. Each reviewer selects exactly one of: `{', '.join(allowed_colors)}`.
3. Record reviewer IDs and UTC completion time in `vcor-color-double-review-queue.csv`.
4. If the two labels differ, set `disagreement=1` and leave `adjudicated_color` blank until an adjudicator resolves it. If they agree, set `disagreement=0` and copy the agreed value into `adjudicated_color`.
5. Set `review_status=double_review_approved` only for exact agreement (including an explicit `unknown`). Use `adjudication_required` for disagreements and `rejected` for unusable crops.
6. Do not edit `vcor-color-double-review-audit-key.csv`; it is the reconciliation key for the existing source label and provenance.

## Release gate

The VCoR rows remain excluded from formal color training until the completed queue is reconciled, disagreement rows are adjudicated, and the rebuilt manifest passes the training-input validator. The fixed replay video is not a tuning or review-selection source.

## Integrity

- Candidate input SHA-256: `{sha256(INPUT)}`
- Queue SHA-256: `{sha256(queue_path)}`
- Audit-key SHA-256: `{sha256(key_path)}`
"""
    protocol_path = OUT / "README.md"
    protocol_path.write_text(protocol, encoding="utf-8")
    report = {
        "schema_version": "1.0",
        "status": "ready_for_two_person_review",
        "source": "VCoR",
        "rows": len(vcor),
        "split_counts": {
            "train": train_count,
            "validation": validation_count,
            "test": test_count,
        },
        "candidate_input": str(INPUT),
        "candidate_input_sha256": sha256(INPUT),
        "reviewer_queue": str(queue_path),
        "reviewer_queue_sha256": sha256(queue_path),
        "audit_key": str(key_path),
        "audit_key_sha256": sha256(key_path),
        "allowed_colors": allowed_colors,
        "labels_contract": str(LABELS_PATH),
        "labels_contract_sha256": sha256(LABELS_PATH),
        "labels_modified": False,
        "release_gate": "completed two-person review plus adjudication and rebuilt-manifest validation",
    }
    report_path = OUT / "review-packet-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
