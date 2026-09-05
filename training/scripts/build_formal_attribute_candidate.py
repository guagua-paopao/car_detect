#!/usr/bin/env python3
"""Filter Stage 2 formal-train candidates without changing the frozen eval set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("input manifest is empty")
    fields = list(rows[0])
    for field in ("formal_train_eligible", "color_review_status", "hard_example_priority"):
        if field not in fields:
            fields.append(field)
    eligible: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    for row in rows:
        license_ok = (row.get("license_train_eligible") or "false").lower() == "true"
        reviewed = (row.get("review_status") or "approved").lower() == "approved"
        quality_ok = (row.get("crop_quality") or "unknown").lower() != "poor"
        train_ok = row.get("split") != "train" or (license_ok and reviewed and quality_ok)
        out = dict(row)
        out["formal_train_eligible"] = "true" if row.get("split") == "train" and train_ok else "false"
        source = row.get("source_dataset") or "unknown"
        if row.get("color_review_status") in {"agent_auto_approved", "agent_auto_rejected_unknown"}:
            # The agent-only audit is fail-closed: accepted labels passed a
            # multi-view agreement gate; rejected colors are explicitly
            # unsupervised unknowns and never become pseudo-labels.
            out["color_review_status"] = row.get("color_review_status")
        elif source == "VCoR" and row.get("color") not in {"", "unknown"}:
            out["color_review_status"] = "source_label_pending_double_review"
        elif row.get("color") in {"", "unknown"}:
            out["color_review_status"] = "unknown_by_policy"
        else:
            out["color_review_status"] = "approved_source_or_human"
        if row.get("vehicle_size") == "small" or row.get("vehicle_size") == "medium" or row.get("lighting") in {"low_light", "night"}:
            out["hard_example_priority"] = "high"
        elif row.get("crop_quality") == "usable":
            out["hard_example_priority"] = "medium"
        else:
            out["hard_example_priority"] = "normal"
        (eligible if train_ok else excluded).append(out)
    # Preserve all validation/test rows; only the train split is filtered.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(eligible)
    report = {
        "schema_version": "1.0",
        "dataset_version": "attribute-domain-v2-formal-candidate",
        "input_manifest": str(args.input),
        "output_manifest": str(args.output),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "policy": {
            "train": "license_train_eligible=true, review_status=approved, crop_quality!=poor",
            "validation_test": "preserved from frozen attribute-domain-v2; never used to train",
            "bmd45_color": "unknown; no pseudo-color labels",
            "vcor_color": "agent_auto_approved only; disagreements are unknown and unsupervised",
        },
        "input_counts": {"rows": len(rows), "split": dict(sorted(Counter(r.get("split", "") for r in rows).items())), "source": dict(sorted(Counter(r.get("source_dataset", "") for r in rows).items()))},
        "candidate_counts": {"rows": len(eligible), "split": dict(sorted(Counter(r.get("split", "") for r in eligible).items())), "source": dict(sorted(Counter(r.get("source_dataset", "") for r in eligible).items())), "hard_example_priority": dict(sorted(Counter(r.get("hard_example_priority", "") for r in eligible).items()))},
        "excluded_train_counts": {"rows": len(excluded), "source": dict(sorted(Counter(r.get("source_dataset", "") for r in excluded).items())), "license": dict(sorted(Counter(r.get("source_license", "") for r in excluded).items())), "quality": dict(sorted(Counter(r.get("crop_quality", "") for r in excluded).items()))},
        "color_review_queue": {"rows": sum(1 for r in eligible if r.get("split") == "train" and r.get("color_review_status") == "source_label_pending_double_review"), "status": "clear" if not any(r.get("split") == "train" and r.get("color_review_status") == "source_label_pending_double_review" for r in eligible) else "pending_double_review"},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
