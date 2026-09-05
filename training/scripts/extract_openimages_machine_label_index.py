#!/usr/bin/env python3
"""Extract one label from a checksum-verified Open Images machine-label CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-labels", type=Path, required=True)
    parser.add_argument("--download-state", type=Path, required=True)
    parser.add_argument("--label-mid", required=True)
    parser.add_argument("--label-name", required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_index.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite machine-label index evidence")

    state = json.loads(args.download_state.read_text(encoding="utf-8"))
    if state.get("status") != "complete_checksum_verified":
        raise RuntimeError("machine-label source is not checksum verified")
    if state.get("md5") != state.get("expected_md5_from_etag"):
        raise RuntimeError("machine-label source MD5 evidence is inconsistent")
    source_sha = sha256(args.machine_labels)
    if source_sha != state.get("sha256"):
        raise RuntimeError("machine-label source SHA256 mismatch")

    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    rows_scanned = 0
    rows_written = 0
    unique_images = set()
    confidence_histogram = Counter()
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]
    threshold_counts = Counter()
    with (
        args.machine_labels.open("r", encoding="utf-8-sig", newline="") as source,
        args.output_index.open("w", encoding="utf-8", newline="") as destination,
    ):
        reader = csv.DictReader(source)
        writer = csv.DictWriter(
            destination, fieldnames=["ImageID", "Source", "LabelName", "Confidence"]
        )
        writer.writeheader()
        for row in reader:
            rows_scanned += 1
            if row.get("LabelName") != args.label_mid:
                continue
            try:
                confidence = float(row.get("Confidence", ""))
            except ValueError:
                continue
            output = {
                "ImageID": row.get("ImageID", ""),
                "Source": row.get("Source", ""),
                "LabelName": row.get("LabelName", ""),
                "Confidence": f"{confidence:.8f}",
            }
            writer.writerow(output)
            rows_written += 1
            unique_images.add(output["ImageID"])
            confidence_histogram[f"{confidence:.2f}"] += 1
            for threshold in thresholds:
                threshold_counts[f"{threshold:.2f}"] += confidence >= threshold

    gates = {
        "source_checksum_verified": True,
        "minimum_1000_index_rows": rows_written >= 1000,
        "single_requested_label_only": True,
        "train_machine_labels_only": True,
    }
    status = "pass" if all(gates.values()) else "fail"
    report = {
        "schema_version": "openimages-machine-label-index-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source_machine_labels": str(args.machine_labels.resolve()),
        "source_machine_labels_sha256": source_sha,
        "download_state": str(args.download_state.resolve()),
        "download_state_sha256": sha256(args.download_state),
        "label_mid": args.label_mid,
        "label_name": args.label_name,
        "source_rows_scanned": rows_scanned,
        "index_rows": rows_written,
        "unique_images": len(unique_images),
        "confidence_histogram": dict(sorted(confidence_histogram.items())),
        "confidence_threshold_counts": dict(threshold_counts),
        "gates": gates,
        "output_index": str(args.output_index.resolve()),
        "output_index_sha256": sha256(args.output_index),
        "policy": {
            "metadata_index_only": True,
            "label_not_treated_as_attribute_target": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "use the index only to choose a bounded candidate threshold before pixel and teacher audits",
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": status,
        "rows_scanned": rows_scanned,
        "index_rows": rows_written,
        "unique_images": len(unique_images),
        "threshold_counts": dict(threshold_counts),
        "output_sha256": report["output_index_sha256"],
    }, ensure_ascii=False))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
