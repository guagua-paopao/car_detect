#!/usr/bin/env python3
"""Build per-image Open Images attribution from a hash-verified proposal manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--output-attribution", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_attribution.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Open Images attribution evidence")

    input_report = json.loads(args.manifest_report.read_text(encoding="utf-8"))
    if input_report.get("status") != "pass":
        raise RuntimeError("combined proposal manifest did not pass")
    if input_report.get("output_manifest_sha256") != sha256(args.manifest):
        raise RuntimeError("combined proposal manifest hash mismatch")
    policy = input_report.get("policy", {})
    if not (
        policy.get("all_rows_train_only") is True
        and policy.get("frozen_video_used") is False
    ):
        raise RuntimeError("combined proposal policy is not isolated")

    images: dict[str, dict[str, str]] = {}
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = row.get("source_image_id") or row.get("source_frame_id") or ""
            license_url = row.get("source_license_url", "")
            if not image_id:
                raise RuntimeError("proposal row has no source image ID")
            if "creativecommons.org/licenses/by/2.0" not in license_url:
                raise RuntimeError(f"proposal row has invalid image license for {image_id}")
            attribution = {
                "sample_id": f"oi_train_{image_id}",
                "image_id": image_id,
                "split": "train",
                "license_id": "CC-BY-2.0-image / CC-BY-4.0-annotation",
                "license_url": license_url,
                "author": row.get("source_author", ""),
                "landing_url": row.get("source_landing_url", ""),
                "source_image_sha256": row.get("source_image_sha256", ""),
                "proposal_source_pool": row.get("proposal_source_pool", ""),
            }
            previous = images.get(image_id)
            if previous is not None and previous != attribution:
                raise RuntimeError(f"conflicting attribution for source image {image_id}")
            images[image_id] = attribution
    if not images:
        raise RuntimeError("no attribution rows were produced")

    rows = [images[image_id] for image_id in sorted(images)]
    args.output_attribution.parent.mkdir(parents=True, exist_ok=True)
    with args.output_attribution.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "openimages-proposal-attribution-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "input_manifest": str(args.manifest.resolve()),
        "input_manifest_sha256": sha256(args.manifest),
        "input_report": str(args.manifest_report.resolve()),
        "input_report_sha256": sha256(args.manifest_report),
        "attributed_unique_images": len(rows),
        "output_attribution": str(args.output_attribution.resolve()),
        "output_attribution_sha256": sha256(args.output_attribution),
        "policy": {
            "per_image_cc_by_2_url_required": True,
            "annotation_license_recorded": True,
            "source_author_and_landing_url_retained": True,
            "source_image_sha256_retained": True,
            "train_only": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False
        },
        "decision": "use this attribution manifest in every downstream color proposal and model-card artifact"
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "pass",
        "images": len(rows),
        "output_sha256": report["output_attribution_sha256"]
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
