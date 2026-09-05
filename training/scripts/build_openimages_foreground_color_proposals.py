#!/usr/bin/env python3
"""Create train-only foreground color proposals for Open Images vehicle crops.

Proposals are not approved labels.  They must pass a subsequent multi-teacher
augmentation audit before they can enter any training manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_foreground_function():
    path = Path(__file__).with_name("build_bmd45_track_color_pseudolabels.py")
    spec = importlib.util.spec_from_file_location("stage52_foreground", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load foreground helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.foreground_color_evidence, path


def preserve_verified_lighting(upstream_lighting: str, is_low_light_proxy: bool) -> str:
    if upstream_lighting not in {"", "unknown"}:
        return upstream_lighting
    return "low_light_proxy" if is_low_light_proxy else "non_low_light_proxy"


def validate_attribution(path: Path, required_image_ids: set[str]) -> int:
    licensed_image_ids = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = row.get("open_images_id") or row.get("image_id") or ""
            license_evidence = " ".join((
                row.get("license_id", ""),
                row.get("license_url", ""),
            )).lower()
            if image_id and (
                "cc-by-2.0" in license_evidence
                or "creativecommons.org/licenses/by/2.0" in license_evidence
            ):
                licensed_image_ids.add(image_id)
    missing = sorted(required_image_ids - licensed_image_ids)
    if missing:
        raise RuntimeError(
            f"attribution does not cover {len(missing)} input source images; examples={missing[:5]}"
        )
    return len(required_image_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-card", type=Path, required=True)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output_manifest, args.output_report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {path}")

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    foreground_color_evidence, helper_path = load_foreground_function()
    targets = [
        index for index, row in enumerate(rows)
        if row.get("split") == "train" and row.get("color", "") in {"", "unknown"}
    ]
    if not targets:
        raise RuntimeError("no Open Images train rows without color labels")
    required_source_image_ids = {
        rows[index].get("source_image_id") or rows[index].get("source_frame_id") or ""
        for index in targets
    } - {""}
    if not required_source_image_ids:
        raise RuntimeError("input rows do not identify Open Images source images")
    attribution_covered_images = validate_attribution(
        args.attribution, required_source_image_ids
    )

    proposal_counts = Counter()
    rejection_reasons = Counter()
    low_light_rows = 0
    small_rows = 0
    unreadable_rows = 0
    for index in targets:
        row = rows[index]
        path = (args.dataset_root / row["image_path"]).resolve()
        path.relative_to(args.dataset_root.resolve())
        evidence = foreground_color_evidence(path)
        proposal = str(evidence.get("label", "unknown"))
        if proposal == "unknown":
            rejection_reasons[str(evidence.get("reason", "unknown"))] += 1
            if evidence.get("reason") == "unreadable":
                unreadable_rows += 1
            continue
        with Image.open(path) as image:
            width, height = image.size
        is_small = min(width, height) < 48 or width * height < 12288
        is_low_light = float(evidence.get("mean_value", 256.0)) < 75.0
        upstream_lighting = row.get("lighting", "")
        upstream_manifest = row.get("source_manifest", "")
        small_rows += int(is_small)
        low_light_rows += int(is_low_light)
        row.update({
            "body_type": row.get("body_type") or "unknown",
            "color": proposal,
            "body_type_supervised": "false",
            "color_supervised": "true",
            "annotation_source": "openimages_foreground_color_proposal_v1",
            "source_dataset": "Open-Images-V7",
            "source_manifest": str(args.source_card.resolve()),
            "upstream_source_manifest": upstream_manifest,
            "source_license": "CC-BY-2.0-image / CC-BY-4.0-annotation",
            "license_train_eligible": "true",
            "review_status": "pending_teacher_audit",
            "color_review_status": "foreground_proposal_pending_teacher_audit",
            "review_method": "foreground_pixels_only_proposal_not_approved_label",
            "review_score": f"{float(evidence['score']):.6f}",
            "formal_train_eligible": "false",
            "pseudo_label": "true",
            "pseudo_label_confidence": "",
            "vehicle_size": row.get("vehicle_size") or ("small" if is_small else "medium_or_large"),
            "lighting": preserve_verified_lighting(upstream_lighting, is_low_light),
            "foreground_lighting_proxy": (
                "low_light_proxy" if is_low_light else "non_low_light_proxy"
            ),
            "foreground_low_light_proxy": str(is_low_light).lower(),
            "stage52_foreground_score": f"{float(evidence['score']):.6f}",
            "stage52_foreground_margin": f"{float(evidence['margin']):.6f}",
            "stage52_mean_value": f"{float(evidence.get('mean_value', 0.0)):.6f}",
            "stage52_mean_saturation": f"{float(evidence.get('mean_saturation', 0.0)):.6f}",
        })
        proposal_counts[proposal] += 1

    additions = [
        "body_type_supervised", "color_supervised", "annotation_source", "source_dataset",
        "source_manifest", "source_license", "license_train_eligible", "color_review_status",
        "review_method", "review_score", "formal_train_eligible", "pseudo_label",
        "pseudo_label_confidence", "vehicle_size", "lighting", "upstream_source_manifest",
        "foreground_lighting_proxy", "foreground_low_light_proxy", "stage52_foreground_score",
        "stage52_foreground_margin", "stage52_mean_value", "stage52_mean_saturation",
    ]
    output_fields = fields + [field for field in additions if field not in fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    proposed_rows = sum(proposal_counts.values())
    report = {
        "schema_version": "openimages-foreground-color-proposal-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if proposed_rows and len(proposal_counts) >= 4 else "fail",
        "input_manifest": str(args.manifest.resolve()),
        "input_manifest_sha256": sha256(args.manifest),
        "source_card": str(args.source_card.resolve()),
        "source_card_sha256": sha256(args.source_card),
        "attribution": str(args.attribution.resolve()),
        "attribution_sha256": sha256(args.attribution),
        "attribution_covered_input_images": attribution_covered_images,
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "foreground_helper": str(helper_path.resolve()),
        "foreground_helper_sha256": sha256(helper_path),
        "source_dataset": "Open Images V7",
        "source_license": "CC-BY-2.0-image / CC-BY-4.0-annotation",
        "eligible_unknown_color_train_rows": len(targets),
        "foreground_proposal_rows": proposed_rows,
        "foreground_proposal_color_counts": dict(sorted(proposal_counts.items())),
        "foreground_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "proposal_low_light_proxy_rows": low_light_rows,
        "proposal_small_rows": small_rows,
        "unreadable_rows": unreadable_rows,
        "policy": {
            "proposals_are_not_approved_labels": True,
            "upstream_verified_lighting_preserved": True,
            "all_input_source_images_covered_by_cc_by_2_attribution": True,
            "multi_teacher_audit_required": True,
            "train_images_opened_only": True,
            "validation_or_test_images_opened": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "run fail-closed multi-teacher augmentation audit before any proposal may enter training",
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "targets": len(targets), "proposals": proposed_rows,
        "counts": dict(sorted(proposal_counts.items())), "low_light": low_light_rows,
        "small": small_rows,
    }, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
