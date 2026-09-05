#!/usr/bin/env python3
"""Append licensed adverse Open Images evidence with exact or partial truth only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "frozen_video")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def keyed(rows: list[dict[str, str]], name: str) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        key = row.get("crop_sha256") or row.get("sha256")
        if not key or key in result:
            raise RuntimeError(f"{name} has an empty or duplicate crop hash")
        result[key] = row
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--body-consensus", type=Path, required=True)
    parser.add_argument("--color-consensus", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite Stage65 manifest evidence")
    for path in (args.base, args.proposals, args.body_consensus, args.color_consensus, args.labels):
        if not path.is_file():
            raise FileNotFoundError(path)

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    body_labels = set(map(str, labels["body_types"]))
    color_labels = set(map(str, labels["colors"]))
    if not {"truck", "bus", "sedan", "suv", "mpv", "other", "unknown"}.issubset(body_labels):
        raise RuntimeError("Stage65 requires taxonomy-v2 body labels")
    if not {"gray", "silver", "unknown"}.issubset(color_labels):
        raise RuntimeError("Stage65 requires taxonomy-v2 fine-color labels")

    base_fields, base_rows = read_csv(args.base)
    proposal_fields, proposals = read_csv(args.proposals)
    _, body_rows = read_csv(args.body_consensus)
    _, color_rows = read_csv(args.color_consensus)
    body_by_hash = keyed(body_rows, "body consensus")
    color_by_hash = keyed(color_rows, "color consensus")
    if len(proposals) != len(body_by_hash) or len(proposals) != len(color_by_hash):
        raise RuntimeError("proposal and consensus row counts differ")

    base_hashes = {row.get("sha256") or row.get("crop_sha256") for row in base_rows}
    base_hashes.discard(None)
    heldout_groups = {
        row.get("source_image_id") or row.get("source_frame_id")
        for row in base_rows if row.get("split") in {"validation", "test"}
    }
    heldout_groups.discard(None)
    extra_fields = [
        "coarse_body_family", "adverse_supervised", "stage65_body_truth_source",
        "stage65_color_truth_source", "stage65_source_manifest",
    ]
    output_fields = list(base_fields)
    for field in [*proposal_fields, *extra_fields]:
        if field not in output_fields:
            output_fields.append(field)

    appended = []
    counts = Counter()
    source_groups = set()
    for proposal in proposals:
        crop_hash = proposal.get("crop_sha256") or proposal.get("sha256")
        if not crop_hash or crop_hash in base_hashes:
            if crop_hash in base_hashes:
                counts["duplicate_base_crop_skipped"] += 1
                continue
            raise RuntimeError("proposal has no crop SHA256")
        if proposal.get("split") != "train":
            raise RuntimeError("adverse proposal is not train-only")
        if proposal.get("source_dataset") != "Open-Images-V7":
            raise RuntimeError("unexpected Stage65 proposal source")
        if not truthy(proposal.get("license_train_eligible")):
            raise RuntimeError("Stage65 proposal is not license-train-eligible")
        official = str(proposal.get("official_vehicle_class", "")).lower()
        if official not in {"car", "bus", "truck"}:
            raise RuntimeError(f"unsupported official vehicle class: {official}")
        marker_text = " ".join(str(value).lower() for value in proposal.values())
        if any(marker in marker_text for marker in FROZEN_MARKERS):
            raise RuntimeError("Stage65 proposal contains a frozen-video marker")
        group = proposal.get("source_image_id") or proposal.get("source_frame_id")
        if not group or group in heldout_groups:
            raise RuntimeError("Stage65 source group is empty or leaks into held-out data")
        body = body_by_hash.get(crop_hash)
        color = color_by_hash.get(crop_hash)
        if body is None or color is None:
            raise RuntimeError("Stage65 consensus evidence is missing")
        if body.get("image_path") != proposal.get("image_path") or color.get("image_path") != proposal.get("image_path"):
            raise RuntimeError("Stage65 consensus image path mismatch")

        output = {field: "" for field in output_fields}
        output.update(proposal)
        output["sha256"] = crop_hash
        output["split"] = "train"
        output["review_status"] = "approved"
        output["formal_train_eligible"] = "true"
        output["adverse_supervised"] = "true"
        output["stage65_source_manifest"] = str(args.proposals.resolve())
        output["color"] = "unknown"
        output["color_supervised"] = "false"
        output["stage65_color_truth_source"] = "none_fail_closed"
        output["coarse_body_family"] = ""
        output["pseudo_label"] = "false"

        body_label = str(body.get("body_type", "unknown"))
        if truthy(body.get("body_type_supervised")) and body_label in body_labels and body_label != "unknown":
            output["body_type"] = body_label
            output["body_type_supervised"] = "true"
            output["stage65_body_truth_source"] = "strict_multiteacher_exact"
            output["pseudo_label"] = "true"
            counts[f"body_exact:{body_label}"] += 1
        elif official == "bus":
            output["body_type"] = "bus"
            output["body_type_supervised"] = "true"
            output["stage65_body_truth_source"] = "official_openimages_bus"
            counts["body_exact:bus"] += 1
        elif official == "truck":
            output["body_type"] = "truck"
            output["body_type_supervised"] = "true"
            output["stage65_body_truth_source"] = "official_openimages_coarse_truck"
            counts["body_coarse:truck"] += 1
        else:
            output["body_type"] = "unknown"
            output["body_type_supervised"] = "false"
            output["coarse_body_family"] = "car"
            output["stage65_body_truth_source"] = "official_openimages_partial_car_family"
            counts["body_partial:car"] += 1

        color_label = str(color.get("color", "unknown"))
        if truthy(color.get("color_supervised")) and color_label in color_labels and color_label != "unknown":
            output["color"] = color_label
            output["color_supervised"] = "true"
            output["stage65_color_truth_source"] = "strict_multiteacher_exact"
            output["pseudo_label"] = "true"
            counts[f"color_exact:{color_label}"] += 1
        elif truthy(color.get("color_supervised")):
            counts[f"color_merged_rejected:{color_label}"] += 1

        try:
            area_ratio = float(proposal.get("source_bbox_area_ratio", "") or 1.0)
        except ValueError:
            area_ratio = 1.0
        if area_ratio <= 0.02:
            output["small_target"] = "true"
            output["vehicle_size"] = "small"
            counts["small"] += 1
        if truthy(output.get("night")):
            counts["night"] += 1
        if truthy(output.get("occluded")) or truthy(output.get("truncated")):
            counts["occluded_or_truncated"] += 1
        source_groups.add(group)
        base_hashes.add(crop_hash)
        appended.append(output)

    if len(appended) < 4000 or counts["night"] < 1600:
        raise RuntimeError("Stage65 adverse evidence quantity unexpectedly regressed")
    all_rows = [*base_rows, *appended]
    split_counts_before = Counter(row.get("split") for row in base_rows)
    split_counts_after = Counter(row.get("split") for row in all_rows)
    for split in ("validation", "test"):
        if split_counts_after[split] != split_counts_before[split]:
            raise RuntimeError(f"Stage65 changed {split} membership")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    report = {
        "schema_version": "stage65-adverse-supervised-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "inputs": {
            "base": {"path": str(args.base.resolve()), "sha256": sha256(args.base)},
            "proposals": {"path": str(args.proposals.resolve()), "sha256": sha256(args.proposals)},
            "body_consensus": {"path": str(args.body_consensus.resolve()), "sha256": sha256(args.body_consensus)},
            "color_consensus": {"path": str(args.color_consensus.resolve()), "sha256": sha256(args.color_consensus)},
            "labels": {"path": str(args.labels.resolve()), "sha256": sha256(args.labels)},
        },
        "output_manifest": str(args.output.resolve()),
        "output_manifest_sha256": sha256(args.output),
        "base_rows": len(base_rows), "appended_rows": len(appended), "total_rows": len(all_rows),
        "appended_unique_source_images": len(source_groups),
        "counts": dict(sorted(counts.items())),
        "split_counts": dict(sorted(split_counts_after.items())),
        "policy": {
            "official_car_is_partial_family_not_exact_subtype": True,
            "official_truck_is_hierarchical_coarse_truth": True,
            "unsupported_merged_color_truth_rejected": True,
            "validation_and_test_membership_preserved": True,
            "same_source_heldout_leakage": 0,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.report)
    print(json.dumps({"status": "pass", "appended": len(appended), "output_sha256": report["output_manifest_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
