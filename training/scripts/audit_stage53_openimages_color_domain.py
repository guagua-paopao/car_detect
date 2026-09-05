#!/usr/bin/env python3
"""Filter Stage53 color labels to target vehicles and deduplicate crop evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


TARGET_CLASSES = {"car", "bus", "truck"}
CROP_PATTERN = re.compile(r"train_(oi_[^/]+)_(\d+)\.[^.]+$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def dhash(path: Path) -> int:
    with Image.open(path) as opened:
        image = opened.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(image.getdata())
    value = 0
    for y in range(8):
        row = y * 9
        for x in range(8):
            value = (value << 1) | int(pixels[row + x] > pixels[row + x + 1])
    return value


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teacher-report", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-card", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--dhash-distance", type=int, default=4)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage53 domain evidence")

    teacher_report = json.loads(args.teacher_report.read_text(encoding="utf-8"))
    if teacher_report.get("status") != "pass":
        raise RuntimeError("teacher audit did not pass")
    policy = teacher_report.get("policy", {})
    if policy.get("test_split_not_used") is not True or policy.get("frozen_video_not_used") is not True:
        raise RuntimeError("teacher audit isolation evidence is invalid")

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    selected = [
        row for row in rows
        if row.get("split") == "train"
        and row.get("color_teacher_consensus") == "accepted"
        and row.get("review_status") == "approved"
        and truthy(row.get("color_supervised"))
        and truthy(row.get("formal_train_eligible"))
    ]
    if len(selected) != int(teacher_report.get("accepted_rows", -1)):
        raise RuntimeError("accepted row count does not match teacher report")

    source_card = json.loads(args.source_card.read_text(encoding="utf-8"))
    samples = {sample["sample_id"]: sample for sample in source_card["samples"]}
    domain_rejections = Counter()
    domain_rows = []
    for row in selected:
        match = CROP_PATTERN.search(row["image_path"])
        if not match:
            domain_rejections["crop_name_unmapped"] += 1
            continue
        sample_id, annotation_index = match.group(1), int(match.group(2))
        sample = samples.get(sample_id)
        detections = sample.get("annotations", {}).get("detections", []) if sample else []
        if annotation_index >= len(detections):
            domain_rejections["source_annotation_unmapped"] += 1
            continue
        annotation = detections[annotation_index]
        official_class = str(annotation.get("vehicle_class", "unknown"))
        if official_class not in TARGET_CLASSES:
            domain_rejections[f"official_class_{official_class}"] += 1
            continue
        copied = dict(row)
        copied["official_vehicle_class"] = official_class
        copied["official_occluded"] = str(bool(annotation.get("occluded"))).lower()
        copied["official_truncated"] = str(bool(annotation.get("truncated"))).lower()
        copied["source_sample_id"] = sample_id
        image_path = (args.dataset_root.resolve() / row["image_path"]).resolve()
        image_path.relative_to(args.dataset_root.resolve())
        copied["stage53_dhash64"] = f"{dhash(image_path):016x}"
        copied["stage53_image_sha256"] = sha256(image_path)
        domain_rows.append(copied)

    union = UnionFind(len(domain_rows))
    by_color = defaultdict(list)
    for index, row in enumerate(domain_rows):
        by_color[row["color"]].append(index)
    for indexes in by_color.values():
        for offset, left in enumerate(indexes):
            left_hash = int(domain_rows[left]["stage53_dhash64"], 16)
            for right in indexes[offset + 1:]:
                right_hash = int(domain_rows[right]["stage53_dhash64"], 16)
                if (left_hash ^ right_hash).bit_count() <= args.dhash_distance:
                    union.union(left, right)
    components = defaultdict(list)
    for index in range(len(domain_rows)):
        components[union.find(index)].append(index)

    retained_indexes = []
    removed_near_duplicates = 0
    component_sizes = Counter()
    for indexes in components.values():
        component_sizes[len(indexes)] += 1
        best = max(
            indexes,
            key=lambda index: (
                float(domain_rows[index].get("color_teacher_confidence") or 0.0),
                float(domain_rows[index].get("review_score") or 0.0),
                domain_rows[index]["image_path"],
            ),
        )
        retained_indexes.append(best)
        removed_near_duplicates += len(indexes) - 1
    retained = [domain_rows[index] for index in sorted(retained_indexes)]

    counts = Counter(row["color"] for row in retained)
    color_class = Counter((row["color"], row["official_vehicle_class"]) for row in retained)
    low_light = sum(row.get("lighting") == "low_light_proxy" for row in retained)
    small = sum(row.get("vehicle_size") == "small" for row in retained)
    occluded_or_truncated = sum(
        truthy(row.get("official_occluded")) or truthy(row.get("official_truncated")) for row in retained
    )
    total = len(retained)
    quota = {
        "low_light_rows": low_light,
        "low_light_fraction": low_light / total if total else 0.0,
        "required_low_light_fraction": 0.30,
        "small_target_rows": small,
        "small_target_fraction": small / total if total else 0.0,
        "required_small_target_fraction": 0.20,
        "occluded_or_truncated_rows": occluded_or_truncated,
        "occluded_or_truncated_fraction": occluded_or_truncated / total if total else 0.0,
        "required_occluded_or_truncated_fraction": 0.15,
    }
    quota_pass = (
        quota["low_light_fraction"] >= quota["required_low_light_fraction"]
        and quota["small_target_fraction"] >= quota["required_small_target_fraction"]
        and quota["occluded_or_truncated_fraction"] >= quota["required_occluded_or_truncated_fraction"]
    )

    additions = [
        "official_vehicle_class", "official_occluded", "official_truncated",
        "source_sample_id", "stage53_dhash64", "stage53_image_sha256",
    ]
    output_fields = fields + [field for field in additions if field not in fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(retained)

    report = {
        "schema_version": "stage53-openimages-color-domain-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if quota_pass else "label_quality_pass_complex_scene_quota_fail_auxiliary_only",
        "input_manifest": str(args.manifest.resolve()),
        "input_manifest_sha256": sha256(args.manifest),
        "teacher_report": str(args.teacher_report.resolve()),
        "teacher_report_sha256": sha256(args.teacher_report),
        "source_card": str(args.source_card.resolve()),
        "source_card_sha256": sha256(args.source_card),
        "teacher_accepted_rows": len(selected),
        "official_target_vehicle_rows_before_dedup": len(domain_rows),
        "domain_rejections": dict(sorted(domain_rejections.items())),
        "dhash_distance": args.dhash_distance,
        "near_duplicate_rows_removed": removed_near_duplicates,
        "component_size_counts": {str(key): value for key, value in sorted(component_sizes.items())},
        "retained_rows": total,
        "retained_color_counts": dict(sorted(counts.items())),
        "retained_color_official_class_counts": {
            f"{color}|{vehicle_class}": value
            for (color, vehicle_class), value in sorted(color_class.items())
        },
        "complex_scene_quota": quota,
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "policy": {
            "official_vehicle_classes_allowed": sorted(TARGET_CLASSES),
            "motorcycles_and_unmapped_rows_excluded": True,
            "one_representative_per_same_color_dhash_component": True,
            "train_rows_only": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": (
            "auxiliary train-only data; do not train alone or promote until new real low-light, "
            "small-target and occlusion evidence closes the failed quotas"
        ),
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
