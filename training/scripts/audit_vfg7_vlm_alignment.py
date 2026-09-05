#!/usr/bin/env python3
"""Audit whether VFG-7 structured attributes align unambiguously to YOLO boxes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def type_class(value: str) -> int | None:
    text = value.strip().lower()
    rules = (
        (6, ("公交", "大巴", "客车", "巴士", "校车")),
        (5, ("半挂", "牵引", "挂车", "轿运")),
        (4, ("搅拌", "罐体", "油罐", "水泥罐", "混凝土", "罐车")),
        (3, ("自卸", "渣土")),
        (2, ("重型厢", "重型箱", "集装箱运输", "重型卡", "重型货", "重型栏板")),
        (1, ("中型货", "轻卡", "蓝牌货", "轻型厢", "轻型箱", "厢式货", "箱式货", "平板货")),
        (0, ("轿车", "suv", "mpv", "面包", "皮卡", "微型货", "小客", "私家", "出租")),
    )
    for class_id, terms in rules:
        if any(term in text for term in terms):
            return class_id
    return None


def body_label(value: str) -> str:
    text = value.strip().lower()
    rules = (
        ("sedan", ("轿车", "私家", "出租")), ("suv", ("suv",)), ("mpv", ("mpv",)),
        ("van", ("面包",)), ("pickup", ("皮卡",)), ("bus", ("公交", "大巴", "铰接式客")),
        ("light_truck", ("中型货", "轻卡", "蓝牌货", "轻型厢", "轻型箱", "微型货", "平板货")),
        ("heavy_truck", ("重型", "半挂", "牵引", "挂车", "轿运", "自卸", "渣土", "搅拌", "罐车", "罐体", "混凝土")),
    )
    for label, terms in rules:
        if any(term in text for term in terms):
            return label
    return "unknown"


def mapped_color(value: str) -> str:
    text = value.strip().lower()
    if not text or any(term in text for term in ("相间", "拼色", "多色", "渐变", "不明", "未知")):
        return "unknown"
    categories = []
    rules = (
        ("black", ("黑",)), ("white", ("白",)), ("silver_gray", ("灰", "银")),
        ("blue", ("蓝", "靛")), ("red", ("红", "酒红", "玫红")),
        ("brown_beige", ("棕", "褐", "米", "卡其")), ("green", ("绿",)),
        ("yellow_orange", ("黄", "橙", "金")),
    )
    for label, terms in rules:
        if any(term in text for term in terms):
            categories.append(label)
    categories = sorted(set(categories))
    return categories[0] if len(categories) == 1 else "unknown"


def parse_labels(path: Path) -> list[int]:
    classes = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if parts:
            classes.append(int(parts[0]))
    return classes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    vlm_path = args.metadata_root / "vlm_annotations.json"
    annotations = json.loads(vlm_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in annotations:
        grouped[str(item.get("image", ""))].append(item)
    labels = {}
    for split in ("train", "val"):
        for path in (args.metadata_root / "labels" / split).glob("*.txt"):
            labels[path.stem] = (split, path)
    counters = Counter()
    unknown_types = Counter()
    accepted_colors = Counter()
    raw_colors = Counter()
    raw_types = Counter()
    track_images: dict[str, set[str]] = defaultdict(set)
    track_types: dict[str, list[str]] = defaultdict(list)
    track_colors: dict[str, list[str]] = defaultdict(list)
    safely_aligned_tracks: dict[str, list[dict]] = defaultdict(list)
    alignment_examples = []
    for image, items in sorted(grouped.items()):
        stem = Path(image).stem
        entry = labels.get(stem)
        counters["json_images"] += 1
        if entry is None:
            counters["missing_label_image"] += 1
            continue
        split, label_path = entry
        counters[f"json_images_{split}"] += 1
        yolo = parse_labels(label_path)
        mapped = [type_class(str(item.get("vehicle_type", ""))) for item in items]
        counters["json_annotations_with_label_file"] += len(items)
        if len(yolo) == len(items):
            counters["equal_count_images"] += 1
            if all(value is not None for value in mapped):
                counters["fully_mapped_equal_count_images"] += 1
                if yolo == mapped:
                    counters["exact_order_match_images"] += 1
                elif len(alignment_examples) < 20:
                    alignment_examples.append({"image": image, "yolo": yolo, "vlm_mapped": mapped})
        else:
            counters["count_mismatch_images"] += 1
        yolo_by_class = Counter(yolo)
        items_by_class: dict[int, list[dict]] = defaultdict(list)
        for item, class_id in zip(items, mapped):
            if class_id is not None:
                items_by_class[class_id].append(item)
        for class_id in sorted(set(yolo_by_class) | set(items_by_class)):
            candidates = items_by_class.get(class_id, [])
            box_count = yolo_by_class.get(class_id, 0)
            if box_count <= 0 or box_count != len(candidates):
                counters["rejected_class_group_count_mismatch"] += max(box_count, len(candidates))
                continue
            body_values = [body_label(str(item.get("vehicle_type", ""))) for item in candidates]
            color_values = [mapped_color(str(item.get("color", ""))) for item in candidates]
            agreements = [float(item.get("agreement", 0.0) or 0.0) for item in candidates]
            body_safe = all(value != "unknown" for value in body_values) and len(set(body_values)) == 1 and min(agreements) >= 0.999
            color_safe = all(value != "unknown" for value in color_values) and len(set(color_values)) == 1 and min(agreements) >= 0.999
            if body_safe:
                counters["safely_aligned_body_boxes"] += box_count
                counters[f"safely_aligned_body_{body_values[0]}"] += box_count
            else:
                counters["rejected_ambiguous_body_boxes"] += box_count
            if color_safe:
                counters["safely_aligned_color_boxes"] += box_count
                counters[f"safely_aligned_color_{color_values[0]}"] += box_count
            else:
                counters["rejected_ambiguous_color_boxes"] += box_count
            if box_count == 1 and (body_safe or color_safe):
                counters["uniquely_aligned_track_observations"] += 1
                track = str(candidates[0].get("track_id", ""))
                if track:
                    safely_aligned_tracks[track].append({
                        "image": image, "split": split,
                        "body": body_values[0] if body_safe else "unknown",
                        "color": color_values[0] if color_safe else "unknown",
                        "class_id": class_id,
                    })
        for item in items:
            vehicle_type = str(item.get("vehicle_type", "")).strip()
            color = str(item.get("color", "")).strip()
            raw_types[vehicle_type] += 1
            raw_colors[color] += 1
            class_id = type_class(vehicle_type)
            if class_id is None:
                unknown_types[vehicle_type] += 1
            agreement = float(item.get("agreement", 0.0) or 0.0)
            color_label = mapped_color(color)
            if agreement >= 0.999 and color_label != "unknown":
                accepted_colors[color_label] += 1
            track = str(item.get("track_id", ""))
            if track:
                track_images[track].add(image)
                if agreement >= 0.999 and class_id is not None:
                    track_types[track].append(str(class_id))
                if agreement >= 0.999 and color_label != "unknown":
                    track_colors[track].append(color_label)
    fully_mapped = counters["fully_mapped_equal_count_images"]
    exact_rate = counters["exact_order_match_images"] / fully_mapped if fully_mapped else 0.0
    repeated_tracks = {key: value for key, value in track_images.items() if len(value) >= 3}
    def consistent_tracks(values: dict[str, list[str]]) -> int:
        return sum(bool(items) and len(set(items)) == 1 for key, items in values.items() if key in repeated_tracks)
    safe_repeated_tracks = {
        key: values for key, values in safely_aligned_tracks.items()
        if len({item["image"] for item in values}) >= 3
    }
    safe_color_tracks = sum(
        len({item["color"] for item in values if item["color"] != "unknown"}) == 1
        and sum(item["color"] != "unknown" for item in values) >= 3
        for values in safe_repeated_tracks.values()
    )
    status = (
        "pass_partial_fail_closed"
        if counters["safely_aligned_color_boxes"] >= 1000 and safe_color_tracks >= 100
        else "fail_closed_alignment_unproven"
    )
    report = {
        "schema_version": "vfg7-vlm-box-alignment-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "policy": {
            "frozen_video_used": False, "model_predictions_used": False,
            "minimum_agreement_for_attribute_truth": 1.0,
            "multicolor_policy": "unknown", "license": "CC BY-NC 4.0",
            "training_eligible": False,
            "direct_order_alignment_acceptance_rate": 0.995,
            "safe_partial_rule": "per-image YOLO class counts must match; singleton groups bind uniquely; multi-box groups require identical agreement=1.0 attributes",
        },
        "metadata_root": str(args.metadata_root),
        "vlm_annotations_sha256": sha256(vlm_path),
        "label_files": len(labels),
        "annotations": len(annotations),
        "counters": dict(sorted(counters.items())),
        "exact_order_match_rate": exact_rate,
        "unknown_vehicle_types": dict(unknown_types.most_common()),
        "raw_vehicle_type_counts": dict(raw_types.most_common()),
        "raw_color_counts": dict(raw_colors.most_common()),
        "accepted_single_color_counts": dict(sorted(accepted_colors.items())),
        "tracks": len(track_images),
        "tracks_with_at_least_3_images": len(repeated_tracks),
        "repeated_tracks_with_consistent_type_truth": consistent_tracks(track_types),
        "repeated_tracks_with_consistent_color_truth": consistent_tracks(track_colors),
        "safely_aligned_tracks": len(safely_aligned_tracks),
        "safely_aligned_tracks_with_at_least_3_images": len(safe_repeated_tracks),
        "safely_aligned_consistent_color_tracks_with_at_least_3_images": safe_color_tracks,
        "alignment_mismatch_examples": alignment_examples,
        "decision": (
            "download images only for the safe partial subset; reject every ambiguous class group"
            if status == "pass_partial_fail_closed" else
            "do not download images or use structured attributes; safe partial evidence is insufficient"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "pass_partial_fail_closed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
