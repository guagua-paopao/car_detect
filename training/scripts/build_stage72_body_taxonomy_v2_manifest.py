#!/usr/bin/env python3
"""Build the isolated Stage72 taxonomy-v2 body manifest fail closed.

Exact v1 body labels remain exact.  A generic/coarse truck annotation is not
fabricated into a subtype (or treated as exact generic output); it is retained
only as partial truck-family supervision.  The v2 decoder may later emit the
generic ``truck`` fallback when neither subtype is sufficiently confident.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48s", "36_48s", "frozen_video")
EXACT_BODY_TYPES = {
    "sedan", "suv", "mpv", "van", "pickup", "bus",
    "light_truck", "heavy_truck", "other",
}
REQUIRED_FIELDS = {
    "image_path", "split", "body_type", "body_type_supervised",
    "coarse_body_family", "source_dataset",
}
ADDED_FIELDS = (
    "stage72_body_truth_source",
    "stage72_taxonomy",
    "stage72_eligibility",
    "stage72_parent_manifest_sha256",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def row_group(row: dict[str, str]) -> str:
    for field in ("track_group", "video_id", "source_group", "source_frame_id"):
        value = str(row.get(field, "")).strip()
        if value:
            return value
    return ""


def content_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_sha256") or row.get("sha256") or "").strip().lower()


def perceptual_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_dhash64") or row.get("dhash64") or "").strip().lower()


class HammingBKTree:
    def __init__(self) -> None:
        self.root: list | None = None

    def add(self, value: int, row: dict[str, str]) -> None:
        if self.root is None:
            self.root = [value, row, {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = [value, row, {}]
                return
            node = child

    def find(self, value: int, radius: int) -> dict[str, str] | None:
        if self.root is None:
            return None
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                return node[1]
            low, high = distance - radius, distance + radius
            stack.extend(child for edge, child in node[2].items() if low <= edge <= high)
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--datasets-safety-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--near-duplicate-hamming", type=int, default=1)
    parser.add_argument("--minimum-train-rows", type=int, default=100000)
    parser.add_argument("--minimum-validation-rows", type=int, default=10000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage72B evidence")
    if not 0 <= args.near_duplicate_hamming <= 3:
        raise ValueError("near-duplicate Hamming radius must be in [0, 3]")
    if file_sha256(args.input_manifest).lower() != args.expected_input_sha256.lower():
        raise RuntimeError("Stage71 body manifest SHA256 mismatch")
    if file_sha256(args.labels).lower() != args.expected_labels_sha256.lower():
        raise RuntimeError("taxonomy-v2 labels SHA256 mismatch")
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if labels.get("labels_version") != "vehicle-labels-v2-offline-candidate":
        raise RuntimeError("Stage72B requires the isolated offline v2 taxonomy")
    if set(labels.get("body_types", [])) != EXACT_BODY_TYPES | {"truck", "unknown"}:
        raise RuntimeError("unexpected taxonomy-v2 body label set")

    with args.input_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        missing = sorted(REQUIRED_FIELDS - set(fields))
        if missing:
            raise RuntimeError(f"input manifest missing required fields: {missing}")
        source_rows = list(reader)

    safety_root = args.datasets_safety_root.resolve()
    failures: list[str] = []
    counters: Counter[str] = Counter()
    output_rows: list[dict[str, str]] = []
    split_groups: dict[str, set[str]] = defaultdict(set)
    exact_by_split: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    phash_trees: dict[str, HammingBKTree] = defaultdict(HammingBKTree)

    for line, source in enumerate(source_rows, start=2):
        searchable = " ".join(str(value).lower() for value in source.values())
        marker = next((item for item in FROZEN_MARKERS if item in searchable), None)
        if marker:
            failures.append(f"line {line}: frozen marker {marker}")
            continue
        split = str(source.get("split", "")).strip().lower()
        if split == "test":
            failures.append(f"line {line}: test row is forbidden")
            continue
        if split not in {"train", "validation"}:
            failures.append(f"line {line}: unsupported split {split!r}")
            continue
        group = row_group(source)
        if not group:
            failures.append(f"line {line}: no track/video/source group")
            continue
        split_groups[split].add(group)

        image = Path(str(source.get("image_path", ""))).expanduser()
        resolved = image.resolve() if image.is_absolute() else (args.input_manifest.parent / image).resolve()
        try:
            resolved.relative_to(safety_root)
        except ValueError:
            failures.append(f"line {line}: image escapes datasets safety root")
            continue
        if not resolved.is_file():
            failures.append(f"line {line}: image is missing")
            continue

        row = dict(source)
        current = str(row.get("body_type", "")).strip().lower()
        coarse = str(row.get("coarse_body_family", "")).strip().lower()
        supervised = truthy(row.get("body_type_supervised"))
        if coarse == "truck" or (supervised and current == "truck"):
            row["body_type"] = "unknown"
            row["body_type_supervised"] = "false"
            row["coarse_body_family"] = "truck"
            row["stage72_body_truth_source"] = "official_coarse_truck_partial_family"
            counters[f"{split}:partial_truck_family"] += 1
        elif supervised and current in EXACT_BODY_TYPES:
            row["body_type"] = current
            row["body_type_supervised"] = "true"
            row["coarse_body_family"] = ""
            row["stage72_body_truth_source"] = "retained_exact_v1_truth"
            counters[f"{split}:exact:{current}"] += 1
        elif not supervised or current in {"", "unknown"}:
            row["body_type"] = "unknown"
            row["body_type_supervised"] = "false"
            row["coarse_body_family"] = coarse if coarse in {"car", "truck"} else ""
            row["stage72_body_truth_source"] = "retained_unknown_or_partial_truth"
            counters[f"{split}:unknown_or_other_partial"] += 1
        else:
            failures.append(f"line {line}: unrecognized supervised body label {current!r}")
            continue
        row["image_path"] = str(resolved)
        row["stage72_taxonomy"] = "vehicle-labels-v2-offline-candidate"
        row["stage72_eligibility"] = "research-only_non-deployable"
        row["stage72_parent_manifest_sha256"] = args.expected_input_sha256.lower()
        output_rows.append(row)

        digest = content_hash(row)
        if digest:
            exact_by_split[split][digest] = row
        phash = perceptual_hash(row)
        if len(phash) == 16:
            try:
                phash_trees[split].add(int(phash, 16), row)
            except ValueError:
                failures.append(f"line {line}: invalid perceptual hash")

    group_leaks = sorted(split_groups["train"] & split_groups["validation"])
    if group_leaks:
        failures.append(f"train/validation group leakage: {len(group_leaks)}")
    exact_leaks = sorted(set(exact_by_split["train"]) & set(exact_by_split["validation"]))
    if exact_leaks:
        failures.append(f"train/validation exact-image leakage: {len(exact_leaks)}")
    near_leaks: list[dict[str, str]] = []
    for row in output_rows:
        if row["split"] != "validation":
            continue
        phash = perceptual_hash(row)
        if len(phash) != 16:
            continue
        match = phash_trees["train"].find(int(phash, 16), args.near_duplicate_hamming)
        if match is not None:
            near_leaks.append({"train": match["image_path"], "validation": row["image_path"]})
            if len(near_leaks) >= 100:
                break
    if near_leaks:
        failures.append(f"train/validation perceptual near leakage: {len(near_leaks)}+")

    splits = Counter(row["split"] for row in output_rows)
    if splits["train"] < args.minimum_train_rows:
        failures.append(f"minimum train rows: {splits['train']} < {args.minimum_train_rows}")
    if splits["validation"] < args.minimum_validation_rows:
        failures.append(
            f"minimum validation rows: {splits['validation']} < {args.minimum_validation_rows}"
        )
    for label in sorted(EXACT_BODY_TYPES):
        if counters[f"train:exact:{label}"] == 0:
            failures.append(f"missing exact train class: {label}")
        if counters[f"validation:exact:{label}"] == 0:
            failures.append(f"missing exact validation class: {label}")
    if counters["train:partial_truck_family"] == 0:
        failures.append("missing train partial truck-family truth")

    status = "pass" if not failures else "fail"
    report = {
        "schema_version": "stage72-body-taxonomy-v2-manifest-v1",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "manifest": str(args.input_manifest.resolve()),
            "sha256": file_sha256(args.input_manifest),
            "rows": len(source_rows),
            "labels": str(args.labels.resolve()),
            "labels_sha256": file_sha256(args.labels),
        },
        "output": {
            "manifest": str(args.output_manifest.resolve()),
            "rows": len(output_rows),
            "splits": dict(sorted(splits.items())),
            "truth_counts": dict(sorted(counters.items())),
        },
        "integrity": {
            "group_leaks": len(group_leaks),
            "exact_image_leaks": len(exact_leaks),
            "near_duplicate_leaks": len(near_leaks),
            "near_duplicate_examples": near_leaks,
            "test_rows": sum("test row" in item for item in failures),
            "frozen_markers": sum("frozen marker" in item for item in failures),
        },
        "policy": {
            "exact_subtype_truth_preserved": True,
            "generic_truck_used_as_exact_truth": False,
            "coarse_truck_used_only_as_partial_family_truth": True,
            "generic_truck_is_conservative_decode_fallback": True,
            "test_rows_imported": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "eligibility": "research-only_non-deployable",
        },
        "failures": failures[:300],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        print(json.dumps({"status": status, "failures": len(failures)}, ensure_ascii=False))
        return 2
    output_fields = fields + [field for field in ADDED_FIELDS if field not in fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    report["output"]["sha256"] = file_sha256(args.output_manifest)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "rows": len(output_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
