#!/usr/bin/env python3
"""Merge axle-clean supervision with isolated MIO road-domain axle proxies."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

MIO_MAPPING = {
    ("pickup", "pickup_truck"): ("light_truck", 1.0, "explicit_pickup_two_axle_proxy"),
    ("heavy_truck", "articulated_truck"): (
        "heavy_truck",
        1.28,
        "explicit_articulated_heavy_truck",
    ),
}
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "baseline-preview-36-48", "36-48s")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def content_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_sha256") or row.get("sha256") or "").strip().lower()


def perceptual_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_dhash64") or row.get("dhash64") or "").strip().lower()


def group_key(row: dict[str, str]) -> str:
    for field in ("track_group", "video_id", "source_group", "camera_id"):
        value = str(row.get(field, "")).strip()
        if value and value.lower() != "unknown":
            return f"{field}:{value}"
    return ""


class HammingBKTree:
    def __init__(self) -> None:
        self.root: list | None = None

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = [value, {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                return
            child = node[1].get(distance)
            if child is None:
                node[1][distance] = [value, {}]
                return
            node = child

    def contains_within(self, value: int, radius: int) -> bool:
        if self.root is None:
            return False
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                return True
            low, high = distance - radius, distance + radius
            stack.extend(child for edge, child in node[1].items() if low <= edge <= high)
        return False


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def frozen_marker_count(rows: list[dict[str, str]]) -> int:
    return sum(
        any(marker in " ".join(row.values()).lower() for marker in FROZEN_MARKERS)
        for row in rows
    )


def map_mio_row(row: dict[str, str]) -> tuple[dict[str, str] | None, str]:
    if row.get("split") != "train":
        return None, "non_train"
    if row.get("source_dataset") != "MIO-TCD-Classification-2017":
        return None, "unexpected_source"
    if row.get("source_license") != "CC-BY-NC-SA-4.0":
        return None, "license_mismatch"
    if row.get("review_status") != "approved_research_only":
        return None, "review_status_mismatch"
    mapping = MIO_MAPPING.get((row.get("body_type", ""), row.get("source_label", "")))
    if mapping is None:
        return None, "not_axle_eligible"
    target, weight, contract = mapping
    output = dict(row)
    output.update(
        {
            "body_type": target,
            "body_type_supervised": "true",
            "color": "unknown",
            "color_supervised": "false",
            "sample_weight": f"{weight:.6f}",
            "research_only": "true",
            "deployment_eligible": "false",
            "formal_train_eligible": "false",
            "stage136_origin": "stage89_mio_train_only",
            "stage136_original_body_type": row.get("body_type", ""),
            "stage136_truth_contract": contract,
            "stage136_license_scope": "research-only_non-deployable",
        }
    )
    return output, "accepted"


def post_split_leaks(rows: list[dict[str, str]], radius: int) -> dict[str, int]:
    validation = [row for row in rows if row.get("split") == "validation"]
    train = [row for row in rows if row.get("split") == "train"]
    train_exact = {content_hash(row) for row in train if content_hash(row)}
    train_groups = {group_key(row) for row in train if group_key(row)}
    tree = HammingBKTree()
    for row in train:
        value = perceptual_hash(row)
        if len(value) == 16:
            tree.add(int(value, 16))
    leaks = {"exact": 0, "near": 0, "group": 0}
    for row in validation:
        digest = content_hash(row)
        if digest and digest in train_exact:
            leaks["exact"] += 1
        value = perceptual_hash(row)
        if len(value) == 16 and tree.contains_within(int(value, 16), radius):
            leaks["near"] += 1
        group = group_key(row)
        if group and group in train_groups:
            leaks["group"] += 1
    return leaks


def merge_rows(
    base_rows: list[dict[str, str]], mio_rows: list[dict[str, str]], radius: int = 4
) -> tuple[list[dict[str, str]], dict[str, object]]:
    if any(row.get("split") == "test" for row in [*base_rows, *mio_rows]):
        raise RuntimeError("Stage136 inputs must not contain test rows")
    base_exact = {content_hash(row) for row in base_rows if content_hash(row)}
    base_groups = {group_key(row) for row in base_rows if group_key(row)}
    base_tree = HammingBKTree()
    for row in base_rows:
        value = perceptual_hash(row)
        if len(value) == 16:
            base_tree.add(int(value, 16))
    accepted: list[dict[str, str]] = []
    accepted_exact: set[str] = set()
    accepted_groups: set[str] = set()
    accepted_tree = HammingBKTree()
    counters: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    for row in mio_rows:
        mapped, reason = map_mio_row(row)
        if mapped is None:
            counters[reason] += 1
            continue
        digest = content_hash(mapped)
        if not digest or digest in base_exact or digest in accepted_exact:
            counters["rejected_exact_or_missing_hash"] += 1
            continue
        group = group_key(mapped)
        if group and (group in base_groups or group in accepted_groups):
            counters["rejected_group_overlap"] += 1
            continue
        value = perceptual_hash(mapped)
        if len(value) != 16:
            counters["rejected_missing_dhash"] += 1
            continue
        value_int = int(value, 16)
        if base_tree.contains_within(value_int, radius) or accepted_tree.contains_within(
            value_int, radius
        ):
            counters["rejected_near_duplicate"] += 1
            continue
        accepted.append(mapped)
        accepted_exact.add(digest)
        if group:
            accepted_groups.add(group)
        accepted_tree.add(value_int)
        counters["accepted"] += 1
        class_counts[mapped["body_type"]] += 1
    output = [dict(row) for row in base_rows] + accepted
    masses: Counter[str] = Counter()
    for row in output:
        if row.get("split") == "train" and row.get("body_type") in {
            "light_truck",
            "heavy_truck",
        }:
            masses[row["body_type"]] += float(row.get("sample_weight") or 1.0)
    return output, {
        "base_rows": len(base_rows),
        "accepted_mio_rows": len(accepted),
        "output_rows": len(output),
        "accepted_class_counts": dict(sorted(class_counts.items())),
        "weighted_class_mass": dict(sorted(masses.items())),
        "counters": dict(sorted(counters.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--expected-base-manifest-sha256", required=True)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--expected-base-report-sha256", required=True)
    parser.add_argument("--mio-manifest", type=Path, required=True)
    parser.add_argument("--expected-mio-manifest-sha256", required=True)
    parser.add_argument("--mio-report", type=Path, required=True)
    parser.add_argument("--expected-mio-report-sha256", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage136 evidence")
    for path, expected, role in (
        (args.base_manifest, args.expected_base_manifest_sha256, "base manifest"),
        (args.base_report, args.expected_base_report_sha256, "base report"),
        (args.mio_manifest, args.expected_mio_manifest_sha256, "MIO manifest"),
        (args.mio_report, args.expected_mio_report_sha256, "MIO report"),
    ):
        actual = sha256(path)
        if actual.lower() != expected.lower():
            raise RuntimeError(f"{role} SHA256 mismatch: expected={expected} actual={actual}")
    base_report = json.loads(args.base_report.read_text(encoding="utf-8"))
    mio_report = json.loads(args.mio_report.read_text(encoding="utf-8"))
    if base_report.get("status") != "pass" or mio_report.get("status") != "pass":
        raise RuntimeError("an input report did not pass")
    if base_report.get("policy", {}).get("test_accessed") is not False:
        raise RuntimeError("base report lost test isolation")
    mio_policy = mio_report.get("policy", {})
    if mio_policy.get("official_test_image_payloads_read") != 0:
        raise RuntimeError("MIO official test payload isolation failed")
    if mio_policy.get("frozen_video_used") is not False:
        raise RuntimeError("MIO report lost frozen-video isolation")
    base_fields, base_rows = read_rows(args.base_manifest)
    mio_fields, mio_rows = read_rows(args.mio_manifest)
    if frozen_marker_count([*base_rows, *mio_rows]):
        raise RuntimeError("frozen marker found in Stage136 inputs")
    output, summary = merge_rows(base_rows, mio_rows, args.near_duplicate_hamming)
    expected_counts = {"heavy_truck": 2259, "light_truck": 2300}
    if summary["accepted_class_counts"] != expected_counts:
        raise RuntimeError(
            f"unexpected accepted MIO class counts: {summary['accepted_class_counts']}"
        )
    masses = list(summary["weighted_class_mass"].values())
    if len(masses) != 2 or max(masses) / min(masses) > 1.03:
        raise RuntimeError("Stage136 weighted class mass imbalance exceeds 3 percent")
    leaks = post_split_leaks(output, args.near_duplicate_hamming)
    if any(leaks.values()):
        raise RuntimeError(f"Stage136 cross-split leak: {leaks}")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=False)
    extra_fields = (
        "sample_weight",
        "research_only",
        "deployment_eligible",
        "formal_train_eligible",
        "stage136_origin",
        "stage136_original_body_type",
        "stage136_truth_contract",
        "stage136_license_scope",
    )
    fields = list(dict.fromkeys([*base_fields, *mio_fields, *extra_fields]))
    with args.output_manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)
    report = {
        "schema_version": "stage136-mio-road-domain-axle-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "inputs": {
            "base_manifest_sha256": sha256(args.base_manifest),
            "base_report_sha256": sha256(args.base_report),
            "mio_manifest_sha256": sha256(args.mio_manifest),
            "mio_report_sha256": sha256(args.mio_report),
        },
        "output": {
            **summary,
            "manifest": str(args.output_manifest.resolve()),
            "manifest_sha256": sha256(args.output_manifest),
        },
        "mapping": {
            "pickup_truck": "light_truck under explicit two-axle proxy",
            "articulated_truck": "heavy_truck",
        },
        "integrity": {
            "near_duplicate_hamming": args.near_duplicate_hamming,
            "post_split_leaks": leaks,
            "frozen_rows": 0,
        },
        "policy": {
            "research_only": True,
            "deployment_eligible": False,
            "official_mio_train_only": True,
            "official_test_image_payloads_read": 0,
            "validation_rows_preserved_without_relabeling": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for path in (args.output_manifest, args.output_report):
        Path(str(path) + ".sha256").write_text(
            f"{sha256(path)}  {path.name}\n", encoding="utf-8"
        )
    print(json.dumps({"status": "pass", **summary, "leaks": leaks}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
