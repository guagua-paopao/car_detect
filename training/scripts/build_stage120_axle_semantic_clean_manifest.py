#!/usr/bin/env python3
"""Build an axle-semantics-clean light/heavy-truck specialist manifest."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

FINE_LABELS = {"light_truck", "heavy_truck"}
INATRC_LABELS = {
    "two_axle_truck": "light_truck",
    "three_axle_truck": "heavy_truck",
    "four_axle_truck": "heavy_truck",
    "five_plus_axle_truck": "heavy_truck",
}
SAMPLE_WEIGHTS = {
    "inatrc_light": 1.0,
    "inatrc_heavy": 2.8,
    "bmd_lcv_light": 0.15,
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
    return str(
        row.get("crop_sha256")
        or row.get("sha256")
        or row.get("image_sha256")
        or row.get("source_image_sha256")
        or ""
    ).strip().lower()


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

    def add(self, value: int, row: dict[str, str]) -> None:
        if self.root is None:
            self.root = [value, [row], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(row)
                return
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = [value, [row], {}]
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
                return node[1][0]
            low, high = distance - radius, distance + radius
            stack.extend(child for edge, child in node[2].items() if low <= edge <= high)
        return None


def post_split_leaks(rows: list[dict[str, str]], radius: int) -> dict[str, int]:
    validation = [row for row in rows if row["split"] == "validation"]
    train = [row for row in rows if row["split"] == "train"]
    train_exact = {content_hash(row) for row in train if content_hash(row)}
    train_tree = HammingBKTree()
    train_groups = {group_key(row) for row in train if group_key(row)}
    for row in train:
        value = perceptual_hash(row)
        if len(value) == 16:
            train_tree.add(int(value, 16), row)
    exact = near = groups = 0
    for row in validation:
        digest = content_hash(row)
        if digest and digest in train_exact:
            exact += 1
        value = perceptual_hash(row)
        if len(value) == 16 and train_tree.find(int(value, 16), radius) is not None:
            near += 1
        group = group_key(row)
        if group and group in train_groups:
            groups += 1
    return {"exact": exact, "near": near, "group": groups}


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def restricted(row: dict[str, str]) -> bool:
    return "NC" in str(row.get("source_license", "")).upper() or truthy(
        row.get("research_only")
    )


def semantic_role(row: dict[str, str]) -> tuple[str | None, str]:
    if row.get("split") != "train":
        return None, "not_train"
    if restricted(row):
        return None, "restricted_or_research_only"
    source = row.get("source_dataset", "")
    label = row.get("source_label", "")
    body = row.get("body_type", "")
    if source == "InaTRC-v1" and label in INATRC_LABELS:
        expected = INATRC_LABELS[label]
        if body != expected:
            return None, "inatrc_axle_truth_conflict"
        return ("inatrc_light" if expected == "light_truck" else "inatrc_heavy"), "accepted"
    if (
        source == "BMD-45-RAW"
        and body == "light_truck"
        and row.get("annotation_source") == "BMD-45 COCO category mapping"
        and row.get("review_model") == "source_category_mapping_v23"
    ):
        # TYPE_MAP v23 maps only the explicit BMD LCV category to light_truck.
        return "bmd_lcv_light", "accepted"
    if source in {"BMD-45", "BMD-45-RAW"} and body == "heavy_truck":
        return None, "generic_bmd_truck_not_axle_verified"
    if source in {"Open-Images-V7", "UVH-26"}:
        return None, "coarse_vehicle_class_not_axle_verified"
    return None, "no_explicit_axle_or_lcv_contract"


def build(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, object]]:
    if any(row.get("split") == "test" for row in rows):
        raise RuntimeError("Stage120 input must not contain test rows")
    validation = [dict(row) for row in rows if row.get("split") == "validation"]
    output_train: list[dict[str, str]] = []
    excluded = Counter()
    role_counts = Counter()
    class_counts = Counter()
    weighted_mass = Counter()
    for row in rows:
        if row.get("split") != "train":
            continue
        role, reason = semantic_role(row)
        if role is None:
            excluded[reason] += 1
            continue
        output = dict(row)
        weight = SAMPLE_WEIGHTS[role]
        output["stage120_previous_sample_weight"] = row.get("sample_weight", "")
        output["sample_weight"] = f"{weight:.6f}"
        output["stage120_semantic_role"] = role
        output["stage120_truth_contract"] = (
            "explicit_bmd_lcv_category"
            if role == "bmd_lcv_light"
            else "explicit_inatrc_axle_count"
        )
        output["stage120_generic_truck_supervision"] = "false"
        output_train.append(output)
        role_counts[role] += 1
        class_counts[output["body_type"]] += 1
        weighted_mass[output["body_type"]] += weight
    output = [*validation, *output_train]
    return output, {
        "input_rows": len(rows),
        "output_rows": len(output),
        "validation_rows_preserved": len(validation),
        "train_rows": len(output_train),
        "train_counts": dict(sorted(class_counts.items())),
        "semantic_role_counts": dict(sorted(role_counts.items())),
        "weighted_class_mass": dict(sorted(weighted_mass.items())),
        "excluded": dict(sorted(excluded.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--expected-input-manifest-sha256", required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--expected-input-report-sha256", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage120 evidence")
    for path, expected, role in (
        (args.input_manifest, args.expected_input_manifest_sha256, "manifest"),
        (args.input_report, args.expected_input_report_sha256, "report"),
    ):
        actual = sha256(path)
        if actual.lower() != expected.lower():
            raise RuntimeError(f"{role} SHA256 mismatch: expected={expected} actual={actual}")
    input_report = json.loads(args.input_report.read_text(encoding="utf-8"))
    if input_report.get("status") != "pass":
        raise RuntimeError("Stage113 input report did not pass")
    policy = input_report.get("policy", {})
    if policy.get("test_accessed") is not False or policy.get("frozen_video_used") is not False:
        raise RuntimeError("Stage113 input lost test/frozen isolation")
    fields, rows = read_rows(args.input_manifest)
    frozen_rows = sum(
        any(marker in " ".join(row.values()).lower() for marker in FROZEN_MARKERS)
        for row in rows
    )
    if frozen_rows:
        raise RuntimeError(f"frozen marker rows found: {frozen_rows}")
    output, summary = build(rows)
    if summary["validation_rows_preserved"] != input_report["output"]["validation_rows_preserved"]:
        raise RuntimeError("validation rows were not preserved exactly")
    if set(summary["train_counts"]) != FINE_LABELS or min(summary["train_counts"].values()) < 2500:
        raise RuntimeError("both axle-clean train classes require at least 2500 rows")
    masses = list(summary["weighted_class_mass"].values())
    if max(masses) / min(masses) > 1.15:
        raise RuntimeError("weighted class mass imbalance exceeds 15%")
    leaks = post_split_leaks(output, 4)
    if any(leaks.values()):
        raise RuntimeError(f"Stage120 cross-split leak: {leaks}")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=False)
    extra = (
        "stage120_previous_sample_weight",
        "stage120_semantic_role",
        "stage120_truth_contract",
        "stage120_generic_truck_supervision",
    )
    output_fields = list(dict.fromkeys([*fields, *extra]))
    with args.output_manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)
    report = {
        "schema_version": "stage120-axle-semantic-clean-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "inputs": {
            "manifest_sha256": sha256(args.input_manifest),
            "report_sha256": sha256(args.input_report),
        },
        "output": {
            **summary,
            "manifest": str(args.output_manifest.resolve()),
            "manifest_sha256": sha256(args.output_manifest),
        },
        "sample_weights": SAMPLE_WEIGHTS,
        "diagnosis": {
            "historical_mapping": "BMD category Truck was mapped directly to heavy_truck",
            "target_contract": "two_axle=light_truck; three_plus_axle=heavy_truck",
            "repair": "exclude generic Truck supervision; retain only explicit BMD LCV and explicit InaTRC axle labels",
        },
        "integrity": {"near_duplicate_hamming": 4, "post_split_leaks": leaks, "frozen_rows": 0},
        "policy": {
            "validation_rows_preserved_without_relabeling": True,
            "generic_truck_never_supervises_heavy_subtype": True,
            "uncertain_subtype_rows_excluded_not_coerced": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path in (args.output_manifest, args.output_report):
        Path(str(path) + ".sha256").write_text(
            f"{sha256(path)}  {path.name}\n", encoding="utf-8"
        )
    print(json.dumps({"status": "pass", **summary, "leaks": leaks}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
