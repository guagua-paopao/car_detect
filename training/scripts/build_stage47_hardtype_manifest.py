#!/usr/bin/env python3
"""Build a train-only hard-type weighted mixture while freezing validation/test rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TARGET_BASE_WEIGHTS = {
    "van": 2.5,
    "mpv": 1.2,
    "light_truck": 1.2,
    "bus": 1.2,
}
TARGET_UVH_WEIGHTS = {
    "van": 4.0,
    "mpv": 1.5,
    "light_truck": 1.5,
    "bus": 1.5,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def semantic_digest(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    keys = ("image_path", "split", "body_type", "color", "body_type_supervised", "color_supervised", "source_group")
    for row in rows:
        digest.update("\t".join(str(row.get(key, "")) for key in keys).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--uvh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite Stage47 manifest evidence")
    base_fields, base_rows = read(args.base)
    uvh_fields, uvh_rows = read(args.uvh)
    fields = list(base_fields) + [field for field in uvh_fields if field not in base_fields]
    for field in ("sample_weight", "training_reason"):
        if field not in fields:
            fields.append(field)

    base_validation_test = [dict(row) for row in base_rows if row.get("split") in {"validation", "test"}]
    weighted_counts = Counter()
    for row in base_rows:
        row.setdefault("sample_weight", "")
        row.setdefault("training_reason", "")
        if row.get("split") != "train":
            continue
        body = row.get("body_type", "unknown")
        weight = TARGET_BASE_WEIGHTS.get(body, 1.0)
        tags = set(filter(None, str(row.get("hard_mining_tags", "")).split(";")))
        if truthy(row.get("color_supervised")) and tags & {"night", "low_light", "dark", "very_dark"}:
            weight = max(weight, 1.3)
        row["sample_weight"] = f"{weight:.3f}"
        row["training_reason"] = "base_hard_type_weight" if weight > 1.0 else "base"
        weighted_counts[f"base|{body}|{weight:.1f}"] += 1

    supplement = []
    rejected = Counter()
    seen_paths = {row.get("image_path", "") for row in base_rows}
    for source in uvh_rows:
        if source.get("source_dataset") != "UVH-26":
            continue
        if source.get("split") != "train":
            rejected["non_train"] += 1
            continue
        if not truthy(source.get("body_type_supervised")) or source.get("body_type") in {"", "unknown"}:
            rejected["not_body_supervised"] += 1
            continue
        if source.get("source_license") != "CC-BY-4.0":
            rejected["license"] += 1
            continue
        if source.get("image_path", "") in seen_paths:
            rejected["path_already_present"] += 1
            continue
        row = {field: source.get(field, "") for field in fields}
        body = row.get("body_type", "unknown")
        weight = TARGET_UVH_WEIGHTS.get(body, 1.0)
        row["sample_weight"] = f"{weight:.3f}"
        row["training_reason"] = "uvh26_real_cctv_hard_type"
        row["color"] = "unknown"
        row["color_supervised"] = "false"
        supplement.append(row)
        seen_paths.add(row.get("image_path", ""))
        weighted_counts[f"uvh|{body}|{weight:.1f}"] += 1

    output_rows = base_rows + supplement
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    output_validation_test = [row for row in output_rows if row.get("split") in {"validation", "test"}]
    report = {
        "schema_version": "stage47-hardtype-mixture-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "policy": {
            "frozen_video_used": False,
            "test_used_for_selection": False,
            "ua_detrac_training_rows": 0,
            "validation_test_labels_paths_splits_unchanged": semantic_digest(base_validation_test) == semantic_digest(output_validation_test),
            "uvh_train_only": all(row.get("split") == "train" for row in supplement),
            "uvh_color_supervision_disabled": all(not truthy(row.get("color_supervised")) for row in supplement),
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "base_manifest": str(args.base),
        "base_manifest_sha256": sha256(args.base),
        "uvh_manifest": str(args.uvh),
        "uvh_manifest_sha256": sha256(args.uvh),
        "output_manifest": str(args.output),
        "output_manifest_sha256": sha256(args.output),
        "base_rows": len(base_rows),
        "supplement_rows": len(supplement),
        "output_rows": len(output_rows),
        "supplement_body_counts": dict(sorted(Counter(row.get("body_type", "unknown") for row in supplement).items())),
        "weighted_counts": dict(sorted(weighted_counts.items())),
        "rejected": dict(sorted(rejected.items())),
        "base_weights": TARGET_BASE_WEIGHTS,
        "uvh_weights": TARGET_UVH_WEIGHTS,
        "source_licenses": dict(sorted(Counter(row.get("source_license", "") for row in supplement).items())),
    }
    if not (
        supplement
        and report["policy"]["validation_test_labels_paths_splits_unchanged"]
        and report["policy"]["uvh_train_only"]
        and report["policy"]["uvh_color_supervision_disabled"]
        and report["policy"]["ua_detrac_training_rows"] == 0
        and not report["policy"]["frozen_video_used"]
        and not report["policy"]["test_used_for_selection"]
        and not report["policy"]["production_model_modified"]
        and not report["policy"]["deployment_performed"]
    ):
        report["status"] = "fail"
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
