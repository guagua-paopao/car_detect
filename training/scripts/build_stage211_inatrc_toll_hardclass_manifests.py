#!/usr/bin/env python3
"""Merge the passing Stage210 InaTRC component into Stage167 research manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TRUE = {"1", "true", "yes", "y"}
WEAK_PASSENGER = {"suv", "mpv", "other"}
SECONDARY_PASSENGER = {"sedan", "van"}
FORBIDDEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "stage148", "stage155")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--base-unlabeled", type=Path, required=True)
    parser.add_argument("--expected-base-unlabeled-sha256", required=True)
    parser.add_argument("--stage210-supervised", type=Path, required=True)
    parser.add_argument("--expected-stage210-supervised-sha256", required=True)
    parser.add_argument("--stage210-unlabeled", type=Path, required=True)
    parser.add_argument("--expected-stage210-unlabeled-sha256", required=True)
    parser.add_argument("--stage210-report", type=Path, required=True)
    parser.add_argument("--expected-stage210-report-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, role: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    actual = sha256_file(path)
    if actual != expected.lower():
        raise RuntimeError(f"{role} SHA256 mismatch: expected={expected} actual={actual}")
    return actual


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in TRUE


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def identity(row: dict[str, str]) -> str:
    for key in ("crop_sha256", "sha256", "source_sha256", "source_image_sha256"):
        value = str(row.get(key, "")).strip().lower()
        if len(value) == 64 and all(ch in "0123456789abcdef" for ch in value):
            return value
    raise RuntimeError(f"row has no exact identity: {row.get('image_path', '')}")


def row_group(row: dict[str, str]) -> str:
    return str(row.get("track_group") or row.get("video_id") or row.get("source_frame_id") or row.get("image_path") or "").strip()


def complex_row(row: dict[str, str]) -> bool:
    lighting = str(row.get("lighting", "")).strip().lower()
    size = str(row.get("vehicle_size", "")).strip().lower()
    occlusion = str(row.get("occlusion_level", "")).strip().lower()
    return (
        truthy(row.get("night"))
        or truthy(row.get("low_light"))
        or truthy(row.get("small_target"))
        or size == "small"
        or truthy(row.get("occluded"))
        or truthy(row.get("truncated"))
        or truthy(row.get("blur"))
        or lighting in {"night", "low_light", "low-light"}
        or occlusion not in {"", "none", "clear", "unknown", "0"}
    )


def weighted(value: str, multiplier: float) -> str:
    try:
        base = float(value or 1.0)
    except ValueError:
        base = 1.0
    return f"{max(0.10, min(10.0, base * multiplier)):.6f}"


def passenger_multiplier(row: dict[str, str]) -> float:
    if str(row.get("split", "")).strip().lower() != "train":
        return 1.0
    if not truthy(row.get("body_type_supervised")):
        return 1.0
    label = str(row.get("body_type", "")).strip().lower()
    hard = complex_row(row)
    if label in WEAK_PASSENGER:
        return 1.25 if hard else 1.10
    if label in SECONDARY_PASSENGER and hard:
        return 1.10
    return 1.0


def validation_signature(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        if str(row.get("split", "")).strip().lower() not in {"validation", "val"}:
            continue
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    inputs = (args.base_manifest, args.base_unlabeled, args.stage210_supervised, args.stage210_unlabeled, args.stage210_report)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    if any(marker in str(path).lower() for marker in FORBIDDEN_MARKERS for path in inputs):
        raise RuntimeError("forbidden frozen/test marker in input")
    hashes = {
        "base_manifest": require_hash(args.base_manifest, args.expected_base_sha256, "base manifest"),
        "base_unlabeled": require_hash(args.base_unlabeled, args.expected_base_unlabeled_sha256, "base unlabeled"),
        "stage210_supervised": require_hash(args.stage210_supervised, args.expected_stage210_supervised_sha256, "Stage210 supervised"),
        "stage210_unlabeled": require_hash(args.stage210_unlabeled, args.expected_stage210_unlabeled_sha256, "Stage210 unlabeled"),
        "stage210_report": require_hash(args.stage210_report, args.expected_stage210_report_sha256, "Stage210 report"),
    }
    stage210 = json.loads(args.stage210_report.read_text(encoding="utf-8"))
    if stage210.get("status") != "pass_component_ready_for_research_candidate_manifest":
        raise RuntimeError("Stage210 component gate did not pass")
    if not stage210.get("gate", {}).get("training_component_authorized"):
        raise RuntimeError("Stage210 training component is not authorized")
    policy = stage210.get("policy", {})
    if policy.get("test_accessed") or policy.get("frozen_video_used"):
        raise RuntimeError("Stage210 policy evidence is unsafe")

    base_fields, base_rows = read_csv(args.base_manifest)
    sup_fields, supplement_supervised = read_csv(args.stage210_supervised)
    unlabeled_fields, base_unlabeled = read_csv(args.base_unlabeled)
    new_unlabeled_fields, supplement_unlabeled = read_csv(args.stage210_unlabeled)

    if any(str(row.get("split", "")).strip().lower() not in {"train", "validation", "val"} for row in base_rows):
        raise RuntimeError("base manifest contains an unexpected split")
    if any(str(row.get("split", "")).strip().lower() != "train" for row in supplement_supervised + base_unlabeled + supplement_unlabeled):
        raise RuntimeError("supplement or unlabeled manifest contains a non-train row")

    base_validation_signature = validation_signature(base_rows)
    base_exact = {identity(row) for row in base_rows}
    base_unlabeled_exact = {identity(row) for row in base_unlabeled}
    if len(base_exact) != len(base_rows):
        raise RuntimeError("base manifest contains exact duplicate identities")
    if len(base_unlabeled_exact) != len(base_unlabeled):
        raise RuntimeError("base unlabeled manifest contains exact duplicate identities")
    baseline_supervised_unlabeled_overlap = base_exact & base_unlabeled_exact

    counters = Counter()
    combined_rows: list[dict[str, str]] = []
    for row in base_rows:
        copied = dict(row)
        if str(copied.get("split", "")).strip().lower() == "train":
            multiplier = passenger_multiplier(copied)
            copied["stage211_passenger_multiplier"] = f"{multiplier:.6f}"
            copied["stage211_origin"] = "stage167_base"
            if multiplier != 1.0:
                copied["sample_weight"] = weighted(copied.get("sample_weight", ""), multiplier)
                counters[f"passenger_reweighted:{copied.get('body_type', 'unknown')}"] += 1
                counters["passenger_reweighted_total"] += 1
        combined_rows.append(copied)

    supplement_exact: set[str] = set()
    for row in supplement_supervised:
        digest = identity(row)
        if digest in base_exact or digest in supplement_exact:
            raise RuntimeError("Stage210 supervised exact overlap drift")
        supplement_exact.add(digest)
        copied = dict(row)
        if not truthy(copied.get("body_type_supervised")) or copied.get("body_type") not in {"truck", "heavy_truck"}:
            raise RuntimeError("Stage210 supervised semantic drift")
        copied["stage211_source_coarse_body_family"] = copied.get("coarse_body_family", "")
        copied["coarse_body_family"] = ""
        copied["formal_train_eligible"] = "true"
        copied["research_only"] = "true"
        copied["deployment_eligible"] = "false"
        copied["stage211_origin"] = "stage210_inatrc_supervised"
        copied["stage211_passenger_multiplier"] = "1.000000"
        copied["sample_weight"] = weighted(copied.get("sample_weight", ""), 0.85)
        copied["training_reason"] = "new_real_toll_cctv_exact_truck_taxonomy"
        combined_rows.append(copied)
        counters[f"stage210_supervised:{copied['body_type']}"] += 1

    combined_exact = {identity(row) for row in combined_rows}
    if len(combined_exact) != len(combined_rows):
        raise RuntimeError("combined supervised manifest contains exact duplicates")
    if validation_signature(combined_rows) != base_validation_signature:
        raise RuntimeError("base validation membership or row content changed")

    combined_unlabeled = [dict(row) for row in base_unlabeled]
    supplement_unlabeled_exact: set[str] = set()
    for row in supplement_unlabeled:
        digest = identity(row)
        if digest in base_exact or digest in base_unlabeled_exact or digest in supplement_exact or digest in supplement_unlabeled_exact:
            raise RuntimeError("Stage210 unlabeled exact overlap drift")
        supplement_unlabeled_exact.add(digest)
        copied = dict(row)
        if truthy(copied.get("body_type_supervised")) or truthy(copied.get("color_supervised")):
            raise RuntimeError("Stage210 unlabeled supervision drift")
        if copied.get("body_type") != "unknown" or copied.get("color") != "unknown":
            raise RuntimeError("Stage210 unlabeled label drift")
        copied["stage211_origin"] = "stage210_inatrc_unlabeled"
        copied["research_only"] = "true"
        copied["deployment_eligible"] = "false"
        combined_unlabeled.append(copied)

    combined_unlabeled_exact = {identity(row) for row in combined_unlabeled}
    if len(combined_unlabeled_exact) != len(combined_unlabeled):
        raise RuntimeError("combined unlabeled manifest contains exact duplicates")
    combined_supervised_unlabeled_overlap = combined_exact & combined_unlabeled_exact
    if combined_supervised_unlabeled_overlap != baseline_supervised_unlabeled_overlap:
        raise RuntimeError("Stage211 introduced new supervised/unlabeled exact overlap")

    train_groups = {row_group(row) for row in combined_rows if str(row.get("split", "")).lower() == "train"}
    validation_groups = {row_group(row) for row in combined_rows if str(row.get("split", "")).lower() in {"validation", "val"}}
    train_validation_group_overlap = len(train_groups & validation_groups)
    if train_validation_group_overlap:
        raise RuntimeError("train/validation group overlap")

    all_fields = list(base_fields)
    for field in sup_fields + ["stage211_source_coarse_body_family", "stage211_passenger_multiplier", "stage211_origin"]:
        if field not in all_fields:
            all_fields.append(field)
    all_unlabeled_fields = list(unlabeled_fields)
    for field in new_unlabeled_fields + ["stage211_origin"]:
        if field not in all_unlabeled_fields:
            all_unlabeled_fields.append(field)

    args.output_dir.mkdir(parents=True)
    supervised_path = args.output_dir / "attribute_manifest.stage211-inatrc-toll-hardclass.csv"
    unlabeled_path = args.output_dir / "attribute_manifest.stage211-unlabeled.csv"
    with supervised_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined_rows)
    with unlabeled_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_unlabeled_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined_unlabeled)

    supervised_train = [row for row in combined_rows if str(row.get("split", "")).lower() == "train"]
    validation = [row for row in combined_rows if str(row.get("split", "")).lower() in {"validation", "val"}]
    exact_supervised_train = [row for row in supervised_train if truthy(row.get("body_type_supervised")) and row.get("body_type") not in {"", "unknown"}]
    unlabeled_night = sum(truthy(row.get("night")) or str(row.get("lighting", "")).lower() == "night" for row in combined_unlabeled)
    expected_night_fraction_weight4 = (unlabeled_night * 4.0) / (unlabeled_night * 4.0 + len(combined_unlabeled) - unlabeled_night)

    report = {
        "schema_version": "stage211-inatrc-toll-hardclass-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_research_candidate_manifests_ready",
        "inputs": {"paths": [str(path) for path in inputs], "sha256": hashes},
        "supervised": {
            "rows": len(combined_rows),
            "train_rows": len(supervised_train),
            "validation_rows": len(validation),
            "exact_supervised_train_rows": len(exact_supervised_train),
            "stage210_rows_added": len(supplement_supervised),
            "stage210_class_counts": {key.split(":", 1)[1]: value for key, value in counters.items() if key.startswith("stage210_supervised:")},
            "passenger_reweighted_rows": counters["passenger_reweighted_total"],
            "passenger_reweighted_class_counts": {key.split(":", 1)[1]: value for key, value in counters.items() if key.startswith("passenger_reweighted:")},
            "train_validation_group_overlap": train_validation_group_overlap,
            "base_validation_signature_preserved": True,
            "baseline_supervised_unlabeled_exact_overlap": len(baseline_supervised_unlabeled_overlap),
            "new_supervised_unlabeled_exact_overlap_added": 0,
        },
        "unlabeled": {
            "rows": len(combined_unlabeled),
            "stage210_rows_added": len(supplement_unlabeled),
            "night_rows": unlabeled_night,
            "expected_sampled_night_fraction_at_weight4": expected_night_fraction_weight4,
            "all_train_only_unknown_unsupervised": True,
        },
        "outputs": {
            "supervised_manifest": str(supervised_path),
            "supervised_manifest_sha256": sha256_file(supervised_path),
            "unlabeled_manifest": str(unlabeled_path),
            "unlabeled_manifest_sha256": sha256_file(unlabeled_path),
        },
        "policy": {
            "stage210_component_gate_required": True,
            "validation_membership_and_content_unchanged": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "research_only": True,
            "deployment_eligible": False,
            "training_started": False,
            "deployment_performed": False,
        },
    }
    report_path = args.output_dir / "stage211-inatrc-toll-hardclass-manifest-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "SHA256SUMS").write_text(
        f"{sha256_file(report_path)}  {report_path.name}\n"
        f"{sha256_file(supervised_path)}  {supervised_path.name}\n"
        f"{sha256_file(unlabeled_path)}  {unlabeled_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "supervised_rows": len(combined_rows),
        "exact_supervised_train_rows": len(exact_supervised_train),
        "unlabeled_rows": len(combined_unlabeled),
        "stage210_supervised_added": len(supplement_supervised),
        "stage210_unlabeled_added": len(supplement_unlabeled),
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
