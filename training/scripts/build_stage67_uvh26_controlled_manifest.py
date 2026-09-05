#!/usr/bin/env python3
"""Build a licensed, group-capped and class-balanced UVH-26 follow-up manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "frozen_video")
LICENSE = "CC-BY-4.0"
DATASET_URL = "https://huggingface.co/datasets/iisc-aim/UVH-26"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def dhash64(path: Path) -> str:
    from PIL import Image

    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return f"{value:016x}"


def stable_score(row: dict[str, str], seed: int) -> str:
    identity = row.get("crop_sha256") or row.get("image_path", "")
    return hashlib.sha256(f"{seed}|{identity}".encode()).hexdigest()


def select_balanced(
    rows: list[dict[str, str]],
    maximum_rows: int,
    maximum_per_group: int,
    seed: int,
) -> list[dict[str, str]]:
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row["body_type"]].append(row)
    for values in by_class.values():
        values.sort(key=lambda row: (stable_score(row, seed), row.get("image_path", "")))
    offsets = {name: 0 for name in by_class}
    group_counts: Counter[str] = Counter()
    selected: list[dict[str, str]] = []
    class_names = sorted(by_class)
    while len(selected) < maximum_rows:
        progressed = False
        for name in class_names:
            values = by_class[name]
            while offsets[name] < len(values):
                row = values[offsets[name]]
                offsets[name] += 1
                group = row.get("track_group") or row.get("source_frame_id") or row.get("image_path", "")
                if group_counts[group] >= maximum_per_group:
                    continue
                selected.append(row)
                group_counts[group] += 1
                progressed = True
                break
            if len(selected) >= maximum_rows:
                break
        if not progressed:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--base-leakage-report", type=Path, required=True)
    parser.add_argument("--uvh-manifest", type=Path, required=True)
    parser.add_argument("--uvh-report", type=Path, required=True)
    parser.add_argument("--license-evidence", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--maximum-rows", type=int, default=3000)
    parser.add_argument("--maximum-per-group", type=int, default=3)
    parser.add_argument("--seed", type=int, default=6701)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage67 manifest evidence")
    if args.maximum_rows <= 0 or args.maximum_per_group <= 0:
        raise ValueError("selection limits must be positive")

    with args.base_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        base_reader = csv.DictReader(handle)
        if not base_reader.fieldnames:
            raise RuntimeError("base manifest has no header")
        base_fields = list(base_reader.fieldnames)
        base_rows = list(base_reader)
    with args.uvh_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        uvh_reader = csv.DictReader(handle)
        if not uvh_reader.fieldnames:
            raise RuntimeError("UVH manifest has no header")
        uvh_fields = list(uvh_reader.fieldnames)
        uvh_rows = list(uvh_reader)

    base_leakage = json.loads(args.base_leakage_report.read_text(encoding="utf-8"))
    if base_leakage.get("status") != "pass" or base_leakage.get("manifest_sha256", "").lower() != sha256(args.base_manifest):
        raise RuntimeError("base manifest leakage evidence is invalid")
    if base_leakage.get("dhash", {}).get("cross_split_near_pairs") != 0:
        raise RuntimeError("base manifest still has cross-split near overlap")
    uvh_report = json.loads(args.uvh_report.read_text(encoding="utf-8"))
    if (
        uvh_report.get("source_dataset") != "iisc-aim/UVH-26"
        or uvh_report.get("source_license") != LICENSE
        or uvh_report.get("output_manifest_sha256", "").lower() != sha256(args.uvh_manifest)
    ):
        raise RuntimeError("UVH source evidence is invalid")
    license_evidence = json.loads(args.license_evidence.read_text(encoding="utf-8"))
    if (
        license_evidence.get("status") != "pass_primary_source_verified"
        or license_evidence.get("declared_dataset_license") != LICENSE
        or not license_evidence.get("primary_source_findings", {}).get("training_use_allowed_with_attribution")
        or not license_evidence.get("decision", {}).get("production_training_license_eligible")
    ):
        raise RuntimeError("UVH primary-source license evidence is invalid")

    candidates = []
    for row in uvh_rows:
        if row.get("source_dataset") != "UVH-26":
            continue
        if (
            row.get("split") != "train"
            or row.get("review_status") != "approved"
            or not truthy(row.get("body_type_supervised"))
            or truthy(row.get("color_supervised"))
            or row.get("body_type") in {"", "unknown"}
            or row.get("color") not in {"", "unknown"}
            or row.get("source_license") != LICENSE
        ):
            raise RuntimeError(f"unsafe UVH row: {row.get('image_path')}")
        marker_text = " ".join(str(value).lower() for value in row.values())
        if any(marker in marker_text for marker in FROZEN_MARKERS):
            raise RuntimeError("frozen-video marker found in UVH source")
        candidates.append(dict(row))
    base_hashes = {row.get("crop_sha256") or row.get("sha256") for row in base_rows} - {None, ""}
    exact_candidates_removed = sum((row.get("crop_sha256") or row.get("sha256")) in base_hashes for row in candidates)
    candidates = [
        row for row in candidates
        if (row.get("crop_sha256") or row.get("sha256")) not in base_hashes
    ]
    selected = select_balanced(candidates, args.maximum_rows, args.maximum_per_group, args.seed)
    if len(selected) != args.maximum_rows:
        raise RuntimeError(f"could select only {len(selected)} safe UVH rows")

    def image_path(row: dict[str, str]) -> Path:
        path = Path(row["image_path"])
        return path if path.is_absolute() else (args.uvh_manifest.parent / path).resolve()

    def audit(row: dict[str, str]) -> tuple[str, str, str]:
        path = image_path(row)
        actual_sha = sha256(path)
        expected_sha = row.get("crop_sha256") or row.get("sha256")
        if expected_sha and expected_sha.lower() != actual_sha:
            raise RuntimeError(f"UVH crop hash mismatch: {path}")
        return row["image_path"], actual_sha, dhash64(path)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        audited = list(executor.map(audit, selected))
    audited_by_path = {path: (file_hash, perceptual) for path, file_hash, perceptual in audited}
    for row in selected:
        file_hash, perceptual = audited_by_path[row["image_path"]]
        row.update({
            "sha256": file_hash,
            "crop_sha256": file_hash,
            "dhash64": perceptual,
            "license_train_eligible": "true",
            "formal_train_eligible": "true",
            "source_landing_url": DATASET_URL,
            "source_license_url": LICENSE_URL,
            "training_reason": "licensed_uvh26_group_capped_small_target_body_truth",
            "stage67_uvh26_licensed": "true",
            "color": "unknown",
            "color_supervised": "false",
        })
    if len({row["crop_sha256"] for row in selected}) != len(selected):
        raise RuntimeError("selected UVH rows contain exact duplicate crops")
    base_train = [row for row in base_rows if row.get("split") == "train"]
    base_heldout = [row for row in base_rows if row.get("split") != "train"]
    output_rows = [*base_train, *selected, *base_heldout]
    output_fields = list(dict.fromkeys([
        *base_fields, *uvh_fields, "source_landing_url", "source_license_url",
        "training_reason", "stage67_uvh26_licensed",
    ]))
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(output_rows)
    group_counts = Counter(row.get("track_group") or row.get("source_frame_id") for row in selected)
    report = {
        "schema_version": "attribute-stage67-uvh26-controlled-manifest-report-v1",
        "status": "pass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_manifest": str(args.base_manifest.resolve()),
        "base_manifest_sha256": sha256(args.base_manifest),
        "base_leakage_report": str(args.base_leakage_report.resolve()),
        "base_leakage_report_sha256": sha256(args.base_leakage_report),
        "uvh_manifest": str(args.uvh_manifest.resolve()),
        "uvh_manifest_sha256": sha256(args.uvh_manifest),
        "uvh_report_sha256": sha256(args.uvh_report),
        "license_evidence_sha256": sha256(args.license_evidence),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "base_rows": len(base_rows),
        "available_uvh_rows": len(candidates),
        "exact_candidates_already_in_base_removed": exact_candidates_removed,
        "selected_uvh_rows": len(selected),
        "selected_unique_groups": len(group_counts),
        "selected_maximum_rows_per_group": max(group_counts.values()),
        "selected_body_type_counts": dict(Counter(row["body_type"] for row in selected)),
        "selected_small_target_rows": sum(truthy(row.get("small_target")) or row.get("vehicle_size") == "small" for row in selected),
        "selected_exact_color_rows": 0,
        "selected_unreadable_rows": 0,
        "output_rows": len(output_rows),
        "output_train_rows": len(base_train) + len(selected),
        "heldout_rows_preserved": len(base_heldout),
        "policy": {
            "maximum_rows": args.maximum_rows,
            "maximum_rows_per_source_group": args.maximum_per_group,
            "selection_seed": args.seed,
            "class_round_robin_balancing": True,
            "all_selected_crop_hashes_verified_from_pixels": True,
            "color_truth_remains_unknown_and_unsupervised": True,
            "heldout_membership_preserved": True,
            "test_labels_or_predictions_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "deployment_paused_by_user": True,
        },
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "selected_uvh_rows": len(selected),
        "selected_unique_groups": len(group_counts),
        "selected_body_type_counts": report["selected_body_type_counts"],
        "output_rows": len(output_rows),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
