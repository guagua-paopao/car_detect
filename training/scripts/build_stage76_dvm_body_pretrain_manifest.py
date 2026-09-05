#!/usr/bin/env python3
"""Build a leakage-preserving DVM body-type pretraining manifest from official metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


AD_TABLE = "19586296/tables/tables/Ad_table.csv"
BODY_MAPPING = {
    "Saloon": "sedan",
    "SUV": "suv",
    "MPV": "mpv",
    "Pickup": "pickup",
    "Combi Van": "van",
    "Panel Van": "van",
    "Car Derived Van": "van",
    "Window Van": "van",
    "Camper": "van",
    "Minibus": "bus",
    "Hatchback": "other",
    "Coupe": "other",
    "Estate": "other",
    "Convertible": "other",
    "Limousine": "other",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"{label} SHA256 mismatch: expected={expected} actual={actual}")
    return actual


def advertisement_id(source_frame_id: str) -> str:
    name = Path(source_frame_id).name
    parts = name.split("$$")
    if len(parts) < 7:
        return ""
    return f"{parts[-3]}$${parts[-2]}"


def read_bodytypes(archive: Path) -> dict[str, str]:
    with zipfile.ZipFile(archive) as opened:
        if AD_TABLE not in opened.namelist():
            raise RuntimeError(f"official metadata table missing: {AD_TABLE}")
        with opened.open(AD_TABLE) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="strict", newline="")
            rows = csv.DictReader(text, skipinitialspace=True)
            mapping: dict[str, str] = {}
            conflicts: set[str] = set()
            for row in rows:
                key = (row.get("Adv_ID") or "").strip()
                value = (row.get("Bodytype") or "").strip()
                if not key:
                    continue
                if key in mapping and mapping[key] != value:
                    conflicts.add(key)
                mapping[key] = value
    if conflicts:
        raise RuntimeError(f"conflicting official Bodytype rows: {len(conflicts)}")
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage76 evidence")

    input_sha = require_hash(args.input_manifest, args.expected_input_sha256, "input manifest")
    archive_sha = require_hash(args.archive, args.expected_archive_sha256, "DVM archive")
    labels_sha = require_hash(args.labels, args.expected_labels_sha256, "label contract")
    body_contract = json.loads(args.labels.read_text(encoding="utf-8"))["body_types"]
    if not set(BODY_MAPPING.values()).issubset(set(body_contract)):
        raise RuntimeError("DVM body mapping is incompatible with the body label contract")
    dataset_root = args.dataset_root.resolve()
    input_parent = args.input_manifest.resolve().parent
    official = read_bodytypes(args.archive)

    with args.input_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        input_rows = list(reader)
    if any(row.get("split") not in {"train", "validation"} for row in input_rows):
        raise RuntimeError("input contains a non train/validation split")
    if any("vcas_rtsp_demo_60s" in str(row).lower() or "36-48" in str(row).lower() for row in input_rows):
        raise RuntimeError("input contains a frozen-video marker")

    counters = Counter()
    output_rows: list[dict[str, str]] = []
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    hashes_by_split: dict[str, set[str]] = defaultdict(set)
    for source in input_rows:
        row = dict(source)
        split = row["split"]
        raw_path = Path(row["image_path"])
        image_path = raw_path if raw_path.is_absolute() else input_parent / raw_path
        image_path = image_path.resolve()
        image_path.relative_to(dataset_root)
        if not image_path.is_file():
            raise FileNotFoundError(f"missing retained DVM crop: {image_path}")
        adv_id = advertisement_id(row.get("source_frame_id", ""))
        source_bodytype = official.get(adv_id)
        if source_bodytype is None:
            raise RuntimeError(f"official advertisement metadata missing: {adv_id}")
        mapped = BODY_MAPPING.get(source_bodytype, "unknown")
        supervised = mapped != "unknown"
        row["image_path"] = str(image_path)
        row["body_type"] = mapped
        row["body_type_supervised"] = "true" if supervised else "false"
        row["color"] = "unknown"
        row["color_supervised"] = "false"
        row["review_status"] = "approved" if supervised else "unknown_official_bodytype"
        row["annotation_source"] = "official_dvm_ad_table_bodytype"
        row["review_method"] = "official_metadata+advertisement_id_join+parent_decode_dedup_audit"
        row["source_bodytype"] = source_bodytype
        row["stage76_body_truth_source"] = (
            f"official_dvm_bodytype:{source_bodytype}" if supervised else "official_dvm_bodytype_unmapped"
        )
        row["stage76_pretrain_eligibility"] = "CC-BY-NC-4.0_research-only_non-deployable"
        row["stage76_parent_manifest_sha256"] = input_sha
        counters[f"rows:{split}"] += 1
        counters[f"official:{source_bodytype or '<blank>'}"] += 1
        counters[f"mapped:{split}:{mapped}"] += 1
        if supervised:
            counters[f"supervised:{split}"] += 1
        else:
            counters[f"unknown:{split}"] += 1
        group = row.get("source_group") or row.get("track_group") or adv_id
        groups_by_split[split].add(group)
        digest = row.get("sha256") or row.get("crop_sha256")
        if digest:
            hashes_by_split[split].add(digest)
        output_rows.append(row)

    group_leaks = groups_by_split["train"] & groups_by_split["validation"]
    exact_leaks = hashes_by_split["train"] & hashes_by_split["validation"]
    required_train = {"sedan", "suv", "mpv", "pickup", "van", "bus", "other"}
    present_train = {
        row["body_type"] for row in output_rows
        if row["split"] == "train" and row["body_type_supervised"] == "true"
    }
    failures: list[str] = []
    if counters["supervised:train"] < 100000:
        failures.append("fewer than 100000 supervised DVM train rows")
    if counters["supervised:validation"] < 5000:
        failures.append("fewer than 5000 supervised DVM validation rows")
    if not required_train.issubset(present_train):
        failures.append(f"missing mapped train classes: {sorted(required_train - present_train)}")
    if group_leaks:
        failures.append(f"source-group leaks: {len(group_leaks)}")
    if exact_leaks:
        failures.append(f"exact-image leaks: {len(exact_leaks)}")

    extra_fields = [
        "source_bodytype", "stage76_body_truth_source", "stage76_pretrain_eligibility",
        "stage76_parent_manifest_sha256",
    ]
    output_fields = fields + [field for field in extra_fields if field not in fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    output_sha = sha256(args.output_manifest)
    report = {
        "schema_version": "stage76-dvm-body-pretrain-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not failures else "fail_closed",
        "input": {"manifest": str(args.input_manifest.resolve()), "sha256": input_sha, "rows": len(input_rows)},
        "archive": {"path": str(args.archive.resolve()), "sha256": archive_sha, "metadata_member": AD_TABLE},
        "labels": {"path": str(args.labels.resolve()), "sha256": labels_sha},
        "output": {"manifest": str(args.output_manifest.resolve()), "sha256": output_sha, "rows": len(output_rows)},
        "mapping": BODY_MAPPING,
        "counters": dict(sorted(counters.items())),
        "integrity": {
            "official_advertisement_joined_rows": len(output_rows),
            "missing_official_advertisements": 0,
            "source_group_cross_split_leaks": len(group_leaks),
            "exact_image_cross_split_leaks": len(exact_leaks),
            "parent_perceptual_near_duplicate_audit_inherited": True,
            "parent_manifest_sha256_pinned": True,
            "test_rows": 0,
            "frozen_markers": 0,
        },
        "policy": {
            "body_truth_source": "official DVM Ad_table.csv Bodytype only",
            "filename_or_pixel_body_inference_used": False,
            "unmapped_bodytypes_remain_unknown": True,
            "one_retained_crop_per_advertisement_inherited": True,
            "color_supervision_disabled": True,
            "usage": "listing/studio body pretraining followed by independent real-CCTV domain fine-tuning",
            "license": "CC BY-NC 4.0; research-only; non-commercial",
            "eligibility": "research-only_non-deployable",
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "failures": failures,
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "output": str(args.output_manifest),
        "output_sha256": output_sha, "counters": report["counters"], "failures": failures,
    }, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
