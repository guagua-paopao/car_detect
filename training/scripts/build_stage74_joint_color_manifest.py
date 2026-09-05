#!/usr/bin/env python3
"""Merge validation-frozen CCTV color evidence into the clean Stage71 base.

The Stage73 UA source failed only its source-local class-support gate, so none
of its rows may be used unless the joint UA+BMD audit independently passes.
All new image hashes are recomputed, exact duplicates are removed globally,
perceptual near duplicates are removed within color, validation groups are
protected, and the Stage71 validation projection must remain byte-logically
identical.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KNOWN_COLORS = {
    "black",
    "white",
    "silver_gray",
    "red",
    "blue",
    "green",
    "yellow_orange",
    "brown_beige",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def dhash64(path: Path) -> str:
    from PIL import Image

    with Image.open(path) as opened:
        image = opened.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(image.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return f"{value:016x}"


def has_near(value: int, candidates: list[int], maximum_hamming: int) -> bool:
    return any((value ^ candidate).bit_count() <= maximum_hamming for candidate in candidates)


def has_near_any_orientation(
    value: int, candidates: list[int], maximum_hamming: int
) -> bool:
    # Historical manifests contain both left>right and right>left dHash bit
    # conventions.  They are complements, so check both orientations.
    complement = value ^ ((1 << 64) - 1)
    return has_near(value, candidates, maximum_hamming) or has_near(
        complement, candidates, maximum_hamming
    )


def projection_digest(rows: list[dict[str, str]], fields: list[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = [str(row.get(field, "")) for field in fields]
        digest.update(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def load_report(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    actual = sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(f"report SHA256 mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8")), actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--ua-manifest", type=Path, required=True)
    parser.add_argument("--expected-ua-manifest-sha256", required=True)
    parser.add_argument("--ua-report", type=Path, required=True)
    parser.add_argument("--expected-ua-report-sha256", required=True)
    parser.add_argument("--ua-root", type=Path, required=True)
    parser.add_argument("--bmd-manifest", type=Path, required=True)
    parser.add_argument("--expected-bmd-manifest-sha256", required=True)
    parser.add_argument("--bmd-report", type=Path, required=True)
    parser.add_argument("--expected-bmd-report-sha256", required=True)
    parser.add_argument("--bmd-root", type=Path, required=True)
    parser.add_argument("--expected-validation-rule-sha256", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--near-duplicate-hamming", type=int, default=2)
    parser.add_argument("--minimum-new-rows", type=int, default=1000)
    parser.add_argument("--minimum-new-classes", type=int, default=6)
    parser.add_argument("--minimum-new-rows-per-class", type=int, default=30)
    args = parser.parse_args()
    for path in (args.output_manifest, args.output_report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {path}")
    if sha256(args.base_manifest).lower() != args.expected_base_sha256.lower():
        raise RuntimeError("base manifest SHA256 mismatch")
    if sha256(args.ua_manifest).lower() != args.expected_ua_manifest_sha256.lower():
        raise RuntimeError("UA manifest SHA256 mismatch")
    if sha256(args.bmd_manifest).lower() != args.expected_bmd_manifest_sha256.lower():
        raise RuntimeError("BMD manifest SHA256 mismatch")
    ua_report, ua_report_sha = load_report(args.ua_report, args.expected_ua_report_sha256)
    bmd_report, bmd_report_sha = load_report(args.bmd_report, args.expected_bmd_report_sha256)
    rule_sha = args.expected_validation_rule_sha256.lower()
    if ua_report.get("status") != "fail_closed_insufficient_train_only_evidence":
        raise RuntimeError("unexpected UA source-local status")
    if int(ua_report.get("accepted_rows", -1)) != 717:
        raise RuntimeError("UA accepted-row evidence changed")
    if ua_report.get("validation_rule_report_sha256", "").lower() != rule_sha:
        raise RuntimeError("UA validation rule mismatch")
    if bmd_report.get("status") != "pass_revalidated_auxiliary_available":
        raise RuntimeError("BMD revalidation did not pass")
    if bmd_report.get("validation_rule_report_sha256", "").lower() != rule_sha:
        raise RuntimeError("BMD validation rule mismatch")
    if ua_report.get("output_manifest_sha256", "").lower() != args.expected_ua_manifest_sha256.lower():
        raise RuntimeError("UA report/manifest mismatch")
    if bmd_report.get("output_manifest_sha256", "").lower() != args.expected_bmd_manifest_sha256.lower():
        raise RuntimeError("BMD report/manifest mismatch")

    base_fields, base_rows = load_csv(args.base_manifest)
    ua_fields, ua_rows = load_csv(args.ua_manifest)
    bmd_fields, bmd_rows = load_csv(args.bmd_manifest)
    if any(row.get("split") not in {"train", "validation"} for row in base_rows):
        raise RuntimeError("base manifest contains a non-training split")
    for row in base_rows:
        if row.get("split") != "train":
            continue
        if row.get("stage71_origin") == "dvm":
            row["sample_weight"] = "0.300000"
        elif row.get("stage71_origin") == "cctv":
            row["sample_weight"] = "10.000000"
    validation_rows = [row for row in base_rows if row.get("split") == "validation"]
    validation_digest_before = projection_digest(validation_rows, base_fields)
    validation_groups = {
        row.get("track_group", "") for row in validation_rows if row.get("track_group")
    }
    train_groups = {
        row.get("track_group", "") for row in base_rows if row.get("split") == "train" and row.get("track_group")
    }
    if train_groups & validation_groups:
        raise RuntimeError("base manifest contains a train/validation group leak")

    seen_sha: set[str] = {
        str(row.get("sha256", "")).lower()
        for row in base_rows
        if str(row.get("sha256", "")).strip()
    }
    seen_dhash_by_color: dict[str, list[int]] = defaultdict(list)
    for row in base_rows:
        color = str(row.get("color", ""))
        value = str(row.get("dhash64", "")).lower()
        if color in KNOWN_COLORS and len(value) == 16:
            seen_dhash_by_color[color].append(int(value, 16))

    candidates: list[tuple[str, Path, dict[str, str]]] = []
    for source, root, rows in (
        ("ua", args.ua_root.resolve(), ua_rows),
        ("bmd", args.bmd_root.resolve(), bmd_rows),
    ):
        for row in rows:
            if row.get("split") != "train" or not truthy(row.get("color_supervised")):
                raise RuntimeError(f"{source} contains a non-train or unsupervised row")
            if row.get("color") not in KNOWN_COLORS:
                raise RuntimeError(f"{source} contains an unsupported color")
            if "test" in str(row.get("image_path", "")).lower():
                raise RuntimeError(f"{source} contains a test path")
            row_rule = row.get(
                "stage73_validation_rule_sha256"
                if source == "ua"
                else "stage74_validation_rule_sha256",
                "",
            )
            if str(row_rule).lower() != rule_sha:
                raise RuntimeError(f"{source} row lacks the frozen validation rule")
            path = (root / row["image_path"]).resolve()
            try:
                path.relative_to(root.parent)
            except ValueError as error:
                raise RuntimeError(f"{source} image escapes its dataset tree") from error
            if not path.is_file():
                raise RuntimeError(f"{source} image is missing: {path}")
            candidates.append((source, path, dict(row)))

    candidates.sort(
        key=lambda item: (
            -float(item[2].get("pseudo_label_confidence", 0.0) or 0.0),
            item[0],
            item[2].get("image_path", ""),
        )
    )
    retained_new: list[dict[str, str]] = []
    rejection_reasons = Counter()
    retained_by_source = Counter()
    for source, path, row in candidates:
        actual_sha = sha256(path)
        stored_sha = str(row.get("sha256", "")).lower()
        if stored_sha and stored_sha != actual_sha:
            raise RuntimeError(f"stored image SHA256 mismatch: {path}")
        actual_dhash = dhash64(path)
        color = row["color"]
        if row.get("track_group", "") in validation_groups:
            rejection_reasons["validation_group_collision"] += 1
            continue
        if actual_sha in seen_sha:
            rejection_reasons["exact_duplicate"] += 1
            continue
        dhash_value = int(actual_dhash, 16)
        if has_near_any_orientation(
            dhash_value,
            seen_dhash_by_color[color],
            args.near_duplicate_hamming,
        ):
            rejection_reasons["perceptual_near_duplicate_same_color"] += 1
            continue
        row["sha256"] = actual_sha
        row["dhash64"] = actual_dhash
        row["stage74_joint_source"] = source
        row["stage71_origin"] = "cctv"
        row["stage71_role"] = "supervised_color"
        row["sample_weight"] = "10.000000"
        retained_new.append(row)
        retained_by_source[source] += 1
        seen_sha.add(actual_sha)
        seen_dhash_by_color[color].append(dhash_value)

    new_counts = Counter(row["color"] for row in retained_new)
    supported_classes = sum(
        count >= args.minimum_new_rows_per_class for count in new_counts.values()
    )
    output_rows = base_rows + retained_new
    output_fields = list(base_fields)
    for field in ua_fields + bmd_fields + ["stage74_joint_source"]:
        if field not in output_fields:
            output_fields.append(field)
    validation_after = [row for row in output_rows if row.get("split") == "validation"]
    validation_digest_after = projection_digest(validation_after, base_fields)
    if validation_digest_after != validation_digest_before:
        raise RuntimeError("base validation projection changed")
    output_train_groups = {
        row.get("track_group", "")
        for row in output_rows
        if row.get("split") == "train" and row.get("track_group")
    }
    group_leaks = output_train_groups & validation_groups
    if group_leaks:
        raise RuntimeError("joint manifest has train/validation group leaks")

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    status = (
        "pass_joint_research_manifest_available"
        if len(retained_new) >= args.minimum_new_rows
        and supported_classes >= args.minimum_new_classes
        else "fail_closed_joint_support_insufficient"
    )
    new_night = sum(truthy(row.get("night")) or row.get("lighting") == "night" for row in retained_new)
    new_small = sum(row.get("vehicle_size") == "small" for row in retained_new)
    new_occluded = sum(truthy(row.get("occluded")) or truthy(row.get("truncated")) for row in retained_new)
    train_rows = [row for row in output_rows if row.get("split") == "train"]
    weighted_total = sum(float(row.get("sample_weight", 1.0) or 1.0) for row in train_rows)
    weighted_cctv = sum(
        float(row.get("sample_weight", 1.0) or 1.0)
        for row in train_rows
        if row.get("stage71_origin") == "cctv"
    )
    report = {
        "schema_version": "stage74-joint-color-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "eligibility": "research-only_non-deployable",
        "base_manifest": str(args.base_manifest.resolve()),
        "base_manifest_sha256": sha256(args.base_manifest),
        "ua_manifest_sha256": sha256(args.ua_manifest),
        "ua_report_sha256": ua_report_sha,
        "bmd_manifest_sha256": sha256(args.bmd_manifest),
        "bmd_report_sha256": bmd_report_sha,
        "validation_rule_sha256": rule_sha,
        "base_rows": len(base_rows),
        "base_train_rows": sum(row.get("split") == "train" for row in base_rows),
        "validation_rows": len(validation_rows),
        "validation_projection_sha256_before": validation_digest_before,
        "validation_projection_sha256_after": validation_digest_after,
        "raw_new_rows": len(candidates),
        "retained_new_rows": len(retained_new),
        "retained_new_by_source": dict(sorted(retained_by_source.items())),
        "retained_new_color_counts": dict(sorted(new_counts.items())),
        "supported_new_classes": supported_classes,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "joint_rows": len(output_rows),
        "joint_train_rows": sum(row.get("split") == "train" for row in output_rows),
        "sample_weight_contract": {
            "new_cctv_sample_weight": 10.0,
            "existing_cctv_sample_weight": 10.0,
            "dvm_replay_sample_weight": 0.3,
            "expected_weighted_cctv_fraction_before_scene_weights": (
                weighted_cctv / weighted_total if weighted_total else 0.0
            ),
        },
        "new_scene_profile": {
            "night_rows": new_night,
            "night_fraction": new_night / len(retained_new) if retained_new else 0.0,
            "small_rows": new_small,
            "small_fraction": new_small / len(retained_new) if retained_new else 0.0,
            "occluded_or_truncated_rows": new_occluded,
            "occluded_or_truncated_fraction": new_occluded / len(retained_new) if retained_new else 0.0,
        },
        "gates": {
            "minimum_new_rows": args.minimum_new_rows,
            "minimum_new_classes": args.minimum_new_classes,
            "minimum_new_rows_per_class": args.minimum_new_rows_per_class,
            "near_duplicate_hamming_within_color": args.near_duplicate_hamming,
            "validation_projection_unchanged": True,
            "group_leaks": 0,
        },
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "policy": {
            "ua_source_local_failure_overridden": False,
            "ua_rows_usable_only_under_this_joint_pass": True,
            "all_new_rows_share_validation_frozen_rule": True,
            "image_sha256_recomputed": True,
            "image_dhash64_recomputed": True,
            "historical_dhash_orientation_normalized": True,
            "new_rows_explicitly_weighted_as_cctv": True,
            "dvm_replay_retained_with_reduced_epoch_sampling_weight": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "only pass may start an isolated research-only Stage74 candidate; deployment remains prohibited",
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "new_rows": len(retained_new), "counts": new_counts}))
    return 0 if status.startswith("pass_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
