#!/usr/bin/env python3
"""Resolve Stage219 novel adverse color keys to auditable train-only images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
HEX16 = re.compile(r"^[0-9a-fA-F]{16}$")
FROZEN_MARKERS = {"vcas_rtsp_demo_60s", "36-48s", "36_48s", "36–48"}
SKIP_PATH_MARKERS = {"test", "holdout", "future", "evaluation", "eval", "sealed", "frozen", "stage148", "stage155", "stage171"}
POSITIVE_TRUTH_MARKERS = {"exact", "official", "registration", "folder", "ground_truth", "ground-truth", "manual", "human", "metadata", "source_label", "source-label", "verified"}
MODEL_TRUTH_MARKERS = {"pseudo", "teacher", "consensus", "model", "machine", "weak", "coarse", "inferred", "predicted", "prediction"}
EVIDENCE_FIELDS = (
    "color_supervision", "annotation_source", "review_method",
    "stage65_color_truth_source", "stage66_color_truth_source",
    "taxonomy_v2_color_source", "color_review_status", "review_status",
    "fine_color_recovery", "stage102_supervision",
)
OUTPUT_FIELDS = [
    "image_path", "split", "body_type", "body_type_supervised", "color",
    "color_supervised", "color_supervision", "source_dataset", "source_license",
    "license_train_eligible", "formal_train_eligible", "research_only",
    "review_status", "review_method", "camera_id", "video_id", "track_group",
    "source_group", "source_frame_id", "source_image_id", "lighting", "night",
    "low_light", "weather", "width", "height", "gray_mean", "gray_stddev",
    "edge_variance", "sha256", "dhash64", "sample_weight", "stage220_origin",
    "stage220_source_manifest", "stage220_scene_claim", "stage220_pixel_mean_luma",
    "stage220_pixel_median_luma", "stage220_pixel_border_mean_luma",
    "stage220_pixel_dark_fraction", "stage220_pixel_highlight_fraction",
    "stage220_pixel_contrast", "stage220_pixel_sharpness", "stage220_lowlight_score",
    "stage220_frame_lowlight_proxy", "stage220_frame_night_candidate",
    "stage220_validation_near_duplicate_distance",
]


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def contains_marker(text: str, markers: set[str]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def skip_manifest(path: Path) -> bool:
    return contains_marker(str(path).replace("\\", "/"), SKIP_PATH_MARKERS)


def row_frozen(row: dict[str, str]) -> bool:
    text = " ".join(str(row.get(field) or "") for field in ("image_path", "source_manifest", "source_frame_id", "video_id"))
    return contains_marker(text, FROZEN_MARKERS)


def row_sha(row: dict[str, str]) -> str:
    value = str(row.get("sha256") or row.get("crop_sha256") or row.get("source_sha256") or "").strip().lower()
    return value if HEX64.fullmatch(value) else ""


def authoritative_color(row: dict[str, str]) -> bool:
    if not truthy(row.get("color_supervised")) or truthy(row.get("pseudo_label")):
        return False
    evidence = " ".join(str(row.get(field) or "") for field in EVIDENCE_FIELDS).lower()
    return not contains_marker(evidence, MODEL_TRUTH_MARKERS) and contains_marker(evidence, POSITIVE_TRUTH_MARKERS)


def resolved_image_path(raw: str, manifest: Path, allowed_root: Path) -> Path | None:
    image = Path(str(raw or "").strip())
    if not image.is_absolute():
        image = manifest.parent / image
    try:
        image = image.resolve(strict=True)
        image.relative_to(allowed_root)
    except (FileNotFoundError, RuntimeError, ValueError):
        return None
    if not image.is_file() or image.is_symlink():
        return None
    return image


def dhash64(gray: np.ndarray) -> int:
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def analyze_image(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("decode_failed")
    original_height, original_width = image.shape[:2]
    scale = min(1.0, 192.0 / max(original_height, original_width))
    if scale < 1.0:
        image = cv2.resize(image, (max(1, round(original_width * scale)), max(1, round(original_height * scale))), interpolation=cv2.INTER_AREA)
    gray_u8 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32)
    height, width = gray.shape
    border_y = max(2, round(height * 0.12))
    border_x = max(2, round(width * 0.12))
    mask = np.zeros_like(gray, dtype=bool)
    mask[:border_y, :] = True
    mask[-border_y:, :] = True
    mask[:, :border_x] = True
    mask[:, -border_x:] = True
    mean = float(gray.mean())
    median = float(np.median(gray))
    p90 = float(np.percentile(gray, 90))
    border_mean = float(gray[mask].mean())
    dark_fraction = float((gray < 55.0).mean())
    highlight_fraction = float((gray > 235.0).mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    score = (
        0.28 * float(np.clip((95.0 - mean) / 65.0, 0.0, 1.0))
        + 0.22 * float(np.clip((90.0 - median) / 65.0, 0.0, 1.0))
        + 0.25 * float(np.clip((100.0 - border_mean) / 70.0, 0.0, 1.0))
        + 0.15 * min(1.0, dark_fraction / 0.60)
        + 0.10 * float(np.clip((165.0 - p90) / 105.0, 0.0, 1.0))
    )
    return {
        "pixel_sha256": hashlib.sha256(payload).hexdigest(),
        "dhash_int": dhash64(gray_u8),
        "dhash64": f"{dhash64(gray_u8):016x}",
        "width": original_width,
        "height": original_height,
        "mean_luma": mean,
        "median_luma": median,
        "border_mean_luma": border_mean,
        "dark_fraction": dark_fraction,
        "highlight_fraction": highlight_fraction,
        "contrast": contrast,
        "sharpness": sharpness,
        "lowlight_score": score,
        "frame_lowlight": score >= 0.62 and mean <= 88.0 and median <= 82.0 and border_mean <= 95.0 and dark_fraction >= 0.30,
        "frame_night_candidate": score >= 0.78 and mean <= 70.0 and median <= 65.0 and border_mean <= 80.0 and dark_fraction >= 0.45,
    }


def dhash_values(row: dict[str, str]) -> set[int]:
    values: set[int] = set()
    for field in ("dhash64", "crop_dhash64", "stage150_dhash64", "stage157_dhash64"):
        value = str(row.get(field) or "").strip().lower()
        if HEX16.fullmatch(value):
            values.add(int(value, 16))
    return values


def group_values(row: dict[str, str]) -> set[str]:
    return {str(row.get(field) or "").strip() for field in ("stage159_validation_group", "track_group", "source_group") if str(row.get(field) or "").strip()}


def candidate_row_score(row: dict[str, str], image: Path | None) -> tuple[int, int, int, int, str]:
    return (
        int(image is not None),
        int(bool(str(row.get("source_license") or "").strip())),
        int(bool(str(row.get("track_group") or row.get("source_group") or "").strip())),
        int(bool(str(row.get("review_method") or row.get("color_supervision") or "").strip())),
        str(row.get("image_path") or ""),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--expected-candidates-sha256", required=True)
    parser.add_argument("--stage218-report", type=Path, required=True)
    parser.add_argument("--expected-stage218-report-sha256", required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--expected-validation-sha256", required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    args = parser.parse_args()

    inputs = (
        (args.candidates.resolve(), args.expected_candidates_sha256),
        (args.stage218_report.resolve(), args.expected_stage218_report_sha256),
        (args.validation_manifest.resolve(), args.expected_validation_sha256),
    )
    for path, expected in inputs:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"input must be a regular non-symlink file: {path}")
        actual = sha256_file(path)
        if actual.lower() != expected.lower():
            raise RuntimeError(f"SHA256 mismatch for {path}: {actual}")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    allowed_root = args.allowed_root.resolve()

    candidates: dict[str, dict[str, str]] = {}
    with args.candidates.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("key_type") != "sha256" or not HEX64.fullmatch(str(row.get("key") or "")):
                raise RuntimeError("Stage220 requires SHA256-keyed candidates")
            candidates[str(row["key"]).lower()] = row

    validation_sha: set[str] = set()
    validation_dhash: set[int] = set()
    validation_groups: set[str] = set()
    with args.validation_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row_sha(row)
            if value:
                validation_sha.add(value)
            validation_dhash.update(dhash_values(row))
            validation_groups.update(group_values(row))

    stage218 = json.loads(args.stage218_report.read_text(encoding="utf-8"))
    selected_manifests: list[tuple[Path, str]] = []
    for record in stage218.get("manifests", []):
        if record.get("status") != "scanned":
            continue
        if int((record.get("counts") or {}).get("truth_authoritative") or 0) <= 0:
            continue
        path = Path(record["path"])
        if skip_manifest(path):
            continue
        selected_manifests.append((path, str(record["sha256"])))

    matches: dict[str, list[tuple[dict[str, str], Path, Path | None]]] = defaultdict(list)
    scanned_manifest_count = 0
    for manifest, expected_sha in selected_manifests:
        if not manifest.is_file() or manifest.is_symlink():
            raise RuntimeError(f"Stage218 manifest missing or unsafe: {manifest}")
        if sha256_file(manifest) != expected_sha:
            raise RuntimeError(f"Stage218 manifest changed: {manifest}")
        scanned_manifest_count += 1
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, 2):
                if str(row.get("split") or "").strip().lower() != "train":
                    continue
                value = row_sha(row)
                if value not in candidates:
                    continue
                if row_frozen(row):
                    raise RuntimeError(f"frozen marker in candidate row {manifest}:{row_number}")
                if not authoritative_color(row):
                    continue
                image = resolved_image_path(str(row.get("image_path") or ""), manifest, allowed_root)
                matches[value].append((dict(row), manifest, image))

    args.output_root.mkdir(parents=True, exist_ok=False)
    accepted_path = args.output_root / "attribute_manifest.stage220-novel-adverse-color-audited.csv"
    quarantine_path = args.output_root / "stage220-quarantined-candidates.csv"
    counts: Counter[str] = Counter()
    accepted_scene: Counter[str] = Counter()
    accepted_color: Counter[str] = Counter()
    accepted_source: Counter[str] = Counter()
    quarantine_reasons: Counter[str] = Counter()
    contact_rows: list[dict[str, Any]] = []

    with accepted_path.open("w", encoding="utf-8", newline="") as accepted_handle, quarantine_path.open("w", encoding="utf-8", newline="") as quarantine_handle:
        accepted_writer = csv.DictWriter(accepted_handle, fieldnames=OUTPUT_FIELDS)
        quarantine_fields = ["sha256", "color", "scene", "source_dataset", "reason", "matched_rows"]
        quarantine_writer = csv.DictWriter(quarantine_handle, fieldnames=quarantine_fields)
        accepted_writer.writeheader()
        quarantine_writer.writeheader()
        for sha, candidate in sorted(candidates.items()):
            counts["candidates"] += 1
            reasons: list[str] = []
            if sha in validation_sha:
                reasons.append("exact_validation_overlap")
            rows = matches.get(sha, [])
            if not rows:
                reasons.append("no_authoritative_train_row_resolved")
                selected = None
            else:
                selected = max(rows, key=lambda item: candidate_row_score(item[0], item[2]))
                colors = {str(item[0].get("color") or "").strip().lower() for item in rows}
                if len(colors) != 1 or next(iter(colors)) != str(candidate.get("color") or "").strip().lower():
                    reasons.append("resolved_color_conflict")
                all_groups = set().union(*(group_values(item[0]) for item in rows))
                if all_groups & validation_groups:
                    reasons.append("validation_group_overlap")
                if selected[2] is None:
                    reasons.append("image_missing_or_outside_allowed_root")
                license_text = str(selected[0].get("source_license") or "").strip()
                if not license_text:
                    reasons.append("missing_license_evidence")
                if str(selected[0].get("license_train_eligible") or "").strip().lower() in {"false", "0", "no"}:
                    reasons.append("license_train_ineligible")

            metrics: dict[str, Any] | None = None
            nearest_distance: int | None = None
            if selected is not None and selected[2] is not None and not reasons:
                try:
                    counts["candidate_train_pixel_decode_attempts"] += 1
                    metrics = analyze_image(selected[2])
                    if metrics["pixel_sha256"] != sha:
                        reasons.append("pixel_sha256_mismatch")
                    if validation_dhash:
                        nearest_distance = min((metrics["dhash_int"] ^ value).bit_count() for value in validation_dhash)
                        if nearest_distance <= args.near_duplicate_hamming:
                            reasons.append("validation_perceptual_near_duplicate")
                except Exception as error:
                    reasons.append(f"decode_error:{type(error).__name__}")

            reasons = sorted(set(reasons))
            if reasons:
                for reason in reasons:
                    quarantine_reasons[reason] += 1
                quarantine_writer.writerow({
                    "sha256": sha,
                    "color": candidate.get("color", ""),
                    "scene": candidate.get("scene", ""),
                    "source_dataset": candidate.get("source_dataset", ""),
                    "reason": "|".join(reasons),
                    "matched_rows": len(rows),
                })
                counts["quarantined"] += 1
                continue

            assert selected is not None and selected[2] is not None and metrics is not None
            row, manifest, image = selected
            output = {field: str(row.get(field) or "") for field in OUTPUT_FIELDS}
            output.update({
                "image_path": str(image),
                "split": "train",
                "body_type": "unknown",
                "body_type_supervised": "false",
                "color": str(candidate["color"]).lower(),
                "color_supervised": "true",
                "formal_train_eligible": "false",
                "research_only": "true",
                "sha256": sha,
                "dhash64": metrics["dhash64"],
                "stage220_origin": "stage219_novel_adverse_authoritative_color",
                "stage220_source_manifest": str(manifest),
                "stage220_scene_claim": candidate["scene"],
                "stage220_pixel_mean_luma": f"{metrics['mean_luma']:.6f}",
                "stage220_pixel_median_luma": f"{metrics['median_luma']:.6f}",
                "stage220_pixel_border_mean_luma": f"{metrics['border_mean_luma']:.6f}",
                "stage220_pixel_dark_fraction": f"{metrics['dark_fraction']:.8f}",
                "stage220_pixel_highlight_fraction": f"{metrics['highlight_fraction']:.8f}",
                "stage220_pixel_contrast": f"{metrics['contrast']:.6f}",
                "stage220_pixel_sharpness": f"{metrics['sharpness']:.6f}",
                "stage220_lowlight_score": f"{metrics['lowlight_score']:.8f}",
                "stage220_frame_lowlight_proxy": str(bool(metrics["frame_lowlight"])).lower(),
                "stage220_frame_night_candidate": str(bool(metrics["frame_night_candidate"])).lower(),
                "stage220_validation_near_duplicate_distance": str(nearest_distance if nearest_distance is not None else ""),
            })
            accepted_writer.writerow(output)
            counts["accepted_for_scene_visual_audit"] += 1
            accepted_scene[candidate["scene"]] += 1
            accepted_color[str(candidate["color"])] += 1
            for source in str(candidate.get("source_dataset") or "unknown").split("|"):
                if source:
                    accepted_source[source] += 1
            contact_rows.append({"image_path": str(image), "sha256": sha, "color": candidate["color"], "scene": candidate["scene"]})

    report = {
        "schema_version": "stage220-novel-adverse-color-resolution-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_fail_closed_candidates_ready_for_agent_visual_scene_audit",
        "inputs": {
            "candidates": str(args.candidates.resolve()),
            "candidates_sha256": sha256_file(args.candidates.resolve()),
            "stage218_report": str(args.stage218_report.resolve()),
            "stage218_report_sha256": sha256_file(args.stage218_report.resolve()),
            "validation_manifest": str(args.validation_manifest.resolve()),
            "validation_manifest_sha256": sha256_file(args.validation_manifest.resolve()),
            "selected_authoritative_manifests": len(selected_manifests),
            "selected_authoritative_manifests_scanned": scanned_manifest_count,
        },
        "counts": dict(counts),
        "accepted": {
            "scene_counts": dict(accepted_scene),
            "color_counts": dict(accepted_color),
            "source_counts": dict(accepted_source),
        },
        "quarantine_reasons": dict(quarantine_reasons),
        "outputs": {
            "audited_manifest": str(accepted_path),
            "audited_manifest_sha256": sha256_file(accepted_path),
            "quarantine": str(quarantine_path),
            "quarantine_sha256": sha256_file(quarantine_path),
        },
        "policy": {
            "candidate_train_pixels_opened": counts["candidate_train_pixel_decode_attempts"],
            "validation_pixels_opened": 0,
            "test_accessed": False,
            "frozen_video_used": False,
            "exact_and_dhash_validation_leakage_fail_closed": True,
            "source_scene_metadata_not_yet_final_truth": True,
            "agent_visual_scene_audit_required_before_training": True,
            "training_manifest_created": False,
            "training_started": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    report_path = args.output_root / "stage220-novel-adverse-color-resolution.json"
    atomic_json(report_path, report)
    for path in (accepted_path, quarantine_path, report_path):
        path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "accepted": report["accepted"], "quarantine_reasons": report["quarantine_reasons"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
