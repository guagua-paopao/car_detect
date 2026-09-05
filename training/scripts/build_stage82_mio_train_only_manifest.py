#!/usr/bin/env python3
"""Recover audited MIO-TCD crops into a v2 train-only body manifest.

The interrupted crop directory is immutable source evidence.  This builder
re-decodes every image, binds labels to the official ``gt_train.csv``, and
deduplicates against the active Stage72 lineage.  It never reads image payloads
from the official MIO test split and never creates validation or test rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageStat

from build_stage70_specialist_manifests import HammingBKTree


SOURCE_TO_BODY = {
    "articulated_truck": "heavy_truck",
    "bus": "bus",
    "pickup_truck": "pickup",
    "single_unit_truck": "truck",
    "work_van": "van",
}
LICENSE_MARKER = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0"
FIELDS = [
    "image_path", "split", "body_type", "body_type_supervised", "coarse_body_family",
    "color", "color_supervised", "source_dataset", "source_label", "source_license",
    "license_train_eligible", "review_status", "review_method", "camera_id", "video_id",
    "track_group", "source_frame_id", "crop_quality", "viewpoint", "lighting",
    "night", "occluded", "truncated", "blur", "width", "height", "gray_mean",
    "gray_stddev", "edge_variance", "sha256", "dhash64",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source_evidence(archive: Path) -> tuple[dict[str, str], str]:
    with tarfile.open(archive, "r:") as opened:
        readme_file = opened.extractfile("README.txt")
        labels_file = opened.extractfile("gt_train.csv")
        if readme_file is None or labels_file is None:
            raise RuntimeError("MIO archive is missing README.txt or gt_train.csv")
        readme = readme_file.read().decode("utf-8", errors="replace")
        labels = {
            row[0].strip(): row[1].strip()
            for row in csv.reader(io.TextIOWrapper(labels_file, encoding="utf-8-sig"))
            if len(row) == 2
        }
    if LICENSE_MARKER not in readme:
        raise RuntimeError("MIO CC BY-NC-SA 4.0 license marker is missing")
    return labels, hashlib.sha256(readme.encode("utf-8")).hexdigest()


def content_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_sha256") or row.get("sha256") or row.get("image_sha256") or "").strip().lower()


def perceptual_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_dhash64") or row.get("dhash64") or "").strip().lower()


def read_successor(path: Path) -> tuple[set[str], HammingBKTree, int]:
    exact: set[str] = set()
    tree = HammingBKTree()
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            digest = content_hash(row)
            if digest:
                exact.add(digest)
            value = perceptual_hash(row)
            if len(value) == 16:
                tree.add(int(value, 16), row)
    return exact, tree, rows


def dhash64(image: Image.Image) -> int:
    values = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    result = 0
    for bit in (values[:, 1:] > values[:, :-1]).ravel():
        result = (result << 1) | int(bit)
    return result


def edge_variance(image: Image.Image) -> float:
    values = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    if min(values.shape) > 4:
        values = values[2:-2, 2:-2]
    return float(values.var())


def inspect(path: Path) -> tuple[str, int, dict[str, object]]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    with Image.open(io.BytesIO(payload)) as source:
        source.load()
        image = source.convert("RGB")
    width, height = image.size
    gray = image.convert("L")
    stats = ImageStat.Stat(gray)
    mean = float(stats.mean[0])
    stddev = float(stats.stddev[0])
    edges = edge_variance(image)
    if min(width, height) < 32:
        raise ValueError("small_dimension")
    if width / height < 0.35 or width / height > 4.5:
        raise ValueError("extreme_aspect")
    if mean < 8.0 or mean > 248.0:
        raise ValueError("extreme_exposure")
    quality = "good" if min(width, height) >= 96 and stddev >= 20.0 and edges >= 80.0 else "usable"
    return digest, dhash64(image), {
        "width": width,
        "height": height,
        "gray_mean": round(mean, 4),
        "gray_stddev": round(stddev, 4),
        "edge_variance": round(edges, 4),
        "crop_quality": quality,
        "lighting": "low_luminance_proxy" if mean < 55.0 else "unknown",
        "blur": edges < 80.0,
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    archive = args.archive.resolve()
    staging_root = args.staging_root.resolve()
    successor = args.successor_manifest.resolve()
    labels_path = args.labels.resolve()
    for path in (archive, staging_root, successor, labels_path):
        if not path.exists():
            raise FileNotFoundError(path)
    archive_sha = sha256_file(archive)
    if archive_sha.lower() != args.expected_archive_sha256.lower():
        raise RuntimeError("MIO archive SHA256 mismatch")
    successor_sha = sha256_file(successor)
    if successor_sha.lower() != args.expected_successor_sha256.lower():
        raise RuntimeError("Stage72 successor SHA256 mismatch")
    labels_sha = sha256_file(labels_path)
    if labels_sha.lower() != args.expected_labels_sha256.lower():
        raise RuntimeError("vehicle_labels.v2 SHA256 mismatch")
    labels_contract = json.loads(labels_path.read_text(encoding="utf-8"))
    missing_contract = sorted(set(SOURCE_TO_BODY.values()) - set(labels_contract["body_types"]))
    if missing_contract:
        raise RuntimeError(f"MIO mapping outside v2 contract: {missing_contract}")

    official, readme_sha = load_source_evidence(archive)
    base_exact, base_tree, successor_rows = read_successor(successor)
    candidate_exact: set[str] = set()
    candidate_tree = HammingBKTree()
    accepted: list[dict[str, str]] = []
    counters: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    low_luminance_counts: Counter[str] = Counter()

    for image_path in sorted(staging_root.glob("mio_*.jpg")):
        counters["staging_images_seen"] += 1
        image_id = image_path.stem.removeprefix("mio_")
        source_label = official.get(image_id, "")
        body_type = SOURCE_TO_BODY.get(source_label)
        if body_type is None:
            counters[f"excluded_official_class:{source_label or 'missing'}"] += 1
            continue
        try:
            digest, dhash, metrics = inspect(image_path)
        except Exception as exc:
            counters[f"rejected_decode_or_quality:{exc}"] += 1
            continue
        if digest in base_exact or digest in candidate_exact:
            counters["rejected_exact_duplicate"] += 1
            continue
        base_match = base_tree.find(dhash, args.near_duplicate_hamming)
        if base_match is not None:
            counters["rejected_stage72_near_duplicate"] += 1
            continue
        candidate_match = candidate_tree.find(dhash, args.near_duplicate_hamming)
        if candidate_match is not None:
            counters["rejected_mio_near_duplicate"] += 1
            if candidate_match.get("body_type") != body_type:
                counters["rejected_mio_near_duplicate_label_conflict"] += 1
            continue

        row = {
            "image_path": str(image_path),
            "split": "train",
            "body_type": body_type,
            "body_type_supervised": "true",
            "coarse_body_family": "truck" if body_type in {"pickup", "truck", "heavy_truck"} else "",
            "color": "unknown",
            "color_supervised": "false",
            "source_dataset": "MIO-TCD-Classification-2017",
            "source_label": source_label,
            "source_license": "CC-BY-NC-SA-4.0",
            "license_train_eligible": "true",
            "review_status": "approved_research_only",
            "review_method": "official_gt_train+decode+sha256+dhash_cross_source_dedup",
            "camera_id": "unknown",
            "video_id": "unknown",
            "track_group": f"mio_train_image_{image_id}",
            "source_frame_id": image_id,
            "crop_quality": str(metrics["crop_quality"]),
            "viewpoint": "unknown",
            "lighting": str(metrics["lighting"]),
            "night": "unknown",
            "occluded": "unknown",
            "truncated": "unknown",
            "blur": str(metrics["blur"]).lower(),
            "width": str(metrics["width"]),
            "height": str(metrics["height"]),
            "gray_mean": str(metrics["gray_mean"]),
            "gray_stddev": str(metrics["gray_stddev"]),
            "edge_variance": str(metrics["edge_variance"]),
            "sha256": digest,
            "dhash64": f"{dhash:016x}",
        }
        accepted.append(row)
        candidate_exact.add(digest)
        candidate_tree.add(dhash, row)
        class_counts[body_type] += 1
        if metrics["lighting"] == "low_luminance_proxy":
            low_luminance_counts[body_type] += 1

    failures: list[str] = []
    if len(accepted) < args.minimum_rows:
        failures.append(f"minimum rows: {len(accepted)} < {args.minimum_rows}")
    for body_type, minimum in ((name, args.minimum_per_class) for name in SOURCE_TO_BODY.values()):
        if class_counts[body_type] < minimum:
            failures.append(f"minimum {body_type}: {class_counts[body_type]} < {minimum}")
    if any(row["split"] != "train" for row in accepted):
        failures.append("non-train split generated")

    report: dict[str, object] = {
        "schema_version": "stage82-mio-train-only-manifest-v1",
        "status": "pass" if not failures else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "archive": str(archive), "archive_sha256": archive_sha,
            "readme_sha256": readme_sha, "license": "CC-BY-NC-SA-4.0",
            "restriction": "research-only; non-commercial; attribution and share-alike required",
            "staging_root": str(staging_root),
        },
        "successor_dedup_reference": {
            "manifest": str(successor), "sha256": successor_sha, "rows": successor_rows,
            "near_duplicate_hamming": args.near_duplicate_hamming,
        },
        "labels": {"path": str(labels_path), "sha256": labels_sha, "mapping": SOURCE_TO_BODY},
        "output": {
            "rows": len(accepted), "splits": {"train": len(accepted)},
            "body_counts": dict(sorted(class_counts.items())),
            "low_luminance_proxy_counts": dict(sorted(low_luminance_counts.items())),
            "counters": dict(sorted(counters.items())),
        },
        "policy": {
            "official_test_image_payloads_read": 0,
            "validation_rows_created": 0,
            "test_rows_created": 0,
            "camera_track_identity_unavailable": True,
            "identity_risk_mitigated_by_train_only": True,
            "low_luminance_is_proxy_not_verified_night_truth": True,
            "color_labels_fabricated": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "eligibility": "research-only_non-deployable",
        },
        "failures": failures,
    }
    return {"report": report, "rows": accepted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--successor-manifest", type=Path, required=True)
    parser.add_argument("--expected-successor-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    parser.add_argument("--minimum-rows", type=int, default=25000)
    parser.add_argument("--minimum-per-class", type=int, default=3000)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage82 evidence")
    if not 0 <= args.near_duplicate_hamming <= 7:
        raise ValueError("near-duplicate radius must be between 0 and 7")

    result = build(args)
    report = result["report"]
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "pass":
        print(json.dumps({"status": "fail", "failures": report["failures"]}, ensure_ascii=False))
        raise SystemExit(1)
    with args.output_manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(result["rows"])
    report["output"]["manifest"] = str(args.output_manifest.resolve())
    report["output"]["manifest_sha256"] = sha256_file(args.output_manifest)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "rows": report["output"]["rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
