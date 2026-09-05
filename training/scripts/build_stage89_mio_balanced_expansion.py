#!/usr/bin/env python3
"""Build a balanced, train-only MIO-TCD expansion from the official archive.

Only official train image payloads are decoded. Existing MIO identities in the
base manifest are excluded, and all accepted samples are exact/perceptual
deduplicated against the complete active training lineage.
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
DEFAULT_ADDITIONS = {
    "heavy_truck": 3800,
    "bus": 2300,
    "pickup": 2300,
    "truck": 650,
    "van": 2400,
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


def content_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_sha256") or row.get("sha256") or row.get("image_sha256") or "").strip().lower()


def perceptual_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_dhash64") or row.get("dhash64") or "").strip().lower()


def load_base(path: Path) -> tuple[set[str], HammingBKTree, set[str], int]:
    exact: set[str] = set()
    tree = HammingBKTree()
    mio_ids: set[str] = set()
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
            if row.get("source_dataset") == "MIO-TCD-Classification-2017":
                source_id = str(row.get("source_frame_id") or "").strip()
                if source_id:
                    mio_ids.add(source_id)
    return exact, tree, mio_ids, rows


def load_source_evidence(opened: tarfile.TarFile) -> tuple[dict[str, str], str]:
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


def deterministic_rank(image_id: str, source_label: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{source_label}|{image_id}".encode("utf-8")).hexdigest()


def train_member_name(image_id: str, source_label: str) -> str:
    return f"train/{source_label}/{image_id}.jpg"


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


def inspect_payload(payload: bytes) -> tuple[str, int, dict[str, object]]:
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


def validate_targets(raw: dict[str, object]) -> dict[str, int]:
    expected = set(SOURCE_TO_BODY.values())
    if set(raw) != expected:
        raise ValueError(f"target keys must equal {sorted(expected)}")
    targets = {key: int(value) for key, value in raw.items()}
    if any(value < 0 for value in targets.values()) or sum(targets.values()) == 0:
        raise ValueError("target additions must be non-negative with a positive total")
    return targets


def build(args: argparse.Namespace) -> dict[str, object]:
    archive = args.archive.resolve()
    base_manifest = args.base_manifest.resolve()
    labels_path = args.labels.resolve()
    output_images = args.output_images.resolve()
    for path in (archive, base_manifest, labels_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    actual_hashes = {
        "archive": sha256_file(archive),
        "base_manifest": sha256_file(base_manifest),
        "labels": sha256_file(labels_path),
    }
    expected_hashes = {
        "archive": args.expected_archive_sha256.lower(),
        "base_manifest": args.expected_base_manifest_sha256.lower(),
        "labels": args.expected_labels_sha256.lower(),
    }
    if actual_hashes != expected_hashes:
        raise RuntimeError(f"pinned input SHA256 mismatch: actual={actual_hashes}")
    labels_contract = json.loads(labels_path.read_text(encoding="utf-8"))
    missing_contract = sorted(set(SOURCE_TO_BODY.values()) - set(labels_contract["body_types"]))
    if missing_contract:
        raise RuntimeError(f"MIO mapping outside v2 contract: {missing_contract}")
    targets = validate_targets(json.loads(args.target_additions_json))
    if output_images.exists() and any(output_images.iterdir()):
        raise FileExistsError("refusing to use non-empty Stage89 image directory")
    output_images.mkdir(parents=True, exist_ok=True)

    base_exact, base_tree, prior_mio_ids, base_rows = load_base(base_manifest)
    candidate_exact: set[str] = set()
    candidate_tree = HammingBKTree()
    accepted: list[dict[str, str]] = []
    accepted_counts: Counter[str] = Counter()
    counters: Counter[str] = Counter()
    low_luminance: Counter[str] = Counter()
    official_counts: Counter[str] = Counter()

    with tarfile.open(archive, "r:") as opened:
        official, readme_sha = load_source_evidence(opened)
        train_members = {
            member.name: member
            for member in opened.getmembers()
            if member.isfile() and member.name.startswith("train/")
        }
        candidates: dict[str, list[tuple[str, str]]] = {body: [] for body in targets}
        for image_id, source_label in official.items():
            body_type = SOURCE_TO_BODY.get(source_label)
            if body_type is None:
                continue
            official_counts[body_type] += 1
            if image_id in prior_mio_ids:
                counters[f"excluded_existing_identity:{body_type}"] += 1
                continue
            candidates[body_type].append((image_id, source_label))
        for body_type in candidates:
            candidates[body_type].sort(key=lambda item: deterministic_rank(item[0], item[1], args.seed))

        for body_type in sorted(targets):
            target = targets[body_type]
            for image_id, source_label in candidates[body_type]:
                if accepted_counts[body_type] >= target:
                    break
                member_name = train_member_name(image_id, source_label)
                member = train_members.get(member_name)
                if member is None:
                    counters[f"rejected_missing_train_member:{body_type}"] += 1
                    continue
                if not member.isfile() or member.name != member_name:
                    counters[f"rejected_unsafe_member:{body_type}"] += 1
                    continue
                extracted = opened.extractfile(member)
                if extracted is None:
                    counters[f"rejected_unreadable_member:{body_type}"] += 1
                    continue
                payload = extracted.read()
                counters[f"train_payloads_read:{body_type}"] += 1
                try:
                    digest, dhash, metrics = inspect_payload(payload)
                except Exception as exc:
                    counters[f"rejected_decode_or_quality:{body_type}:{exc}"] += 1
                    continue
                if digest in base_exact or digest in candidate_exact:
                    counters[f"rejected_exact_duplicate:{body_type}"] += 1
                    continue
                if base_tree.find(dhash, args.near_duplicate_hamming) is not None:
                    counters[f"rejected_base_near_duplicate:{body_type}"] += 1
                    continue
                near = candidate_tree.find(dhash, args.near_duplicate_hamming)
                if near is not None:
                    counters[f"rejected_candidate_near_duplicate:{body_type}"] += 1
                    if near.get("body_type") != body_type:
                        counters["rejected_candidate_near_duplicate_label_conflict"] += 1
                    continue

                destination = output_images / f"mio_{image_id}.jpg"
                destination.write_bytes(payload)
                row = {
                    "image_path": str(destination),
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
                    "review_method": "official_gt_train+decode+sha256+dhash_active_lineage_dedup+balanced_cap",
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
                accepted_counts[body_type] += 1
                candidate_exact.add(digest)
                candidate_tree.add(dhash, row)
                if metrics["lighting"] == "low_luminance_proxy":
                    low_luminance[body_type] += 1

    failures = [
        f"target {body_type}: {accepted_counts[body_type]} < {target}"
        for body_type, target in sorted(targets.items())
        if accepted_counts[body_type] < target
    ]
    report = {
        "schema_version": "stage89-mio-balanced-expansion-v1",
        "status": "pass" if not failures else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "archive": str(archive), "archive_sha256": actual_hashes["archive"],
            "readme_sha256": readme_sha, "license": "CC-BY-NC-SA-4.0",
            "restriction": "research-only; non-commercial; attribution and share-alike required",
            "official_train_counts": dict(sorted(official_counts.items())),
        },
        "base_dedup_reference": {
            "manifest": str(base_manifest), "sha256": actual_hashes["base_manifest"],
            "rows": base_rows, "existing_mio_identities": len(prior_mio_ids),
            "near_duplicate_hamming": args.near_duplicate_hamming,
        },
        "labels": {"path": str(labels_path), "sha256": actual_hashes["labels"], "mapping": SOURCE_TO_BODY},
        "selection": {"seed": args.seed, "target_additions": targets, "pickup_is_strictly_capped": True},
        "output": {
            "rows": len(accepted), "splits": {"train": len(accepted)},
            "body_counts": dict(sorted(accepted_counts.items())),
            "low_luminance_proxy_counts": dict(sorted(low_luminance.items())),
            "counters": dict(sorted(counters.items())), "image_root": str(output_images),
        },
        "policy": {
            "official_test_image_payloads_read": 0,
            "validation_rows_created": 0, "test_rows_created": 0,
            "identity_risk_mitigated_by_train_only": True,
            "low_luminance_is_proxy_not_verified_night_truth": True,
            "color_labels_fabricated": False, "frozen_video_used": False,
            "production_model_modified": False, "deployment_performed": False,
            "eligibility": "research-only_non-deployable",
        },
        "failures": failures,
    }
    return {"report": report, "rows": accepted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--expected-base-manifest-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--output-images", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--target-additions-json", default=json.dumps(DEFAULT_ADDITIONS))
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    parser.add_argument("--seed", default="stage89-mio-balanced-20260830")
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage89 evidence")
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
