#!/usr/bin/env python3
"""Create leakage-safe InaTRC train crops for supervised truck and unlabeled consistency use."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageFilter, ImageStat

from build_stage70_specialist_manifests import HammingBKTree


CLASS_MAPPING = {
    0: ("unknown", False, "mixed_nontruck_unknown"),
    1: ("truck", True, "two_axle_truck"),
    2: ("heavy_truck", True, "three_axle_truck"),
    3: ("heavy_truck", True, "four_axle_truck"),
    4: ("heavy_truck", True, "five_plus_axle_truck"),
}
FIELDS = [
    "image_path", "split", "body_type", "body_type_supervised", "coarse_body_family",
    "color", "color_supervised", "source_dataset", "source_label", "source_license",
    "license_train_eligible", "review_status", "review_method", "camera_id", "video_id",
    "track_group", "source_frame_id", "source_box_index", "crop_quality", "viewpoint",
    "lighting", "night", "occluded", "truncated", "blur", "width", "height",
    "gray_mean", "gray_stddev", "edge_variance", "sha256", "dhash64",
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


def load_base(path: Path) -> tuple[set[str], HammingBKTree, int]:
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


def parse_yolo(text: str) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(f"line {line_number}: invalid YOLO field count")
        class_id = int(fields[0])
        values = tuple(float(value) for value in fields[1:])
        if class_id not in CLASS_MAPPING:
            raise ValueError(f"line {line_number}: unknown class {class_id}")
        x, y, width, height = values
        if not (0 < width <= 1 and 0 < height <= 1 and 0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"line {line_number}: invalid normalized box")
        boxes.append((class_id, x, y, width, height))
    return boxes


def crop_box(image: Image.Image, box: tuple[float, float, float, float], padding: float) -> Image.Image:
    x, y, width, height = box
    image_width, image_height = image.size
    width *= 1 + 2 * padding
    height *= 1 + 2 * padding
    left = max(0, int(round((x - width / 2) * image_width)))
    top = max(0, int(round((y - height / 2) * image_height)))
    right = min(image_width, int(round((x + width / 2) * image_width)))
    bottom = min(image_height, int(round((y + height / 2) * image_height)))
    if right - left < 8 or bottom - top < 8:
        raise ValueError("crop_dimension_below_8")
    return image.crop((left, top, right, bottom)).convert("RGB")


def inspect_crop(image: Image.Image) -> dict[str, object]:
    width, height = image.size
    gray = image.convert("L")
    stats = ImageStat.Stat(gray)
    mean = float(stats.mean[0])
    stddev = float(stats.stddev[0])
    edges = edge_variance(image)
    quality_supervised = min(width, height) >= 24 and stddev >= 8 and edges >= 20 and 6 <= mean <= 249
    quality = "good" if min(width, height) >= 64 and stddev >= 18 and edges >= 60 else "small_or_hard"
    return {
        "width": width, "height": height, "gray_mean": round(mean, 4),
        "gray_stddev": round(stddev, 4), "edge_variance": round(edges, 4),
        "quality_supervised": quality_supervised, "crop_quality": quality,
        "lighting": "low_luminance_proxy" if mean < 55 else "unknown",
        "blur": edges < 60,
    }


def encode_jpeg(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=95, optimize=True)
    return output.getvalue()


def build(args: argparse.Namespace) -> dict[str, object]:
    archive = args.archive.resolve()
    audit_report = args.audit_report.resolve()
    base_manifest = args.base_manifest.resolve()
    labels = args.labels.resolve()
    output_images = args.output_images.resolve()
    for path in (archive, audit_report, base_manifest, labels):
        if not path.is_file():
            raise FileNotFoundError(path)
    actual = {
        "archive": sha256_file(archive), "audit_report": sha256_file(audit_report),
        "base_manifest": sha256_file(base_manifest), "labels": sha256_file(labels),
    }
    expected = {
        "archive": args.expected_archive_sha256.lower(),
        "audit_report": args.expected_audit_report_sha256.lower(),
        "base_manifest": args.expected_base_manifest_sha256.lower(),
        "labels": args.expected_labels_sha256.lower(),
    }
    if actual != expected:
        raise RuntimeError(f"pinned input SHA mismatch: {actual}")
    audit = json.loads(audit_report.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise RuntimeError("Stage91 archive audit did not pass")
    access = audit.get("access_policy", {})
    if access.get("validation_payloads_read") != 0 or access.get("test_payloads_read") != 0:
        raise RuntimeError("Stage91 audit accessed held-out payloads")
    contract = json.loads(labels.read_text(encoding="utf-8"))
    if not {"unknown", "truck", "heavy_truck"}.issubset(set(contract["body_types"])):
        raise RuntimeError("vehicle labels contract cannot represent Stage91 mapping")
    if output_images.exists() and any(output_images.iterdir()):
        raise FileExistsError("refusing to use non-empty Stage91 crop directory")
    output_images.mkdir(parents=True, exist_ok=True)

    base_exact, base_tree, base_rows = load_base(base_manifest)
    candidate_exact: set[str] = set()
    candidate_tree = HammingBKTree()
    rows: list[dict[str, str]] = []
    counters: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    supervised_counts: Counter[str] = Counter()
    full_frame_hashes: set[str] = set()
    full_frame_dhashes: set[int] = set()

    with zipfile.ZipFile(archive) as opened:
        names = opened.namelist()
        roots = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
        if len(roots) != 1:
            raise RuntimeError("Stage91 archive root is ambiguous")
        root = next(iter(roots))
        images = {
            Path(name).stem: name for name in names
            if name.startswith(f"{root}/train/images/") and Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
        }
        labels_by_stem = {
            Path(name).stem: name for name in names
            if name.startswith(f"{root}/train/labels/") and Path(name).suffix.lower() == ".txt" and not name.endswith("classes.txt")
        }
        if set(images) != set(labels_by_stem) or len(images) != 2600:
            raise RuntimeError("Stage91 train image/label pairing drift")

        for frame_stem in sorted(images):
            image_payload = opened.read(images[frame_stem])
            label_payload = opened.read(labels_by_stem[frame_stem])
            counters["train_image_payloads_read"] += 1
            counters["train_label_payloads_read"] += 1
            frame_sha = hashlib.sha256(image_payload).hexdigest()
            with Image.open(io.BytesIO(image_payload)) as source:
                source.load()
                frame = source.convert("RGB")
            frame_dhash = dhash64(frame)
            if frame_sha in full_frame_hashes:
                counters["rejected_exact_duplicate_frame"] += 1
                continue
            full_frame_hashes.add(frame_sha)
            if frame_dhash in full_frame_dhashes:
                counters["rejected_repeated_dhash_frame"] += 1
                continue
            full_frame_dhashes.add(frame_dhash)
            boxes = parse_yolo(label_payload.decode("utf-8-sig", errors="strict"))
            for box_index, (class_id, x, y, width, height) in enumerate(boxes):
                counters[f"source_class:{class_id}"] += 1
                try:
                    crop = crop_box(frame, (x, y, width, height), args.padding)
                    metrics = inspect_crop(crop)
                except Exception as exc:
                    counters[f"rejected_crop:{exc}"] += 1
                    continue
                body_type, class_supervised, source_label = CLASS_MAPPING[class_id]
                supervised = bool(class_supervised and metrics["quality_supervised"])
                if class_supervised and not supervised:
                    counters[f"demoted_low_quality_supervision:{body_type}"] += 1
                    body_type = "unknown"
                crop_dhash = dhash64(crop)
                encoded = encode_jpeg(crop)
                digest = hashlib.sha256(encoded).hexdigest()
                if digest in base_exact or digest in candidate_exact:
                    counters["rejected_exact_duplicate_crop"] += 1
                    continue
                if base_tree.find(crop_dhash, args.near_duplicate_hamming) is not None:
                    counters["rejected_base_near_duplicate_crop"] += 1
                    continue
                near = candidate_tree.find(crop_dhash, args.near_duplicate_hamming)
                if near is not None:
                    counters["rejected_candidate_near_duplicate_crop"] += 1
                    if near.get("source_label") != source_label:
                        counters["rejected_near_duplicate_label_conflict"] += 1
                    continue
                destination = output_images / f"inatrc_{frame_stem}_{box_index:03d}.jpg"
                destination.write_bytes(encoded)
                row = {
                    "image_path": str(destination), "split": "train", "body_type": body_type,
                    "body_type_supervised": str(supervised).lower(),
                    "coarse_body_family": "truck" if supervised else "",
                    "color": "unknown", "color_supervised": "false",
                    "source_dataset": "InaTRC-v1", "source_label": source_label,
                    "source_license": "CC-BY-4.0", "license_train_eligible": "true",
                    "review_status": "approved_official_train_bbox_quality_gate" if supervised else "approved_unlabeled_consistency_only",
                    "review_method": "official_train_yolo+decode+quality_gate+sha256+dhash_active_lineage_dedup",
                    "camera_id": "inatrc_toll_cctv_unknown", "video_id": "inatrc_official_train",
                    "track_group": f"inatrc_train_frame_{frame_stem}", "source_frame_id": frame_stem,
                    "source_box_index": str(box_index), "crop_quality": str(metrics["crop_quality"]),
                    "viewpoint": "overhead_toll_cctv", "lighting": str(metrics["lighting"]),
                    "night": "unknown", "occluded": "unknown", "truncated": "unknown",
                    "blur": str(metrics["blur"]).lower(), "width": str(metrics["width"]),
                    "height": str(metrics["height"]), "gray_mean": str(metrics["gray_mean"]),
                    "gray_stddev": str(metrics["gray_stddev"]), "edge_variance": str(metrics["edge_variance"]),
                    "sha256": digest, "dhash64": f"{crop_dhash:016x}",
                }
                rows.append(row)
                candidate_exact.add(digest)
                candidate_tree.add(crop_dhash, row)
                class_counts[body_type] += 1
                if supervised:
                    supervised_counts[body_type] += 1

    supervised_rows = [row for row in rows if row["body_type_supervised"] == "true"]
    unlabeled_rows = [row for row in rows if row["body_type_supervised"] != "true"]
    failures: list[str] = []
    if len(supervised_rows) < args.minimum_supervised_rows:
        failures.append(f"supervised rows {len(supervised_rows)} < {args.minimum_supervised_rows}")
    if supervised_counts["truck"] < args.minimum_truck_rows:
        failures.append(f"truck rows {supervised_counts['truck']} < {args.minimum_truck_rows}")
    if supervised_counts["heavy_truck"] < args.minimum_heavy_truck_rows:
        failures.append(f"heavy_truck rows {supervised_counts['heavy_truck']} < {args.minimum_heavy_truck_rows}")
    report = {
        "schema_version": "stage91-inatrc-train-crops-v1",
        "status": "pass" if not failures else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "archive": str(archive), "archive_sha256": actual["archive"],
            "audit_report": str(audit_report), "audit_report_sha256": actual["audit_report"],
            "base_manifest": str(base_manifest), "base_manifest_sha256": actual["base_manifest"],
            "base_rows": base_rows, "labels": str(labels), "labels_sha256": actual["labels"],
        },
        "mapping": {str(key): {"body_type": value[0], "supervised_if_quality_passes": value[1], "source_label": value[2]} for key, value in CLASS_MAPPING.items()},
        "output": {
            "rows": len(rows), "supervised_rows": len(supervised_rows), "unlabeled_rows": len(unlabeled_rows),
            "body_counts": dict(sorted(class_counts.items())),
            "supervised_body_counts": dict(sorted(supervised_counts.items())),
            "counters": dict(sorted(counters.items())), "images": str(output_images),
        },
        "policy": {
            "source_validation_payloads_read": 0, "source_test_payloads_read": 0,
            "validation_rows_created": 0, "test_rows_created": 0,
            "mixed_nontruck_fine_labels_fabricated": False, "color_labels_fabricated": False,
            "verified_night_truth_claimed": False, "all_outputs_train_only": True,
            "frozen_video_used": False, "production_model_modified": False,
            "deployment_performed": False, "deployment_paused_by_user": True,
        },
        "failures": failures,
    }
    return {"report": report, "all_rows": rows, "supervised_rows": supervised_rows, "unlabeled_rows": unlabeled_rows}


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--expected-audit-report-sha256", required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--expected-base-manifest-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--output-images", type=Path, required=True)
    parser.add_argument("--output-all-manifest", type=Path, required=True)
    parser.add_argument("--output-supervised-manifest", type=Path, required=True)
    parser.add_argument("--output-unlabeled-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--padding", type=float, default=0.05)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    parser.add_argument("--minimum-supervised-rows", type=int, default=6000)
    parser.add_argument("--minimum-truck-rows", type=int, default=4000)
    parser.add_argument("--minimum-heavy-truck-rows", type=int, default=1500)
    args = parser.parse_args()
    outputs = [args.output_all_manifest, args.output_supervised_manifest, args.output_unlabeled_manifest, args.output_report]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite Stage91 crop evidence")
    if not 0 <= args.near_duplicate_hamming <= 7 or not 0 <= args.padding <= 0.5:
        raise ValueError("invalid dedup radius or crop padding")
    result = build(args)
    report = result["report"]
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "pass":
        print(json.dumps({"status": "fail", "failures": report["failures"]}, ensure_ascii=False))
        raise SystemExit(1)
    write_manifest(args.output_all_manifest, result["all_rows"])
    write_manifest(args.output_supervised_manifest, result["supervised_rows"])
    write_manifest(args.output_unlabeled_manifest, result["unlabeled_rows"])
    report["output"].update({
        "all_manifest": str(args.output_all_manifest.resolve()),
        "all_manifest_sha256": sha256_file(args.output_all_manifest),
        "supervised_manifest": str(args.output_supervised_manifest.resolve()),
        "supervised_manifest_sha256": sha256_file(args.output_supervised_manifest),
        "unlabeled_manifest": str(args.output_unlabeled_manifest.resolve()),
        "unlabeled_manifest_sha256": sha256_file(args.output_unlabeled_manifest),
    })
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "supervised": len(result["supervised_rows"]), "unlabeled": len(result["unlabeled_rows"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
