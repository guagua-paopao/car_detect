#!/usr/bin/env python3
"""Build an aggressively deduplicated, train-only L-VAD night consistency pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageFilter, ImageStat


CLASS_MAPPING = {
    0: ("exclude", "motorcycle"),
    1: ("car", "mixed_car_family"),
    2: ("truck", "mixed_truck_bus_family"),
}
FIELDS = [
    "image_path", "split", "body_type", "body_type_supervised", "coarse_body_family",
    "color", "color_supervised", "source_dataset", "source_label", "source_class_id",
    "source_license", "license_train_eligible", "review_status", "review_method",
    "camera_id", "video_id", "track_group", "source_frame_id", "source_box_index",
    "crop_quality", "viewpoint", "lighting", "night", "occluded", "truncated",
    "blur", "width", "height", "gray_mean", "gray_stddev", "edge_variance",
    "sha256", "dhash64", "source_frame_sha256", "source_frame_dhash64",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class BKNode:
    def __init__(self, value: int, row: dict[str, str]):
        self.value = value
        self.row = row
        self.children: dict[int, BKNode] = {}


class HammingBKTree:
    def __init__(self) -> None:
        self.root: BKNode | None = None

    def add(self, value: int, row: dict[str, str]) -> None:
        if self.root is None:
            self.root = BKNode(value, row)
            return
        node = self.root
        while True:
            distance = (value ^ node.value).bit_count()
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKNode(value, row)
                return
            node = child

    def find(self, value: int, radius: int) -> dict[str, str] | None:
        if self.root is None:
            return None
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = (value ^ node.value).bit_count()
            if distance <= radius:
                return node.row
            low, high = distance - radius, distance + radius
            pending.extend(child for edge, child in node.children.items() if low <= edge <= high)
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_sha256") or row.get("sha256") or row.get("image_sha256") or "").strip().lower()


def row_dhash(row: dict[str, str]) -> str:
    return str(row.get("crop_dhash64") or row.get("dhash64") or "").strip().lower()


def load_bases(paths: list[Path]) -> tuple[set[str], HammingBKTree, int]:
    exact: set[str] = set()
    tree = HammingBKTree()
    rows = 0
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                digest = row_hash(row)
                if digest:
                    exact.add(digest)
                value = row_dhash(row)
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


def frame_identity(stem: str) -> tuple[str, int]:
    base = re.sub(r"_jpg\.rf\.[0-9a-f]+(?:\(\d+\))?$", "", stem, flags=re.IGNORECASE)
    match = re.match(r"^(.*?)(\d+)$", base)
    if not match:
        raise ValueError(f"unparseable temporal frame id: {stem}")
    return match.group(1).rstrip("_"), int(match.group(2))


def temporal_keep(last_by_sequence: dict[str, int], sequence: str, index: int, minimum_gap: int) -> bool:
    previous = last_by_sequence.get(sequence)
    if previous is not None and index - previous < minimum_gap:
        return False
    last_by_sequence[sequence] = index
    return True


def parse_yolo(text: str) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(f"line {line_number}: invalid field count")
        class_id = int(fields[0])
        values = tuple(float(value) for value in fields[1:])
        if class_id not in CLASS_MAPPING:
            raise ValueError(f"line {line_number}: out-of-contract class {class_id}")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"line {line_number}: non-finite box")
        x, y, width, height = values
        if not (0 < width <= 1 and 0 < height <= 1 and 0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"line {line_number}: invalid normalized box")
        if x - width / 2 < -1e-3 or x + width / 2 > 1 + 1e-3:
            raise ValueError(f"line {line_number}: horizontal box outside image")
        if y - height / 2 < -1e-3 or y + height / 2 > 1 + 1e-3:
            raise ValueError(f"line {line_number}: vertical box outside image")
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
    stats = ImageStat.Stat(image.convert("L"))
    mean = float(stats.mean[0])
    stddev = float(stats.stddev[0])
    edges = edge_variance(image)
    usable = min(width, height) >= 16 and stddev >= 5 and edges >= 10 and 2 <= mean <= 253
    quality = "good" if min(width, height) >= 64 and stddev >= 14 and edges >= 45 else "small_or_hard"
    return {
        "width": width, "height": height, "gray_mean": round(mean, 4),
        "gray_stddev": round(stddev, 4), "edge_variance": round(edges, 4),
        "usable": usable, "crop_quality": quality, "blur": edges < 45,
    }


def encode_jpeg(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=95, optimize=True)
    return output.getvalue()


def build(args: argparse.Namespace) -> dict[str, object]:
    archive = args.archive.resolve()
    audit_report = args.audit_report.resolve()
    evidence = args.source_evidence.resolve()
    bases = [path.resolve() for path in args.base_manifest]
    output_images = args.output_images.resolve()
    for path in [archive, audit_report, evidence, *bases]:
        if not path.is_file():
            raise FileNotFoundError(path)
    actual = {
        "archive": sha256_file(archive), "audit_report": sha256_file(audit_report),
        "source_evidence": sha256_file(evidence),
        "base_manifests": [sha256_file(path) for path in bases],
    }
    expected = {
        "archive": args.expected_archive_sha256.lower(),
        "audit_report": args.expected_audit_report_sha256.lower(),
        "source_evidence": args.expected_source_evidence_sha256.lower(),
        "base_manifests": [value.lower() for value in args.expected_base_manifest_sha256],
    }
    if actual != expected:
        raise RuntimeError(f"pinned input SHA mismatch: {actual}")
    audit = json.loads(audit_report.read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or audit.get("schema_version") != "stage92-lvad-archive-audit-v2":
        raise RuntimeError("Stage92 R2 archive audit did not pass")
    access = audit.get("access_policy", {})
    if access.get("validation_payloads_read") != 0 or access.get("test_payloads_read") != 0:
        raise RuntimeError("Stage92 audit accessed held-out payloads")
    if output_images.exists() and any(output_images.iterdir()):
        raise FileExistsError("refusing to use non-empty Stage92 image directory")
    output_images.mkdir(parents=True, exist_ok=True)

    base_exact, base_tree, base_rows = load_bases(bases)
    candidate_exact: set[str] = set()
    candidate_tree = HammingBKTree()
    frame_exact: set[str] = set()
    frame_tree = HammingBKTree()
    last_by_sequence: dict[str, int] = {}
    rows: list[dict[str, str]] = []
    counters: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()

    with zipfile.ZipFile(archive) as opened:
        names = opened.namelist()
        roots = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
        if len(roots) != 1:
            raise RuntimeError("Stage92 archive root is ambiguous")
        root = next(iter(roots))
        images = {
            Path(name).stem: name for name in names
            if name.startswith(f"{root}/train/images/") and Path(name).suffix.lower() in IMAGE_SUFFIXES
        }
        labels = {
            Path(name).stem: name for name in names
            if name.startswith(f"{root}/train/labels/") and Path(name).suffix.lower() == ".txt"
        }
        if len(images) != 10934 or len(labels) != 10919 or not set(labels).issubset(set(images)):
            raise RuntimeError("Stage92 train structure drift")
        ordered = sorted(images, key=lambda stem: frame_identity(stem))
        for stem in ordered:
            sequence, frame_index = frame_identity(stem)
            if not temporal_keep(last_by_sequence, sequence, frame_index, args.minimum_temporal_gap):
                counters["rejected_temporal_near_frame"] += 1
                continue
            image_payload = opened.read(images[stem])
            label_payload = opened.read(labels[stem]) if stem in labels else b""
            counters["train_image_payloads_read"] += 1
            counters["train_label_payloads_read"] += int(stem in labels)
            try:
                boxes = parse_yolo(label_payload.decode("utf-8-sig", errors="strict"))
            except ValueError as exc:
                counters[f"rejected_annotation_contract:{exc}"] += 1
                continue
            frame_sha = hashlib.sha256(image_payload).hexdigest()
            if frame_sha in frame_exact:
                counters["rejected_exact_duplicate_frame"] += 1
                continue
            with Image.open(io.BytesIO(image_payload)) as source:
                source.load()
                frame = source.convert("RGB")
            frame_dhash = dhash64(frame)
            if frame_tree.find(frame_dhash, args.frame_near_duplicate_hamming) is not None:
                counters["rejected_perceptual_duplicate_frame"] += 1
                continue
            frame_exact.add(frame_sha)
            frame_tree.add(frame_dhash, {"source_frame_id": stem})
            for box_index, (class_id, x, y, width, height) in enumerate(boxes):
                family, source_label = CLASS_MAPPING[class_id]
                if family == "exclude":
                    counters["excluded_motorcycle_box"] += 1
                    continue
                try:
                    crop = crop_box(frame, (x, y, width, height), args.padding)
                    metrics = inspect_crop(crop)
                except Exception as exc:
                    counters[f"rejected_crop:{exc}"] += 1
                    continue
                if not metrics["usable"]:
                    counters[f"rejected_unusable_crop:{family}"] += 1
                    continue
                crop_dhash = dhash64(crop)
                encoded = encode_jpeg(crop)
                digest = hashlib.sha256(encoded).hexdigest()
                if digest in base_exact or digest in candidate_exact:
                    counters["rejected_exact_duplicate_crop"] += 1
                    continue
                if base_tree.find(crop_dhash, args.crop_near_duplicate_hamming) is not None:
                    counters["rejected_base_near_duplicate_crop"] += 1
                    continue
                if candidate_tree.find(crop_dhash, args.crop_near_duplicate_hamming) is not None:
                    counters["rejected_candidate_near_duplicate_crop"] += 1
                    continue
                destination = output_images / f"lvad_{sequence}_{frame_index:06d}_{box_index:03d}.jpg"
                destination.write_bytes(encoded)
                row = {
                    "image_path": str(destination), "split": "train", "body_type": "unknown",
                    "body_type_supervised": "false", "coarse_body_family": family,
                    "color": "unknown", "color_supervised": "false", "source_dataset": "L-VAD-v2",
                    "source_label": source_label, "source_class_id": str(class_id),
                    "source_license": "CC-BY-4.0", "license_train_eligible": "true",
                    "review_status": "approved_unknown_safe_night_consistency_only",
                    "review_method": (
                        f"official_train_bbox+temporal{args.minimum_temporal_gap}"
                        f"+frame_dhash_h{args.frame_near_duplicate_hamming}"
                        f"+crop_sha256+dhash_h{args.crop_near_duplicate_hamming}+quality_gate"
                    ),
                    "camera_id": "lvad_mobile_roadside_unknown", "video_id": f"lvad_{sequence}",
                    "track_group": f"lvad_{sequence}_{frame_index // max(1, args.track_window_frames)}",
                    "source_frame_id": stem, "source_box_index": str(box_index),
                    "crop_quality": str(metrics["crop_quality"]), "viewpoint": "roadside_45_or_90_unknown",
                    "lighting": "official_night_low_light", "night": "true", "occluded": "unknown",
                    "truncated": "unknown", "blur": str(metrics["blur"]).lower(),
                    "width": str(metrics["width"]), "height": str(metrics["height"]),
                    "gray_mean": str(metrics["gray_mean"]), "gray_stddev": str(metrics["gray_stddev"]),
                    "edge_variance": str(metrics["edge_variance"]), "sha256": digest,
                    "dhash64": f"{crop_dhash:016x}", "source_frame_sha256": frame_sha,
                    "source_frame_dhash64": f"{frame_dhash:016x}",
                }
                rows.append(row)
                candidate_exact.add(digest)
                candidate_tree.add(crop_dhash, row)
                family_counts[family] += 1

    failures: list[str] = []
    if len(rows) < args.minimum_rows:
        failures.append(f"rows {len(rows)} < {args.minimum_rows}")
    if family_counts["car"] < args.minimum_car_rows:
        failures.append(f"car rows {family_counts['car']} < {args.minimum_car_rows}")
    if family_counts["truck"] < args.minimum_truck_rows:
        failures.append(f"truck rows {family_counts['truck']} < {args.minimum_truck_rows}")
    report = {
        "schema_version": "stage92-lvad-night-pool-v1",
        "status": "pass" if not failures else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "archive": str(archive), "archive_sha256": actual["archive"],
            "audit_report": str(audit_report), "audit_report_sha256": actual["audit_report"],
            "source_evidence": str(evidence), "source_evidence_sha256": actual["source_evidence"],
            "base_manifests": [str(path) for path in bases],
            "base_manifest_sha256": actual["base_manifests"], "base_rows": base_rows,
        },
        "selection": {
            "minimum_temporal_gap": args.minimum_temporal_gap,
            "track_window_frames": args.track_window_frames,
            "frame_near_duplicate_hamming": args.frame_near_duplicate_hamming,
            "crop_near_duplicate_hamming": args.crop_near_duplicate_hamming,
            "padding": args.padding,
        },
        "output": {
            "rows": len(rows), "coarse_family_counts": dict(sorted(family_counts.items())),
            "counters": dict(sorted(counters.items())), "images": str(output_images),
        },
        "policy": {
            "all_outputs_train_only": True, "exact_body_supervised_rows": 0,
            "color_supervised_rows": 0, "motorcycle_rows": 0,
            "source_validation_payloads_read": 0, "source_test_payloads_read": 0,
            "fine_body_labels_fabricated": False, "color_labels_fabricated": False,
            "verified_night_source": True, "frozen_video_used": False,
            "production_model_modified": False, "deployment_performed": False,
        },
        "failures": failures,
    }
    return {"report": report, "rows": rows}


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
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--expected-source-evidence-sha256", required=True)
    parser.add_argument("--base-manifest", type=Path, action="append", required=True)
    parser.add_argument("--expected-base-manifest-sha256", action="append", required=True)
    parser.add_argument("--output-images", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--minimum-temporal-gap", type=int, default=15)
    parser.add_argument("--track-window-frames", type=int, default=30)
    parser.add_argument("--frame-near-duplicate-hamming", type=int, default=4)
    parser.add_argument("--crop-near-duplicate-hamming", type=int, default=4)
    parser.add_argument("--padding", type=float, default=0.05)
    parser.add_argument("--minimum-rows", type=int, default=1000)
    parser.add_argument("--minimum-car-rows", type=int, default=800)
    parser.add_argument("--minimum-truck-rows", type=int, default=100)
    args = parser.parse_args()
    if len(args.base_manifest) != len(args.expected_base_manifest_sha256):
        raise ValueError("base manifest/hash counts differ")
    if args.minimum_temporal_gap < 1 or not 0 <= args.frame_near_duplicate_hamming <= 7:
        raise ValueError("invalid temporal or frame dedup policy")
    if not 0 <= args.crop_near_duplicate_hamming <= 7 or not 0 <= args.padding <= 0.5:
        raise ValueError("invalid crop dedup or padding policy")
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage92 pool evidence")
    result = build(args)
    report = result["report"]
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "pass":
        print(json.dumps({"status": "fail", "failures": report["failures"]}, ensure_ascii=False))
        raise SystemExit(1)
    write_manifest(args.output_manifest, result["rows"])
    report["output"].update({
        "manifest": str(args.output_manifest.resolve()),
        "manifest_sha256": sha256_file(args.output_manifest),
    })
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "rows": len(result["rows"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
