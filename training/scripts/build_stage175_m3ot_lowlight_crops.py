from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageFilter, ImageStat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_stage70_specialist_manifests import HammingBKTree  # noqa: E402


FIELDS = [
    "image_path", "split", "body_type", "body_type_supervised", "coarse_body_family",
    "color", "color_supervised", "source_dataset", "source_label", "source_license",
    "source_license_url", "license_train_eligible", "review_status", "review_method",
    "camera_id", "video_id", "track_group", "source_frame_id", "source_box_index",
    "crop_quality", "viewpoint", "lighting", "night", "low_light", "small_target",
    "occluded", "truncated", "blur", "width", "height", "gray_mean", "gray_stddev",
    "edge_variance", "sha256", "dhash64", "crop_sha256", "crop_dhash64",
    "source_bbox_area", "source_bbox_width", "source_bbox_height", "source_bbox_area_ratio",
    "adverse_supervised", "formal_train_eligible", "research_only", "deployment_eligible",
    "pseudo_label", "pseudo_label_confidence", "teacher_consensus", "teacher_audit_policy",
    "stage175_origin", "stage175_condition_source", "stage175_track_sampling_policy",
    "stage175_dedup_policy", "stage175_annotation_id", "stage175_instance_id",
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


def load_base(paths: list[Path], expected_sha256: list[str]) -> tuple[set[str], HammingBKTree, Counter[str]]:
    if len(paths) != len(expected_sha256):
        raise ValueError("base manifest and SHA256 counts differ")
    exact: set[str] = set()
    tree = HammingBKTree()
    counters: Counter[str] = Counter()
    for path, expected in zip(paths, expected_sha256):
        if sha256_file(path).lower() != expected.lower():
            raise ValueError(f"base manifest SHA256 mismatch: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                counters["base_rows"] += 1
                digest = content_hash(row)
                if digest:
                    exact.add(digest)
                    counters["base_exact_hashes"] += 1
                value = perceptual_hash(row)
                if len(value) == 16:
                    try:
                        tree.add(int(value, 16), row)
                        counters["base_perceptual_hashes"] += 1
                    except ValueError:
                        counters["base_invalid_perceptual_hashes"] += 1
    return exact, tree, counters


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


def inspect_crop(image: Image.Image) -> dict[str, object]:
    gray = image.convert("L")
    stats = ImageStat.Stat(gray)
    mean = float(stats.mean[0])
    stddev = float(stats.stddev[0])
    edges = edge_variance(image)
    usable = min(image.size) >= 20 and stddev >= 4 and edges >= 8 and 1 <= mean <= 254
    quality = "good" if min(image.size) >= 32 and stddev >= 12 and edges >= 30 else "tiny_lowlight_hard"
    return {
        "width": image.width,
        "height": image.height,
        "gray_mean": round(mean, 4),
        "gray_stddev": round(stddev, 4),
        "edge_variance": round(edges, 4),
        "usable": usable,
        "crop_quality": quality,
        "blur": edges < 30,
    }


def square_crop(image: Image.Image, bbox: list[float], context: float, minimum_side: int) -> tuple[Image.Image, bool]:
    x, y, width, height = map(float, bbox)
    if not all(math.isfinite(value) for value in (x, y, width, height)) or width <= 0 or height <= 0:
        raise ValueError("invalid_bbox")
    if x < 0 or y < 0 or x + width > image.width + 1 or y + height > image.height + 1:
        raise ValueError("out_of_bounds_bbox")
    side = max(float(minimum_side), max(width, height) * context)
    center_x = x + width / 2
    center_y = y + height / 2
    left = int(math.floor(center_x - side / 2))
    top = int(math.floor(center_y - side / 2))
    right = int(math.ceil(center_x + side / 2))
    bottom = int(math.ceil(center_y + side / 2))
    clipped = left < 0 or top < 0 or right > image.width or bottom > image.height
    left = max(0, left)
    top = max(0, top)
    right = min(image.width, right)
    bottom = min(image.height, bottom)
    if right - left < 20 or bottom - top < 20:
        raise ValueError("crop_too_small")
    return image.crop((left, top, right, bottom)), clipped


def encode_jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95, subsampling=0, optimize=False, progressive=False)
    return buffer.getvalue()


def uniformly_sample(records: list[dict[str, object]], maximum: int) -> list[dict[str, object]]:
    ordered = sorted(records, key=lambda row: (int(row["mot_frame_id"]), int(row["annotation_id"])))
    if len(ordered) <= maximum:
        return ordered
    if maximum <= 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(index * (len(ordered) - 1) / (maximum - 1)) for index in range(maximum)]
    return [ordered[index] for index in sorted(set(indices))]


def resolve_image_path(root: Path, drone: str, file_name: str) -> Path:
    parts = PurePosixPath(file_name.replace("\\", "/")).parts
    try:
        index = parts.index("train")
    except ValueError as error:
        raise ValueError(f"COCO image is not train: {file_name}") from error
    suffix = parts[index + 1 :]
    if len(suffix) != 4 or suffix[0] != drone or suffix[2] != "img1":
        raise ValueError(f"unexpected COCO image path: {file_name}")
    resolved = (root / drone / "rgb" / "train" / suffix[1] / suffix[2] / suffix[3]).resolve()
    resolved.relative_to(root.resolve())
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def build(args: argparse.Namespace) -> dict[str, object]:
    root = args.extraction_root.resolve()
    photometry_path = args.photometry_report.resolve()
    if sha256_file(photometry_path).lower() != args.expected_photometry_sha256.lower():
        raise ValueError("photometry report SHA256 mismatch")
    photometry = json.loads(photometry_path.read_text(encoding="utf-8"))
    if photometry.get("status") != "pass":
        raise ValueError("photometry audit did not pass")
    if photometry.get("validation_payload_opened") or photometry.get("test_payload_opened"):
        raise ValueError("photometry audit records validation/test access")
    conditions = {sequence: str(stats["condition"]) for sequence, stats in photometry["sequence_stats"].items()}
    allowed_conditions = set(args.conditions)
    if not allowed_conditions <= {"night", "dusk"}:
        raise ValueError("only night/dusk train conditions may be selected")

    output_images = args.output_images.resolve()
    if output_images.exists() and any(output_images.iterdir()):
        raise FileExistsError("refusing non-empty Stage175 output directory")
    output_images.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_images).free < args.minimum_free_bytes:
        raise OSError("initial free-space guard failed")

    base_exact, base_tree, counters = load_base(args.base_manifest, args.expected_base_sha256)
    candidate_exact: set[str] = set()
    candidate_tree = HammingBKTree()
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    source_images: dict[tuple[str, int], dict[str, object]] = {}
    source_annotation_counts: Counter[str] = Counter()

    for drone in ("1", "2"):
        annotation_path = root / "Annotations" / drone / "rgb" / "train_cocoformat.json"
        document = json.loads(annotation_path.read_text(encoding="utf-8"))
        if document.get("categories") != [{"id": 1, "name": "vehicle"}]:
            raise ValueError(f"unexpected categories: {annotation_path}")
        for image in document["images"]:
            sequence = PurePosixPath(str(image["file_name"])).parts[-3]
            if sequence not in conditions:
                raise ValueError(f"sequence missing from photometry: {sequence}")
            source_images[(drone, int(image["id"]))] = {
                **image,
                "sequence": sequence,
                "condition": conditions[sequence],
                "path": resolve_image_path(root, drone, str(image["file_name"])),
            }
        for annotation in document["annotations"]:
            image = source_images[(drone, int(annotation["image_id"]))]
            condition = str(image["condition"])
            source_annotation_counts[condition] += 1
            if condition not in allowed_conditions:
                counters["skipped_non_lowlight_annotation"] += 1
                continue
            x, y, width, height = map(float, annotation["bbox"])
            area = width * height
            if area < args.minimum_bbox_area or max(width, height) < args.minimum_bbox_max_side:
                counters[f"rejected_too_tiny_bbox:{condition}"] += 1
                continue
            record = {
                "drone": drone,
                "sequence": image["sequence"],
                "condition": condition,
                "path": image["path"],
                "image_id": int(annotation["image_id"]),
                "mot_frame_id": int(image["mot_frame_id"]),
                "annotation_id": int(annotation["id"]),
                "instance_id": int(annotation["instance_id"]),
                "bbox": annotation["bbox"],
                "image_width": int(image["width"]),
                "image_height": int(image["height"]),
            }
            grouped[(drone, str(image["sequence"]), int(annotation["instance_id"]))].append(record)

    selected: list[dict[str, object]] = []
    for track, records in sorted(grouped.items()):
        sampled = uniformly_sample(records, args.maximum_frames_per_track)
        selected.extend(sampled)
        counters["track_groups"] += 1
        counters["pre_dedup_selected"] += len(sampled)
        counters["temporal_frames_removed_by_track_cap"] += len(records) - len(sampled)

    by_image: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for record in selected:
        by_image[(str(record["drone"]), int(record["image_id"]))].append(record)

    rows: list[dict[str, str]] = []
    output_bytes = 0
    retained_conditions: Counter[str] = Counter()
    retained_tracks: set[str] = set()
    for key in sorted(by_image):
        records = by_image[key]
        image_path = Path(records[0]["path"])
        try:
            with Image.open(image_path) as source:
                source.load()
                frame = source.convert("RGB")
        except Exception:
            counters["rejected_invalid_source_image"] += len(records)
            continue
        counters["source_images_opened"] += 1
        for record in records:
            condition = str(record["condition"])
            try:
                crop, context_clipped = square_crop(frame, list(record["bbox"]), args.context_factor, args.minimum_crop_side)
                metrics = inspect_crop(crop)
            except Exception as error:
                counters[f"rejected_crop:{error}"] += 1
                continue
            if not metrics["usable"]:
                counters[f"rejected_unusable_crop:{condition}"] += 1
                continue
            encoded = encode_jpeg(crop)
            with Image.open(io.BytesIO(encoded)) as persisted:
                persisted.load()
                value = dhash64(persisted)
            digest = hashlib.sha256(encoded).hexdigest()
            if digest in base_exact or digest in candidate_exact:
                counters["rejected_exact_duplicate"] += 1
                continue
            if base_tree.find(value, args.near_duplicate_hamming) is not None:
                counters["rejected_base_near_duplicate"] += 1
                continue
            if candidate_tree.find(value, args.near_duplicate_hamming) is not None:
                counters["rejected_candidate_near_duplicate"] += 1
                continue
            if output_bytes + len(encoded) > args.maximum_output_bytes:
                raise OSError("output byte guard exceeded")
            if len(rows) % 250 == 0 and shutil.disk_usage(output_images).free < args.minimum_free_bytes:
                raise OSError("runtime free-space guard failed")

            frame_stem = image_path.stem
            name = f"m3ot_{record['drone']}_{record['sequence']}_{frame_stem}_{int(record['instance_id']):04d}_{int(record['annotation_id']):07d}.jpg"
            destination = output_images / name
            destination.write_bytes(encoded)
            output_bytes += len(encoded)
            x, y, box_width, box_height = map(float, record["bbox"])
            track_group = f"m3ot:{record['drone']}:{record['sequence']}:{int(record['instance_id'])}"
            row = {
                "image_path": str(destination), "split": "train", "body_type": "unknown",
                "body_type_supervised": "false", "coarse_body_family": "", "color": "unknown",
                "color_supervised": "false", "source_dataset": "M3OT", "source_label": "vehicle",
                "source_license": "CC-BY-4.0", "source_license_url": "https://creativecommons.org/licenses/by/4.0/",
                "license_train_eligible": "true", "review_status": "approved_unlabeled_domain_consistency_only",
                "review_method": "official_COCO_box+train_only+track_cap+quality_gate+sha256+dhash4_lineage",
                "camera_id": f"m3ot_drone_{record['drone']}", "video_id": f"m3ot_{record['drone']}_{record['sequence']}",
                "track_group": track_group, "source_frame_id": f"{record['sequence']}/{frame_stem}",
                "source_box_index": str(record["annotation_id"]), "crop_quality": str(metrics["crop_quality"]),
                "viewpoint": "aerial_topdown_small_target_auxiliary", "lighting": condition,
                "night": str(condition == "night").lower(), "low_light": "true", "small_target": "true",
                "occluded": "unknown", "truncated": str(context_clipped).lower(), "blur": str(metrics["blur"]).lower(),
                "width": str(metrics["width"]), "height": str(metrics["height"]),
                "gray_mean": str(metrics["gray_mean"]), "gray_stddev": str(metrics["gray_stddev"]),
                "edge_variance": str(metrics["edge_variance"]), "sha256": digest,
                "dhash64": f"{value:016x}", "crop_sha256": digest, "crop_dhash64": f"{value:016x}",
                "source_bbox_area": str(round(box_width * box_height, 4)), "source_bbox_width": str(box_width),
                "source_bbox_height": str(box_height),
                "source_bbox_area_ratio": str((box_width * box_height) / (int(record["image_width"]) * int(record["image_height"]))),
                "adverse_supervised": "false", "formal_train_eligible": "true", "research_only": "true",
                "deployment_eligible": "false", "pseudo_label": "false", "pseudo_label_confidence": "",
                "teacher_consensus": "false", "teacher_audit_policy": "pending_multi_frame_and_augmentation_consensus",
                "stage175_origin": "m3ot_official_rgb_train_lowlight_track_capped",
                "stage175_condition_source": "stage174_paired_sequence_photometric_cluster",
                "stage175_track_sampling_policy": f"uniform_max_{args.maximum_frames_per_track}_per_drone_sequence_instance",
                "stage175_dedup_policy": f"exact+dhash<={args.near_duplicate_hamming}_against_stage167_stage172_and_candidate",
                "stage175_annotation_id": str(record["annotation_id"]), "stage175_instance_id": str(record["instance_id"]),
            }
            rows.append(row)
            candidate_exact.add(digest)
            candidate_tree.add(value, row)
            retained_conditions[condition] += 1
            retained_tracks.add(track_group)

    failures: list[str] = []
    if len(rows) < args.minimum_rows:
        failures.append(f"retained rows {len(rows)} < {args.minimum_rows}")
    if retained_conditions["night"] < args.minimum_night_rows:
        failures.append(f"retained night rows {retained_conditions['night']} < {args.minimum_night_rows}")
    if retained_conditions["dusk"] < args.minimum_dusk_rows:
        failures.append(f"retained dusk rows {retained_conditions['dusk']} < {args.minimum_dusk_rows}")
    if any(row["body_type"] != "unknown" or row["color"] != "unknown" for row in rows):
        failures.append("fine labels were assigned before teacher audit")

    return {
        "schema_version": "stage175-m3ot-lowlight-crops-v1",
        "status": "pass" if not failures else "fail",
        "extraction_root": str(root),
        "photometry_report": str(photometry_path),
        "photometry_report_sha256": sha256_file(photometry_path),
        "base_manifests": [{"path": str(path.resolve()), "sha256": sha256_file(path.resolve())} for path in args.base_manifest],
        "output_images": str(output_images),
        "output_rows": len(rows),
        "output_bytes": output_bytes,
        "retained_condition_counts": dict(sorted(retained_conditions.items())),
        "retained_track_groups": len(retained_tracks),
        "source_annotation_counts": dict(sorted(source_annotation_counts.items())),
        "counters": dict(sorted(counters.items())),
        "body_labels_all_unknown": True,
        "color_labels_all_unknown": True,
        "validation_payload_opened": False,
        "test_payload_opened": False,
        "ir_payload_opened": False,
        "training_started": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
        "failures": failures,
        "rows": rows,
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train-only, track-capped M3OT night/dusk crops with lineage-wide deduplication.")
    parser.add_argument("--extraction-root", type=Path, required=True)
    parser.add_argument("--photometry-report", type=Path, required=True)
    parser.add_argument("--expected-photometry-sha256", required=True)
    parser.add_argument("--base-manifest", type=Path, action="append", required=True)
    parser.add_argument("--expected-base-sha256", action="append", required=True)
    parser.add_argument("--output-images", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--conditions", action="append", default=[])
    parser.add_argument("--maximum-frames-per-track", type=int, default=64)
    parser.add_argument("--minimum-bbox-area", type=float, default=64.0)
    parser.add_argument("--minimum-bbox-max-side", type=float, default=8.0)
    parser.add_argument("--context-factor", type=float, default=1.6)
    parser.add_argument("--minimum-crop-side", type=int, default=32)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    parser.add_argument("--minimum-rows", type=int, default=5000)
    parser.add_argument("--minimum-night-rows", type=int, default=500)
    parser.add_argument("--minimum-dusk-rows", type=int, default=4000)
    parser.add_argument("--minimum-free-bytes", type=int, default=107374182400)
    parser.add_argument("--maximum-output-bytes", type=int, default=4294967296)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.conditions:
        args.conditions = ["night", "dusk"]
    if args.maximum_frames_per_track <= 0 or not 0 <= args.near_duplicate_hamming <= 7:
        raise ValueError("invalid sampling or near-duplicate policy")
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage175 evidence")
    result = build(args)
    rows = result.pop("rows")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output_manifest, rows)
    result["output_manifest"] = str(args.output_manifest.resolve())
    result["output_manifest_sha256"] = sha256_file(args.output_manifest)
    args.output_report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path in (args.output_manifest, args.output_report):
        path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
