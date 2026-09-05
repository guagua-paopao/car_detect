from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageStat


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute quantile of empty input")
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    weight = index - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def cluster_conditions(pair_luminance: dict[str, float]) -> tuple[dict[str, str], dict[str, float]]:
    if len(pair_luminance) < 3:
        raise ValueError("at least three paired sequences are required")
    values = sorted(pair_luminance.values())
    centers = [values[0], statistics.median(values), values[-1]]
    assignments: dict[str, int] = {}
    for _ in range(100):
        new_assignments = {
            key: min(range(3), key=lambda index: (abs(value - centers[index]), index))
            for key, value in pair_luminance.items()
        }
        groups = [[pair_luminance[key] for key, group in new_assignments.items() if group == index] for index in range(3)]
        if any(not group for group in groups):
            raise ValueError("photometric clustering produced an empty condition")
        new_centers = [statistics.mean(group) for group in groups]
        if new_assignments == assignments and all(abs(a - b) < 1e-9 for a, b in zip(new_centers, centers)):
            break
        assignments = new_assignments
        centers = new_centers
    ordered_clusters = sorted(range(3), key=lambda index: centers[index])
    label_for_cluster = {
        ordered_clusters[0]: "night",
        ordered_clusters[1]: "dusk",
        ordered_clusters[2]: "day",
    }
    labels = {key: label_for_cluster[group] for key, group in assignments.items()}
    labeled_centers = {label_for_cluster[index]: centers[index] for index in range(3)}
    return labels, labeled_centers


def audit_photometry(
    extraction_root: Path,
    extraction_report: Path,
    *,
    expected_extraction_report_sha256: str,
    expected_images: int,
) -> dict[str, object]:
    if not extraction_root.is_dir():
        raise FileNotFoundError(extraction_root)
    if not extraction_report.is_file():
        raise FileNotFoundError(extraction_report)
    report_sha256 = sha256_file(extraction_report)
    if report_sha256.lower() != expected_extraction_report_sha256.lower():
        raise ValueError("extraction report SHA256 mismatch")
    extraction = json.loads(extraction_report.read_text(encoding="utf-8"))
    if extraction.get("status") != "pass":
        raise ValueError("extraction report did not pass")
    if extraction.get("validation_payload_opened") or extraction.get("test_payload_opened"):
        raise ValueError("extraction report records validation/test access")
    if Path(str(extraction.get("output_root"))).resolve() != extraction_root.resolve():
        raise ValueError("extraction root differs from sealed extraction report")

    failures: list[str] = []
    sequence_values: dict[str, list[float]] = defaultdict(list)
    sequence_dimensions: dict[str, Counter[str]] = defaultdict(Counter)
    invalid_images = 0
    image_count = 0
    for drone in ("1", "2"):
        train_root = extraction_root / drone / "rgb" / "train"
        if not train_root.is_dir():
            failures.append(f"missing RGB train root for drone {drone}")
            continue
        for sequence_root in sorted(path for path in train_root.iterdir() if path.is_dir()):
            image_root = sequence_root / "img1"
            for image_path in sorted(image_root.glob("*.PNG")):
                image_count += 1
                try:
                    with Image.open(image_path) as image:
                        image.load()
                        width, height = image.size
                        gray = image.convert("L").resize((64, 51), Image.Resampling.BILINEAR)
                        luminance = float(ImageStat.Stat(gray).mean[0])
                except Exception:
                    invalid_images += 1
                    continue
                sequence_values[sequence_root.name].append(luminance)
                sequence_dimensions[sequence_root.name][f"{width}x{height}"] += 1

    if image_count != expected_images:
        failures.append(f"image count mismatch: {image_count} != {expected_images}")
    if invalid_images:
        failures.append(f"invalid RGB train images: {invalid_images}")
    if len(sequence_values) != 14:
        failures.append(f"expected 14 RGB train sequences, found {len(sequence_values)}")
    unexpected_dimensions = {
        sequence: dict(counter)
        for sequence, counter in sequence_dimensions.items()
        if set(counter) != {"640x512"}
    }
    if unexpected_dimensions:
        failures.append(f"unexpected image dimensions: {unexpected_dimensions}")

    sequence_stats: dict[str, dict[str, object]] = {}
    paired_values: dict[str, list[float]] = defaultdict(list)
    for sequence, values in sorted(sequence_values.items()):
        suffix = sequence.split("-", 1)[1]
        median = statistics.median(values)
        paired_values[suffix].append(median)
        sequence_stats[sequence] = {
            "images": len(values),
            "luminance_p10": quantile(values, 0.10),
            "luminance_median": median,
            "luminance_p90": quantile(values, 0.90),
            "dimensions": dict(sequence_dimensions[sequence]),
        }
    malformed_pairs = {key: len(values) for key, values in paired_values.items() if len(values) != 2}
    if malformed_pairs:
        failures.append(f"drone sequence pairing mismatch: {malformed_pairs}")

    pair_medians = {key: statistics.mean(values) for key, values in paired_values.items() if len(values) == 2}
    conditions: dict[str, str] = {}
    centers: dict[str, float] = {}
    try:
        conditions, centers = cluster_conditions(pair_medians)
    except ValueError as error:
        failures.append(str(error))
    for sequence, stats in sequence_stats.items():
        suffix = sequence.split("-", 1)[1]
        stats["condition"] = conditions.get(suffix, "unknown")
    condition_counts = Counter(stats["condition"] for stats in sequence_stats.values())
    condition_image_counts = Counter()
    for stats in sequence_stats.values():
        condition_image_counts[str(stats["condition"])] += int(stats["images"])
    if set(condition_counts) != {"day", "dusk", "night"}:
        failures.append(f"all three published conditions were not recovered: {dict(condition_counts)}")
    if centers and not (centers["night"] < centers["dusk"] < centers["day"]):
        failures.append(f"condition luminance order is invalid: {centers}")

    return {
        "schema_version": "m3ot-rgb-train-photometry-v1",
        "status": "pass" if not failures else "fail",
        "extraction_root": str(extraction_root),
        "extraction_report": str(extraction_report),
        "extraction_report_sha256": report_sha256,
        "decoded_rgb_train_images": image_count - invalid_images,
        "invalid_images": invalid_images,
        "sequence_stats": sequence_stats,
        "paired_sequence_luminance": dict(sorted(pair_medians.items())),
        "condition_centers": centers,
        "condition_sequence_counts": dict(sorted(condition_counts.items())),
        "condition_image_counts": dict(sorted(condition_image_counts.items())),
        "classification_policy": "deterministic three-cluster 1D k-means over paired-drone sequence median luminance; cluster order is mapped to the official published day/dusk/night conditions",
        "domain_policy": "aerial auxiliary train-only evidence; conditions are not gate-camera truth and cannot be used for validation/test claims",
        "validation_payload_opened": False,
        "test_payload_opened": False,
        "ir_payload_opened": False,
        "training_started": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode-audit M3OT RGB train images and classify paired sequences by published lighting conditions.")
    parser.add_argument("--extraction-root", type=Path, required=True)
    parser.add_argument("--extraction-report", type=Path, required=True)
    parser.add_argument("--expected-extraction-report-sha256", required=True)
    parser.add_argument("--expected-images", type=int, default=8631)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_photometry(
        args.extraction_root.resolve(),
        args.extraction_report.resolve(),
        expected_extraction_report_sha256=args.expected_extraction_report_sha256,
        expected_images=args.expected_images,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{sha256_file(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
