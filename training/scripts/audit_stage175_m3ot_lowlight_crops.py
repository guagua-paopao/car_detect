from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_stage70_specialist_manifests import HammingBKTree  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash64(image: Image.Image) -> int:
    values = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    result = 0
    for bit in (values[:, 1:] > values[:, :-1]).ravel():
        result = (result << 1) | int(bit)
    return result


def content_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_sha256") or row.get("sha256") or row.get("image_sha256") or "").strip().lower()


def perceptual_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_dhash64") or row.get("dhash64") or "").strip().lower()


def load_base(paths: list[Path], expected: list[str]) -> tuple[set[str], HammingBKTree, int]:
    if len(paths) != len(expected):
        raise ValueError("base manifest/SHA count mismatch")
    exact: set[str] = set()
    tree = HammingBKTree()
    rows = 0
    for path, pinned in zip(paths, expected):
        if sha256_file(path).lower() != pinned.lower():
            raise ValueError(f"base manifest SHA mismatch: {path}")
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


def validate_row_contract(row: dict[str, str]) -> list[str]:
    failures: list[str] = []
    if row.get("split") != "train":
        failures.append("non-train split")
    if row.get("body_type") != "unknown" or row.get("body_type_supervised") != "false":
        failures.append("body truth fabricated")
    if row.get("coarse_body_family", ""):
        failures.append("coarse body truth fabricated")
    if row.get("color") != "unknown" or row.get("color_supervised") != "false":
        failures.append("color truth fabricated")
    if row.get("source_dataset") != "M3OT" or row.get("source_label") != "vehicle":
        failures.append("source contract mismatch")
    if row.get("source_license") != "CC-BY-4.0" or row.get("license_train_eligible") != "true":
        failures.append("license contract mismatch")
    if row.get("lighting") not in {"night", "dusk"} or row.get("low_light") != "true":
        failures.append("low-light contract mismatch")
    if row.get("night") != str(row.get("lighting") == "night").lower():
        failures.append("night flag mismatch")
    if row.get("small_target") != "true":
        failures.append("small-target flag missing")
    if row.get("pseudo_label") != "false" or row.get("teacher_consensus") != "false":
        failures.append("teacher label assigned before audit")
    if row.get("research_only") != "true" or row.get("deployment_eligible") != "false":
        failures.append("research/deployment contract mismatch")
    if row.get("stage175_origin") != "m3ot_official_rgb_train_lowlight_track_capped":
        failures.append("origin mismatch")
    return failures


def audit(args: argparse.Namespace) -> dict[str, object]:
    manifest = args.manifest.resolve()
    crop_report = args.crop_report.resolve()
    builder = args.builder.resolve()
    images_root = args.images_root.resolve()
    actual = {
        "manifest": sha256_file(manifest),
        "crop_report": sha256_file(crop_report),
        "builder": sha256_file(builder),
    }
    expected = {
        "manifest": args.expected_manifest_sha256.lower(),
        "crop_report": args.expected_crop_report_sha256.lower(),
        "builder": args.expected_builder_sha256.lower(),
    }
    failures: list[str] = []
    if actual != expected:
        failures.append(f"pinned input mismatch: actual={actual} expected={expected}")
    source = json.loads(crop_report.read_text(encoding="utf-8"))
    if source.get("status") != "pass" or source.get("failures"):
        failures.append("source crop report did not pass")
    if str(source.get("output_manifest_sha256", "")).lower() != actual["manifest"]:
        failures.append("source report manifest SHA mismatch")
    if Path(str(source.get("output_images", ""))).resolve() != images_root:
        failures.append("source report image-root mismatch")
    if images_root.is_symlink() or not images_root.is_dir():
        failures.append("image root missing or symbolic")

    base_exact, base_tree, base_rows = load_base(args.base_manifest, args.expected_base_sha256)
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(source.get("output_rows", -1)):
        failures.append(f"row count {len(rows)} != source report {source.get('output_rows')}")

    manifest_paths: set[Path] = set()
    exact_hashes: set[str] = set()
    candidate_tree = HammingBKTree()
    counters: Counter[str] = Counter()
    invalid_examples: list[str] = []
    near_examples: list[dict[str, str]] = []
    tracks: set[str] = set()

    for index, row in enumerate(rows, 2):
        def invalid(reason: str) -> None:
            counters[f"invalid:{reason}"] += 1
            if len(invalid_examples) < 100:
                invalid_examples.append(f"row {index}: {reason}")

        for reason in validate_row_contract(row):
            invalid(reason)
        raw_path = Path(row.get("image_path", ""))
        try:
            path = raw_path.resolve()
            path.relative_to(images_root)
        except Exception:
            invalid("path outside image root")
            continue
        if path.is_symlink() or not path.is_file():
            invalid("missing or symbolic image")
            continue
        if path in manifest_paths:
            invalid("duplicate image path")
            continue
        manifest_paths.add(path)
        tracks.add(row.get("track_group", ""))

        digest = sha256_file(path)
        if digest != row.get("sha256", "").lower() or digest != row.get("crop_sha256", "").lower():
            invalid("file SHA256 mismatch")
        if digest in exact_hashes:
            invalid("within-manifest exact duplicate")
        if digest in base_exact:
            invalid("base exact duplicate")
        exact_hashes.add(digest)
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                value = dhash64(image)
        except Exception as error:
            invalid(f"image decode failed: {error}")
            continue
        if str(width) != row.get("width") or str(height) != row.get("height"):
            invalid("decoded dimensions mismatch")
        value_hex = f"{value:016x}"
        if value_hex != row.get("dhash64", "").lower() or value_hex != row.get("crop_dhash64", "").lower():
            invalid("persisted dHash mismatch")
        base_near = base_tree.find(value, args.near_duplicate_hamming)
        if base_near is not None:
            invalid("base perceptual near duplicate")
            if len(near_examples) < 50:
                near_examples.append({"kind": "base", "left": base_near.get("image_path", ""), "right": str(path)})
        candidate_near = candidate_tree.find(value, args.near_duplicate_hamming)
        if candidate_near is not None:
            invalid("within-manifest perceptual near duplicate")
            if len(near_examples) < 50:
                near_examples.append({"kind": "candidate", "left": candidate_near.get("image_path", ""), "right": str(path)})
        candidate_tree.add(value, row)
        counters[f"condition:{row.get('lighting', '')}"] += 1

    filesystem_paths = set(images_root.glob("*.jpg")) if images_root.is_dir() else set()
    unexpected = sorted(str(path) for path in filesystem_paths - manifest_paths)
    missing = sorted(str(path) for path in manifest_paths - filesystem_paths)
    if unexpected:
        failures.append(f"unlisted image files: {len(unexpected)}")
    if missing:
        failures.append(f"manifest image files missing: {len(missing)}")
    invalid_count = sum(value for key, value in counters.items() if key.startswith("invalid:"))
    if invalid_count:
        failures.append(f"invalid rows/files/dedup findings: {invalid_count}")
    expected_conditions = {key: int(value) for key, value in source.get("retained_condition_counts", {}).items()}
    actual_conditions = {key.split(":", 1)[1]: value for key, value in counters.items() if key.startswith("condition:")}
    if actual_conditions != expected_conditions:
        failures.append(f"condition count mismatch: {actual_conditions} != {expected_conditions}")
    if len(tracks) != int(source.get("retained_track_groups", -1)):
        failures.append(f"track count {len(tracks)} != {source.get('retained_track_groups')}")

    return {
        "schema_version": "stage175-m3ot-lowlight-crops-audit-v1",
        "status": "pass" if not failures else "fail",
        "inputs": {
            "manifest": str(manifest), "manifest_sha256": actual["manifest"],
            "crop_report": str(crop_report), "crop_report_sha256": actual["crop_report"],
            "builder": str(builder), "builder_sha256": actual["builder"],
            "images_root": str(images_root),
            "base_manifests": [{"path": str(path.resolve()), "sha256": sha256_file(path.resolve())} for path in args.base_manifest],
        },
        "verified": {
            "rows": len(rows), "tracks": len(tracks), "manifest_paths": len(manifest_paths),
            "filesystem_jpegs": len(filesystem_paths), "exact_hashes": len(exact_hashes),
            "base_rows": base_rows, "near_duplicate_hamming": args.near_duplicate_hamming,
            "condition_counts": actual_conditions, "unexpected_files": unexpected[:100],
            "missing_files": missing[:100], "near_duplicate_examples": near_examples,
            "invalid_examples": invalid_examples, "counters": dict(sorted(counters.items())),
        },
        "policy": {
            "all_rows_train_only": True, "all_body_labels_unknown": True,
            "all_color_labels_unknown": True, "validation_rows_read": 0, "test_rows_read": 0,
            "ir_rows_read": 0, "frozen_video_used": False, "production_model_modified": False,
            "training_started": False, "deployment_performed": False,
        },
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Independently audit Stage175 M3OT train-only low-light crops.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--crop-report", type=Path, required=True)
    parser.add_argument("--expected-crop-report-sha256", required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--expected-builder-sha256", required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, action="append", required=True)
    parser.add_argument("--expected-base-sha256", action="append", required=True)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{sha256_file(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "verified": report["verified"], "failures": report["failures"]}, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
