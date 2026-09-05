#!/usr/bin/env python3
"""Build fail-closed Stage71 body and DVM-replay color teacher manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "frozen_video", "36-48")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def load_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def resolve_image(row: dict[str, str], manifest: Path, image_root: Path, safety_root: Path) -> str:
    raw = Path(row.get("image_path", ""))
    if not str(raw):
        raise ValueError("manifest row is missing image_path")
    candidates = [raw] if raw.is_absolute() else [image_root / raw, manifest.parent / raw]
    resolved = next((candidate.resolve() for candidate in candidates if candidate.resolve().is_file()), None)
    if resolved is None:
        raise FileNotFoundError(candidates[0])
    resolved.relative_to(safety_root.resolve())
    return str(resolved)


def content_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_sha256") or row.get("sha256") or "").strip().lower()


def perceptual_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_dhash64") or row.get("dhash64") or "").strip().lower()


class HammingBKTree:
    def __init__(self) -> None:
        self.root: list | None = None

    def add(self, value: int, row: dict[str, str]) -> None:
        if self.root is None:
            self.root = [value, [row], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(row)
                return
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = [value, [row], {}]
                return
            node = child

    def find(self, value: int, radius: int, removed: set[int] | None = None) -> dict[str, str] | None:
        if self.root is None:
            return None
        removed = removed or set()
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                candidate = next((row for row in node[1] if id(row) not in removed), None)
                if candidate is not None:
                    return candidate
            low, high = distance - radius, distance + radius
            stack.extend(child for edge, child in node[2].items() if low <= edge <= high)
        return None


def deduplicate_color(rows: list[dict[str, str]], radius: int) -> tuple[list[dict[str, str]], dict]:
    # Withheld CCTV validation is authoritative and considered before either
    # training source. CCTV truth is preferred to listing-domain DVM evidence.
    priority = {("cctv", "validation"): 0, ("cctv", "train"): 1, ("dvm", "train"): 2}
    ordered = sorted(rows, key=lambda row: (priority[(row["stage71_origin"], row["split"])], row["image_path"]))
    exact: dict[str, list[dict[str, str]]] = defaultdict(list)
    trees: dict[str, HammingBKTree] = defaultdict(HammingBKTree)
    kept: list[dict[str, str]] = []
    removed: set[int] = set()
    counts = Counter()
    conflicts = Counter()
    examples: list[dict] = []
    for row in ordered:
        digest = content_hash(row)
        candidate = next((item for item in exact.get(digest, []) if id(item) not in removed), None) if digest else None
        phash = perceptual_hash(row)
        if candidate is None and len(phash) == 16:
            candidate = trees[row["color"]].find(int(phash, 16), radius, removed)
        if candidate is None:
            kept.append(row)
            if digest:
                exact[digest].append(row)
            if len(phash) == 16:
                trees[row["color"]].add(int(phash, 16), row)
            continue
        counts["duplicate_rows"] += 1
        if candidate["split"] != row["split"]:
            counts["cross_split_source_duplicates"] += 1
        if candidate["color"] != row["color"]:
            conflicts[f"{candidate['color']}->{row['color']}"] += 1
            removed.add(id(candidate))
            if len(examples) < 100:
                examples.append({
                    "kept_then_removed": candidate["image_path"],
                    "removed": row["image_path"],
                    "left_color": candidate["color"],
                    "right_color": row["color"],
                })
    output = [row for row in kept if id(row) not in removed]
    return output, {
        "input_rows": len(rows),
        "output_rows": len(output),
        "counts": dict(sorted(counts.items())),
        "color_conflicts_removed_both": dict(sorted(conflicts.items())),
        "conflict_examples": examples,
    }


def cross_split_leaks(rows: list[dict[str, str]], radius: int, limit: int = 100) -> list[dict]:
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    exact = {content_hash(row): row for row in train if content_hash(row)}
    trees: dict[str, HammingBKTree] = defaultdict(HammingBKTree)
    for row in train:
        phash = perceptual_hash(row)
        if len(phash) == 16:
            trees[row["color"]].add(int(phash, 16), row)
    found: list[dict] = []
    for row in validation:
        candidate = exact.get(content_hash(row)) if content_hash(row) else None
        phash = perceptual_hash(row)
        if candidate is None and len(phash) == 16:
            candidate = trees[row["color"]].find(int(phash, 16), radius)
        if candidate is not None:
            found.append({"train": candidate["image_path"], "validation": row["image_path"]})
            if len(found) >= limit:
                break
    return found


def write_manifest(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-manifest", type=Path, required=True)
    parser.add_argument("--body-root", type=Path, required=True)
    parser.add_argument("--dvm-manifest", type=Path, required=True)
    parser.add_argument("--dvm-root", type=Path, required=True)
    parser.add_argument("--cctv-color-manifest", type=Path, required=True)
    parser.add_argument("--cctv-color-root", type=Path, required=True)
    parser.add_argument("--datasets-safety-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-body-sha256", required=True)
    parser.add_argument("--expected-dvm-sha256", required=True)
    parser.add_argument("--expected-cctv-color-sha256", required=True)
    parser.add_argument("--dvm-sample-weight", type=float, default=1.0)
    parser.add_argument("--cctv-sample-weight", type=float, default=10.0)
    parser.add_argument("--near-duplicate-hamming", type=int, default=3)
    parser.add_argument("--minimum-color-train-rows", type=int, default=118338)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_root}")
    if not 0 <= args.near_duplicate_hamming <= 7:
        raise ValueError("near-duplicate Hamming threshold must be between 0 and 7")
    if not 0.10 <= args.dvm_sample_weight <= 10.0 or not 0.10 <= args.cctv_sample_weight <= 10.0:
        raise ValueError("sample weights must stay within the trainer's [0.10, 10.0] clamp")
    expected = {
        args.body_manifest: args.expected_body_sha256,
        args.dvm_manifest: args.expected_dvm_sha256,
        args.cctv_color_manifest: args.expected_cctv_color_sha256,
    }
    for path, digest in expected.items():
        if sha256(path).lower() != digest.lower():
            raise RuntimeError(f"input SHA256 mismatch: {path}")

    body_fields, body_rows = load_manifest(args.body_manifest)
    dvm_fields, dvm_rows = load_manifest(args.dvm_manifest)
    cctv_fields, cctv_rows = load_manifest(args.cctv_color_manifest)
    safety = args.datasets_safety_root.resolve()
    all_text = "\n".join(" ".join(row.values()).lower() for rows in (body_rows, dvm_rows, cctv_rows) for row in rows)
    if any(marker in all_text for marker in FROZEN_MARKERS):
        raise RuntimeError("frozen asset marker detected in a Stage71 input")
    if any(row.get("split") == "test" for rows in (body_rows, dvm_rows, cctv_rows) for row in rows):
        raise RuntimeError("Stage71 inputs must not contain test rows")

    prepared_body: list[dict[str, str]] = []
    for source in body_rows:
        if source.get("split") not in {"train", "validation"}:
            continue
        row = dict(source)
        row["image_path"] = resolve_image(row, args.body_manifest, args.body_root, safety)
        row["stage71_origin"] = "stage70_body"
        row["stage71_role"] = "body_teacher"
        prepared_body.append(row)

    prepared_color: list[dict[str, str]] = []
    for origin, manifest, root, rows in (
        ("dvm", args.dvm_manifest, args.dvm_root, dvm_rows),
        ("cctv", args.cctv_color_manifest, args.cctv_color_root, cctv_rows),
    ):
        for source in rows:
            split = source.get("split")
            if origin == "dvm" and split != "train":
                continue
            if origin == "cctv" and split not in {"train", "validation"}:
                continue
            if not truthy(source.get("color_supervised")) or source.get("color") in {"", "unknown"}:
                raise RuntimeError(f"{origin} contains an unknown or unsupervised selected color row")
            row = dict(source)
            row["image_path"] = resolve_image(row, manifest, root, safety)
            row["body_type"] = "unknown"
            row["body_type_supervised"] = "false"
            row["coarse_body_family"] = ""
            row["stage71_origin"] = origin
            row["stage71_role"] = "color_teacher"
            if origin == "cctv" and (
                truthy(row.get("night"))
                or truthy(row.get("low_light"))
                or str(row.get("lighting", "")).strip().lower()
                in {"night", "low_light", "low_light_proxy", "night_machine_photometric_consensus"}
            ):
                row["night"] = "true"
            source_weight = args.dvm_sample_weight if origin == "dvm" else args.cctv_sample_weight
            row["sample_weight"] = f"{source_weight:.6f}"
            prepared_color.append(row)

    prepared_color, dedup = deduplicate_color(prepared_color, args.near_duplicate_hamming)
    leaks = cross_split_leaks(prepared_color, args.near_duplicate_hamming)
    group_leaks: list[str] = []
    train_groups = {row.get("track_group") or row.get("video_id") or row.get("source_group") for row in prepared_color if row["split"] == "train"}
    validation_groups = {row.get("track_group") or row.get("video_id") or row.get("source_group") for row in prepared_color if row["split"] == "validation"}
    group_leaks = sorted((train_groups & validation_groups) - {None, ""})[:100]
    prepared_body.sort(key=lambda row: (row.get("split", ""), row.get("source_dataset", ""), row["image_path"]))
    prepared_color.sort(key=lambda row: (row.get("split", ""), row.get("stage71_origin", ""), row.get("color", ""), row["image_path"]))

    all_fields: list[str] = []
    for field in body_fields + dvm_fields + cctv_fields + ["stage71_origin", "stage71_role", "sample_weight"]:
        if field not in all_fields:
            all_fields.append(field)
    args.output_root.mkdir(parents=True, exist_ok=True)
    body_path = args.output_root / "attribute_manifest.stage71-body-teacher.csv"
    color_path = args.output_root / "attribute_manifest.stage71-color-teacher.csv"
    write_manifest(body_path, all_fields, prepared_body)
    write_manifest(color_path, all_fields, prepared_color)

    color_train = [row for row in prepared_color if row["split"] == "train"]
    color_validation = [row for row in prepared_color if row["split"] == "validation"]
    weighted_total = sum(float(row["sample_weight"]) for row in color_train)
    weighted_cctv = sum(float(row["sample_weight"]) for row in color_train if row["stage71_origin"] == "cctv")
    status = "pass"
    if (
        len(color_train) < args.minimum_color_train_rows
        or not color_validation
        or leaks
        or group_leaks
        or not prepared_body
        or any(row["split"] == "test" for row in prepared_body + prepared_color)
    ):
        status = "fail"
    report = {
        "schema_version": "stage71-teacher-manifests-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "eligibility": "research-only_non-deployable",
        "inputs": {
            "body": {"path": str(args.body_manifest.resolve()), "sha256": sha256(args.body_manifest), "rows": len(body_rows)},
            "dvm": {"path": str(args.dvm_manifest.resolve()), "sha256": sha256(args.dvm_manifest), "rows": len(dvm_rows)},
            "cctv_color": {"path": str(args.cctv_color_manifest.resolve()), "sha256": sha256(args.cctv_color_manifest), "rows": len(cctv_rows)},
        },
        "outputs": {
            "body_teacher": {"path": str(body_path), "sha256": sha256(body_path), "rows": len(prepared_body), "splits": dict(Counter(row["split"] for row in prepared_body))},
            "color_teacher": {
                "path": str(color_path),
                "sha256": sha256(color_path),
                "rows": len(prepared_color),
                "splits": dict(Counter(row["split"] for row in prepared_color)),
                "train_origins": dict(Counter(row["stage71_origin"] for row in color_train)),
                "train_colors": dict(Counter(row["color"] for row in color_train)),
                "validation_colors": dict(Counter(row["color"] for row in color_validation)),
                "train_night_rows": sum(truthy(row.get("night")) for row in color_train),
                "sample_weights": {"dvm": args.dvm_sample_weight, "cctv": args.cctv_sample_weight},
                "expected_weighted_cctv_fraction_before_class_weighting": weighted_cctv / weighted_total if weighted_total else 0.0,
                "dedup": dedup,
            },
        },
        "post_filter_cross_split_near_leaks": leaks,
        "post_filter_group_leaks": group_leaks,
        "policy": {
            "dvm_validation_imported": False,
            "cctv_validation_used_for_model_selection_only": True,
            "test_rows_imported": False,
            "frozen_video_used": False,
            "threshold_or_temperature_selected": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "color_teacher_uses_dvm_replay_instead_of_cctv_only_refinetune": True,
        },
    }
    report_path = args.output_root / "stage71-teacher-manifests-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
