#!/usr/bin/env python3
"""Build a leakage-safe light/heavy-truck specialist manifest.

Validation truth is taken only from the immutable Stage83 validation split.
Train-only heavy-truck additions are accepted only from explicit articulated or
three-plus-axle truth. Generic truck and coarse-family labels never supervise
the specialist head.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from build_stage70_specialist_manifests import HammingBKTree


FINE_LABELS = {"light_truck", "heavy_truck"}
STAGE89_ALLOWED = {("heavy_truck", "articulated_truck")}
STAGE91_ALLOWED = {
    ("heavy_truck", "three_axle_truck"),
    ("heavy_truck", "four_axle_truck"),
    ("heavy_truck", "five_plus_axle_truck"),
}
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "baseline-preview-36-48", "36-48s")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def content_hash(row: dict[str, str]) -> str:
    return str(
        row.get("crop_sha256")
        or row.get("sha256")
        or row.get("image_sha256")
        or row.get("source_image_sha256")
        or ""
    ).strip().lower()


def perceptual_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_dhash64") or row.get("dhash64") or "").strip().lower()


def group_key(row: dict[str, str]) -> str:
    for field in ("track_group", "video_id", "source_group", "camera_id"):
        value = str(row.get(field, "")).strip()
        if value and value.lower() != "unknown":
            return f"{field}:{value}"
    return ""


def resolve_image(manifest: Path, safety_root: Path, raw: str) -> Path:
    value = Path(raw)
    resolved = value.resolve() if value.is_absolute() else (manifest.parent / value).resolve()
    root = safety_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"image path escapes safety root: {resolved}") from exc
    return resolved


def readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (FileNotFoundError, OSError, ValueError):
        return False


def normalize(
    row: dict[str, str], *, origin: str, manifest: Path, safety_root: Path
) -> dict[str, str]:
    output = dict(row)
    output["image_path"] = str(resolve_image(manifest, safety_root, row.get("image_path", "")))
    output["body_type_supervised"] = "true"
    output["color"] = "unknown"
    output["color_supervised"] = "false"
    output["stage93_original_review_status"] = row.get("review_status", "")
    output["review_status"] = "approved"
    output["stage93_origin"] = origin
    output["stage93_previous_sample_weight"] = row.get("sample_weight", "")
    if origin == "stage91_inatrc":
        output["sample_weight"] = "1.250"
        output["stage93_fine_truth_source"] = "official_three_plus_axle_toll_cctv"
    elif origin == "stage89_mio":
        output["sample_weight"] = "0.750"
        output["stage93_fine_truth_source"] = "official_articulated_truck_train"
    else:
        output["sample_weight"] = row.get("sample_weight", "1.000") or "1.000"
        output["stage93_fine_truth_source"] = "stage83_preserved_exact_truth"
    license_name = str(output.get("source_license", "")).upper()
    restricted = "NC" in license_name or truthy(output.get("research_only"))
    output["research_only"] = "true" if restricted else "false"
    output["deployment_eligible"] = "false" if restricted else "true"
    output["stage93_eligibility"] = (
        "research-only_non-deployable" if restricted else "license_train_eligible_candidate"
    )
    return output


def eligible_base(row: dict[str, str]) -> bool:
    return (
        row.get("split", "").strip().lower() in {"train", "validation"}
        and row.get("body_type", "").strip().lower() in FINE_LABELS
        and truthy(row.get("body_type_supervised"))
        and row.get("review_status", "").strip().lower() == "approved"
        and truthy(row.get("license_train_eligible"))
    )


def eligible_addition(row: dict[str, str], allowed: set[tuple[str, str]]) -> bool:
    pair = (
        row.get("body_type", "").strip().lower(),
        row.get("source_label", "").strip().lower(),
    )
    return (
        row.get("split", "").strip().lower() == "train"
        and pair in allowed
        and truthy(row.get("body_type_supervised"))
        and truthy(row.get("license_train_eligible"))
    )


def deduplicate_validation_first(
    rows: list[dict[str, str]], radius: int
) -> tuple[list[dict[str, str]], dict[str, object]]:
    validation = sorted(
        (row for row in rows if row["split"] == "validation"),
        key=lambda row: row["image_path"],
    )
    train = sorted(
        (row for row in rows if row["split"] == "train"),
        key=lambda row: (row.get("stage93_origin", ""), row["image_path"]),
    )
    counters = Counter()
    conflicts = Counter()
    validation_exact_candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    validation_candidate_tree = HammingBKTree()
    kept_validation: list[dict[str, str]] = []
    removed_validation_ids: set[int] = set()
    for row in validation:
        digest = content_hash(row)
        previous = (
            next(
                (item for item in validation_exact_candidates.get(digest, []) if id(item) not in removed_validation_ids),
                None,
            )
            if digest else None
        )
        value = perceptual_hash(row)
        match_kind = "exact" if previous is not None else ""
        if previous is None and len(value) == 16:
            previous = validation_candidate_tree.find(
                int(value, 16), radius, removed_validation_ids
            )
            match_kind = "near" if previous is not None else ""
        if previous is not None:
            counters["validation_duplicate_or_near"] += 1
            if previous["body_type"] != row["body_type"]:
                conflicts[
                    f"validation_{match_kind}:{previous['body_type']}->{row['body_type']}"
                ] += 1
                removed_validation_ids.add(id(previous))
                counters["validation_conflict_rows_removed"] += 2
            continue
        kept_validation.append(row)
        if digest:
            validation_exact_candidates[digest].append(row)
        if len(value) == 16:
            validation_candidate_tree.add(int(value, 16), row)
    validation = [row for row in kept_validation if id(row) not in removed_validation_ids]

    # Rebuild the protected reference only from the final unambiguous
    # validation rows. Train rows are filtered against it, never the reverse.
    validation_exact: dict[str, dict[str, str]] = {}
    validation_tree = HammingBKTree()
    validation_groups = {group_key(row) for row in validation if group_key(row)}
    for row in validation:
        digest = content_hash(row)
        if digest:
            validation_exact[digest] = row
        value = perceptual_hash(row)
        if len(value) == 16:
            validation_tree.add(int(value, 16), row)

    accepted_train: list[dict[str, str]] = []
    train_exact: dict[str, dict[str, str]] = {}
    train_tree = HammingBKTree()
    removed_train_ids: set[int] = set()
    for row in train:
        group = group_key(row)
        if group and group in validation_groups:
            counters["train_group_overlap_validation"] += 1
            continue
        digest = content_hash(row)
        candidate = validation_exact.get(digest) if digest else None
        value = perceptual_hash(row)
        if candidate is None and len(value) == 16:
            candidate = validation_tree.find(int(value, 16), radius)
        if candidate is not None:
            counters["train_duplicate_or_near_validation"] += 1
            if candidate["body_type"] != row["body_type"]:
                conflicts[f"cross_split:{candidate['body_type']}->{row['body_type']}"] += 1
            continue

        candidate = train_exact.get(digest) if digest else None
        if candidate is None and len(value) == 16:
            candidate = train_tree.find(int(value, 16), radius, removed_train_ids)
        if candidate is not None:
            counters["train_duplicate_or_near"] += 1
            if candidate["body_type"] != row["body_type"]:
                conflicts[f"train:{candidate['body_type']}->{row['body_type']}"] += 1
                removed_train_ids.add(id(candidate))
            continue
        accepted_train.append(row)
        if digest:
            train_exact[digest] = row
        if len(value) == 16:
            train_tree.add(int(value, 16), row)
    accepted_train = [row for row in accepted_train if id(row) not in removed_train_ids]
    return [*validation, *accepted_train], {
        "counters": dict(sorted(counters.items())),
        "label_conflicts": dict(sorted(conflicts.items())),
        "validation_input_rows": len(kept_validation) + counters["validation_duplicate_or_near"],
        "validation_rows_after_dedup": len(validation),
        "train_rows_after_dedup": len(accepted_train),
    }


def post_split_leaks(rows: list[dict[str, str]], radius: int) -> dict[str, int]:
    validation = [row for row in rows if row["split"] == "validation"]
    train = [row for row in rows if row["split"] == "train"]
    train_exact = {content_hash(row) for row in train if content_hash(row)}
    train_tree = HammingBKTree()
    train_groups = {group_key(row) for row in train if group_key(row)}
    for row in train:
        value = perceptual_hash(row)
        if len(value) == 16:
            train_tree.add(int(value, 16), row)
    exact = near = groups = 0
    for row in validation:
        digest = content_hash(row)
        if digest and digest in train_exact:
            exact += 1
        value = perceptual_hash(row)
        if len(value) == 16 and train_tree.find(int(value, 16), radius) is not None:
            near += 1
        group = group_key(row)
        if group and group in train_groups:
            groups += 1
    return {"exact": exact, "near": near, "group": groups}


def build(args: argparse.Namespace) -> dict[str, object]:
    inputs = {
        "stage83": args.stage83_manifest,
        "stage89": args.stage89_manifest,
        "stage91": args.stage91_manifest,
        "labels": args.labels,
    }
    expected = {
        "stage83": args.expected_stage83_sha256.lower(),
        "stage89": args.expected_stage89_sha256.lower(),
        "stage91": args.expected_stage91_sha256.lower(),
        "labels": args.expected_labels_sha256.lower(),
    }
    actual = {name: sha256(path) for name, path in inputs.items()}
    mismatches = [name for name in inputs if actual[name].lower() != expected[name]]
    if mismatches:
        raise RuntimeError(f"immutable input SHA256 mismatch: {mismatches}")
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if labels.get("body_types") != ["light_truck", "heavy_truck", "unknown"]:
        raise RuntimeError("truck subtype labels contract mismatch")
    if labels.get("colors") != ["unknown"]:
        raise RuntimeError("specialist color contract mismatch")

    source_rows: dict[str, list[dict[str, str]]] = {}
    fields: list[str] = []
    for name, path in inputs.items():
        if name == "labels":
            continue
        manifest_fields, rows = read_manifest(path)
        fields.extend(manifest_fields)
        source_rows[name] = rows
    if any(row.get("split", "").strip().lower() == "test" for rows in source_rows.values() for row in rows):
        raise RuntimeError("test rows are forbidden in Stage93 inputs")

    selected: list[dict[str, str]] = []
    source_selection = Counter()
    for row in source_rows["stage83"]:
        if eligible_base(row):
            selected.append(normalize(row, origin="stage83", manifest=inputs["stage83"], safety_root=args.safety_root))
            source_selection[f"stage83:{row.get('split')}:{row.get('body_type')}"] += 1
    for name, allowed, origin in (
        ("stage89", STAGE89_ALLOWED, "stage89_mio"),
        ("stage91", STAGE91_ALLOWED, "stage91_inatrc"),
    ):
        for row in source_rows[name]:
            if eligible_addition(row, allowed):
                selected.append(normalize(row, origin=origin, manifest=inputs[name], safety_root=args.safety_root))
                source_selection[f"{name}:train:{row.get('body_type')}"] += 1

    failures: list[str] = []
    frozen_rows = sum(any(marker in " ".join(row.values()).lower() for marker in FROZEN_MARKERS) for row in selected)
    if frozen_rows:
        failures.append(f"frozen markers found: {frozen_rows}")
    unreadable_train = 0
    unreadable_validation = 0
    readable: list[dict[str, str]] = []
    for row in selected:
        if readable_image(Path(row["image_path"])):
            readable.append(row)
        elif row["split"] == "validation":
            unreadable_validation += 1
        else:
            unreadable_train += 1
    if unreadable_validation:
        failures.append(f"unreadable validation images: {unreadable_validation}")

    deduplicated, dedup = deduplicate_validation_first(readable, args.near_duplicate_hamming)
    leaks = post_split_leaks(deduplicated, args.near_duplicate_hamming)
    if any(leaks.values()):
        failures.append(f"post-split leaks: {leaks}")
    counts = Counter(f"{row['split']}:{row['body_type']}" for row in deduplicated)
    for label in sorted(FINE_LABELS):
        if counts[f"train:{label}"] < args.minimum_train_per_class:
            failures.append(f"train {label} {counts[f'train:{label}']} < {args.minimum_train_per_class}")
        if counts[f"validation:{label}"] < args.minimum_validation_per_class:
            failures.append(f"validation {label} {counts[f'validation:{label}']} < {args.minimum_validation_per_class}")
    if any(row["body_type"] not in FINE_LABELS for row in deduplicated):
        failures.append("non-fine body label survived")
    if any(truthy(row.get("color_supervised")) or row.get("color") != "unknown" for row in deduplicated):
        failures.append("color supervision survived")

    report: dict[str, object] = {
        "schema_version": "stage93-truck-subtype-manifest-v1",
        "status": "pass" if not failures else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": actual[name]}
            for name, path in inputs.items()
        },
        "selection": {
            "source_counts": dict(sorted(source_selection.items())),
            "fine_labels": sorted(FINE_LABELS),
            "stage89_allowed": sorted([list(value) for value in STAGE89_ALLOWED]),
            "stage91_allowed": sorted([list(value) for value in STAGE91_ALLOWED]),
            "generic_truck_rows_allowed": 0,
        },
        "output": {
            "rows": len(deduplicated),
            "counts": dict(sorted(counts.items())),
            "unreadable_train_dropped": unreadable_train,
            "unreadable_validation": unreadable_validation,
        },
        "integrity": {
            "near_duplicate_hamming": args.near_duplicate_hamming,
            "dedup": dedup,
            "post_split_leaks": leaks,
            "frozen_markers": frozen_rows,
        },
        "policy": {
            "stage83_validation_is_only_validation_source": True,
            "train_filtered_against_validation_not_reverse": True,
            "generic_and_coarse_truck_excluded_from_fine_head": True,
            "stage89_and_stage91_are_train_only": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
            "eligibility": "research-only_non-deployable_upper-bound_candidate",
        },
        "failures": failures,
    }
    return {"report": report, "rows": deduplicated, "fields": list(dict.fromkeys([
        *fields, "stage93_original_review_status", "stage93_origin",
        "stage93_previous_sample_weight", "stage93_fine_truth_source",
        "stage93_eligibility", "research_only", "deployment_eligible", "sample_weight",
    ]))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage83-manifest", type=Path, required=True)
    parser.add_argument("--expected-stage83-sha256", required=True)
    parser.add_argument("--stage89-manifest", type=Path, required=True)
    parser.add_argument("--expected-stage89-sha256", required=True)
    parser.add_argument("--stage91-manifest", type=Path, required=True)
    parser.add_argument("--expected-stage91-sha256", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-labels-sha256", required=True)
    parser.add_argument("--safety-root", type=Path, required=True)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    parser.add_argument("--minimum-train-per-class", type=int, default=10000)
    parser.add_argument("--minimum-validation-per-class", type=int, default=300)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage93 evidence")
    result = build(args)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(result["report"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if result["report"]["status"] != "pass":
        return 1
    with args.output_manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result["fields"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["rows"])
    result["report"]["output"]["manifest"] = str(args.output_manifest.resolve())
    result["report"]["output"]["manifest_sha256"] = sha256(args.output_manifest)
    args.output_report.write_text(json.dumps(result["report"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "rows": len(result["rows"]), "counts": result["report"]["output"]["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
