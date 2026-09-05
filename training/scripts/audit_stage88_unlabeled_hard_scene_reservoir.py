from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from audit_stage87_supervised_scene_quotas import (
    FROZEN_MARKERS,
    compact_counts,
    is_night,
    is_occluded_or_truncated,
    is_small,
    sample_identity,
    sha256_file,
    supervised,
    truthy,
)


SCENES = {
    "night_or_low_light": (is_night, 0.30),
    "small_target": (is_small, 0.20),
    "occluded_overlapped_or_truncated": (is_occluded_or_truncated, 0.15),
}


def has_frozen_marker(row: dict[str, str]) -> bool:
    searchable = " ".join(
        row.get(field, "")
        for field in ("image_path", "source_manifest", "source_frame_id", "video_id")
    ).lower()
    return any(marker in searchable for marker in FROZEN_MARKERS)


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def base_supervised_state(path: Path, body_types: set[str]) -> tuple[set[str], dict[str, int]]:
    rows, fields = load_rows(path)
    required = {"image_path", "body_type", "body_type_supervised", "split", "review_status"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"base manifest missing required columns: {sorted(missing)}")
    identities: set[str] = set()
    scene_counts = Counter()
    for row in rows:
        if has_frozen_marker(row):
            raise ValueError("frozen-video marker found in base manifest")
        if row.get("split") != "train" or row.get("review_status") != "approved":
            continue
        body = row.get("body_type", "").strip().lower()
        if not supervised(row, "body_type") or body not in body_types:
            continue
        identity = sample_identity(row)
        if identity in identities:
            continue
        identities.add(identity)
        for name, (predicate, _) in SCENES.items():
            if predicate(row):
                scene_counts[name] += 1
    if not identities:
        raise ValueError("base manifest has no unique supervised train identities")
    return identities, compact_counts(scene_counts)


def candidate_is_unlabeled(row: dict[str, str]) -> bool:
    return (
        not truthy(row.get("body_type_supervised"))
        and not truthy(row.get("color_supervised"))
        and row.get("body_type", "").strip().lower() in {"", "unknown"}
        and row.get("color", "").strip().lower() in {"", "unknown"}
        and bool(row.get("image_path", "").strip())
    )


def audit_reservoir(
    base_manifest: Path,
    labels_path: Path,
    candidate_manifests: list[Path],
) -> dict[str, object]:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    body_types = {str(value).strip().lower() for value in labels["body_types"]}
    body_types.discard("unknown")
    base_ids, base_scene_counts = base_supervised_state(base_manifest, body_types)
    base_total = len(base_ids)

    eligible_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    unverified_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    per_manifest: list[dict[str, object]] = []
    nontrain_rows_total = 0
    rejected_known_or_supervised_total = 0

    for manifest in candidate_manifests:
        rows, fields = load_rows(manifest)
        required = {
            "image_path",
            "body_type",
            "body_type_supervised",
            "color",
            "color_supervised",
            "split",
            "license_train_eligible",
        }
        missing = required - set(fields)
        if missing:
            raise ValueError(f"candidate manifest {manifest} missing required columns: {sorted(missing)}")
        counts = Counter()
        for row in rows:
            if has_frozen_marker(row):
                raise ValueError(f"frozen-video marker found in candidate manifest: {manifest}")
            if row.get("split") != "train":
                counts["nontrain"] += 1
                continue
            if not candidate_is_unlabeled(row):
                counts["known_or_supervised"] += 1
                continue
            identity = sample_identity(row)
            if identity in base_ids:
                counts["overlap_with_base"] += 1
                continue
            if truthy(row.get("license_train_eligible")):
                eligible_groups[identity].append(row)
                counts["eligible_occurrences"] += 1
            else:
                unverified_groups[identity].append(row)
                counts["license_unverified_occurrences"] += 1
        nontrain_rows_total += counts["nontrain"]
        rejected_known_or_supervised_total += counts["known_or_supervised"]
        per_manifest.append(
            {
                "path": str(manifest),
                "sha256": sha256_file(manifest),
                "rows": len(rows),
                **compact_counts(counts),
            }
        )

    for identity in list(unverified_groups):
        if identity in eligible_groups:
            del unverified_groups[identity]

    eligible_scene_counts = Counter()
    source_counts = Counter()
    license_counts = Counter()
    for group in eligible_groups.values():
        for name, (predicate, _) in SCENES.items():
            if any(predicate(row) for row in group):
                eligible_scene_counts[name] += 1
        sources = {
            row.get("source_dataset", "").strip() or row.get("source_group", "").strip() or "unknown"
            for row in group
        }
        licenses = {row.get("source_license", "").strip() or "unknown" for row in group}
        for source in sources:
            source_counts[source] += 1
        for license_name in licenses:
            license_counts[license_name] += 1

    capacity: dict[str, object] = {}
    for name, (_, target) in SCENES.items():
        base_positive = base_scene_counts.get(name, 0)
        required_additions = math.ceil(max(0.0, (target * base_total - base_positive) / (1.0 - target)))
        candidate_positive = eligible_scene_counts[name]
        upper_fraction = (base_positive + candidate_positive) / (base_total + candidate_positive)
        capacity[name] = {
            "base_unique_positive_rows": base_positive,
            "target_fraction": target,
            "minimum_positive_only_additions_required": required_additions,
            "licensed_unique_unlabeled_positive_candidates": candidate_positive,
            "candidate_shortfall_before_teacher_rejection": max(0, required_additions - candidate_positive),
            "upper_bound_fraction_if_every_candidate_received_correct_exact_truth": upper_fraction,
            "can_reach_target_even_at_100_percent_acceptance": candidate_positive >= required_additions,
        }

    return {
        "schema_version": "stage88-unlabeled-hard-scene-reservoir-audit-v1",
        "status": (
            "sufficient_upper_bound_pending_teacher_truth"
            if all(item["can_reach_target_even_at_100_percent_acceptance"] for item in capacity.values())
            else "insufficient_licensed_unlabeled_reservoir"
        ),
        "base_manifest": str(base_manifest),
        "base_manifest_sha256": sha256_file(base_manifest),
        "labels": str(labels_path),
        "labels_sha256": sha256_file(labels_path),
        "base_unique_supervised_train_rows": base_total,
        "base_scene_counts": base_scene_counts,
        "candidate_manifests": per_manifest,
        "unique_licensed_unlabeled_candidates_excluding_base": len(eligible_groups),
        "unique_license_unverified_candidates_excluding_base": len(unverified_groups),
        "eligible_scene_counts": compact_counts(eligible_scene_counts),
        "source_counts": compact_counts(source_counts),
        "license_counts": compact_counts(license_counts),
        "capacity": capacity,
        "upper_bound_policy": "capacity assumes every licensed candidate receives an independently correct exact label; actual conservative teacher acceptance can only be lower",
        "nontrain_rows_skipped_without_image_access": nontrain_rows_total,
        "known_or_supervised_candidate_rows_skipped": rejected_known_or_supervised_total,
        "images_opened": False,
        "test_payload_opened": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit deduplicated licensed unlabeled hard-scene capacity.")
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_reservoir(
        args.base_manifest.resolve(),
        args.labels.resolve(),
        [path.resolve() for path in args.candidate_manifest],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{sha256_file(args.output)}  {args.output.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
