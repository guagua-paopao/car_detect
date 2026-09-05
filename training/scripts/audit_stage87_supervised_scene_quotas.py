from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


TRUE_VALUES = {"1", "true", "yes", "y"}
FALSE_VALUES = {"", "0", "false", "no", "n"}
UNKNOWN_SCENE_VALUES = {"unknown", "unset", "na", "n/a", "none", "null"}
FROZEN_MARKERS = {
    "vcas_rtsp_demo_60s",
    "36-48s",
    "36_48s",
    "36–48",
}
NIGHT_LIGHTING = {"night", "nighttime", "low_light", "lowlight", "dark"}
OCCLUDED_LEVELS = {"partial", "partially_occluded", "heavy", "severe", "occluded"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def scene_truthy(value: str | None) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized in UNKNOWN_SCENE_VALUES:
        return False
    return truthy(value)


def supervised(row: dict[str, str], head: str) -> bool:
    field = f"{head}_supervised"
    if field not in row or not row[field].strip():
        return True
    return truthy(row[field])


def is_night(row: dict[str, str]) -> bool:
    return (
        scene_truthy(row.get("night"))
        or scene_truthy(row.get("low_light"))
        or row.get("lighting", "").strip().lower() in NIGHT_LIGHTING
    )


def is_small(row: dict[str, str]) -> bool:
    return scene_truthy(row.get("small_target")) or row.get("vehicle_size", "").strip().lower() == "small"


def is_occluded_or_truncated(row: dict[str, str]) -> bool:
    tags = {tag.strip().lower() for tag in row.get("hard_mining_tags", "").replace(",", ";").split(";")}
    return (
        scene_truthy(row.get("occluded"))
        or scene_truthy(row.get("truncated"))
        or row.get("occlusion_level", "").strip().lower() in OCCLUDED_LEVELS
        or bool(tags & {"occluded", "occlusion", "truncated", "truncation", "overlap", "overlapping"})
    )


def sample_identity(row: dict[str, str]) -> str:
    for field in ("crop_sha256", "sha256", "source_sha256"):
        value = row.get(field, "").strip().lower()
        if value:
            return f"sha256:{value}"
    image_path = row.get("image_path", "").strip()
    if not image_path:
        raise ValueError("loader-usable row has neither a content hash nor image_path")
    return f"path:{image_path}"


def compact_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def summarize_rows(rows: Iterable[dict[str, str]], body_types: set[str]) -> dict[str, object]:
    selected: list[dict[str, str]] = []
    for row in rows:
        if row.get("split") != "train":
            continue
        if row.get("review_status") != "approved":
            continue
        body_type = row.get("body_type", "").strip().lower()
        if not supervised(row, "body_type") or body_type not in body_types:
            continue
        selected.append(row)

    identity_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        identity_groups[sample_identity(row)].append(row)

    unique_rows = [group[0] for group in identity_groups.values()]
    duplicate_rows = len(selected) - len(unique_rows)
    total = len(unique_rows)
    if not total:
        raise ValueError("manifest has no unique loader-usable supervised body train rows")

    flags = {
        "night_or_low_light": is_night,
        "small_target": is_small,
        "occluded_overlapped_or_truncated": is_occluded_or_truncated,
    }
    targets = {
        "night_or_low_light": 0.30,
        "small_target": 0.20,
        "occluded_overlapped_or_truncated": 0.15,
    }

    quota: dict[str, object] = {}
    for name, predicate in flags.items():
        count = sum(predicate(row) for row in unique_rows)
        required = math.ceil(total * targets[name])
        positive_only_additions = math.ceil(max(0.0, (targets[name] * total - count) / (1.0 - targets[name])))
        quota[name] = {
            "unique_rows": count,
            "fraction": count / total,
            "target_fraction": targets[name],
            "minimum_unique_rows_at_current_total": required,
            "additional_unique_rows_required": max(0, required - count),
            "minimum_positive_only_additions_to_reach_final_fraction": positive_only_additions,
            "pass": count >= required,
        }

    body_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    per_body: dict[str, Counter[str]] = defaultdict(Counter)
    per_source: dict[str, Counter[str]] = defaultdict(Counter)
    track_groups: set[str] = set()
    camera_groups: set[str] = set()
    video_groups: set[str] = set()
    missing_group_rows = 0

    for row in unique_rows:
        body = row.get("body_type", "").strip().lower()
        source = row.get("source_dataset", "").strip() or row.get("source_group", "").strip() or "unknown"
        license_name = row.get("source_license", "").strip() or "unknown"
        body_counts[body] += 1
        source_counts[source] += 1
        license_counts[license_name] += 1
        per_body[body]["total"] += 1
        per_source[source]["total"] += 1
        for name, predicate in flags.items():
            if predicate(row):
                per_body[body][name] += 1
                per_source[source][name] += 1
        group_present = False
        for field, output in (
            ("track_group", track_groups),
            ("camera_id", camera_groups),
            ("video_id", video_groups),
        ):
            value = row.get(field, "").strip()
            if value:
                output.add(value)
                group_present = True
        if not group_present:
            missing_group_rows += 1

    weighted_rows = sum(
        1
        for row in unique_rows
        if row.get("sample_weight", "").strip() not in {"", "1", "1.0", "1.00", "1.000"}
    )

    return {
        "selected_rows_before_exact_dedup": len(selected),
        "unique_loader_usable_supervised_train_rows": total,
        "exact_duplicate_rows_excluded_from_quota": duplicate_rows,
        "identity_policy": "crop_sha256 -> sha256 -> source_sha256 -> image_path; one row per identity",
        "weighted_sampling_does_not_change_unique_counts": True,
        "rows_with_non_unit_sample_weight": weighted_rows,
        "quota": quota,
        "all_scene_quotas_pass": all(item["pass"] for item in quota.values()),
        "body_type_counts": compact_counts(body_counts),
        "source_counts": compact_counts(source_counts),
        "license_counts": compact_counts(license_counts),
        "per_body_type": {key: compact_counts(per_body[key]) for key in sorted(per_body)},
        "per_source": {key: compact_counts(per_source[key]) for key in sorted(per_source)},
        "grouping": {
            "unique_track_groups": len(track_groups),
            "unique_camera_ids": len(camera_groups),
            "unique_video_ids": len(video_groups),
            "rows_missing_all_group_fields": missing_group_rows,
        },
    }


def audit_manifest(manifest: Path, labels: Path) -> dict[str, object]:
    label_data = json.loads(labels.read_text(encoding="utf-8"))
    body_types = {str(value).strip().lower() for value in label_data["body_types"]}
    body_types.discard("unknown")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "body_type", "split", "review_status"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest missing required columns: {sorted(missing)}")
        rows = list(reader)

    frozen_rows = []
    for index, row in enumerate(rows, start=2):
        searchable = " ".join(
            row.get(field, "")
            for field in ("image_path", "source_manifest", "source_frame_id", "video_id")
        ).lower()
        if any(marker in searchable for marker in FROZEN_MARKERS):
            frozen_rows.append(index)
    if frozen_rows:
        raise ValueError(f"frozen-video markers found in manifest rows: {frozen_rows[:20]}")

    split_counts = Counter(row.get("split", "") for row in rows)
    summary = summarize_rows(rows, body_types)
    return {
        "schema_version": "stage87-supervised-scene-quota-audit-v1",
        "status": "pass" if summary["all_scene_quotas_pass"] else "fail_supervised_scene_quota",
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "labels": str(labels),
        "labels_sha256": sha256_file(labels),
        "split_counts": compact_counts(split_counts),
        "frozen_markers": 0,
        "test_payload_opened": False,
        "images_opened": False,
        "production_model_modified": False,
        "deployment_performed": False,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit unique supervised hard-scene quotas without opening images.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_manifest(args.manifest.resolve(), args.labels.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    sidecar.write_text(f"{sha256_file(args.output)}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
