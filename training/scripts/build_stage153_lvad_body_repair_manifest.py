#!/usr/bin/env python3
"""Build a fail-closed, quota-balanced Stage153 body repair manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "stage148")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def row_identity(row: dict[str, str]) -> str:
    return str(
        row.get("sha256")
        or row.get("crop_sha256")
        or row.get("source_sha256")
        or row.get("image_path")
        or ""
    ).strip().lower()


def group_identity(row: dict[str, str]) -> str:
    return str(
        row.get("track_group")
        or row.get("video_id")
        or row.get("source_frame_id")
        or row.get("image_path")
        or ""
    ).strip()


def deterministic_key(row: dict[str, str], seed: str) -> str:
    payload = f"{seed}|{group_identity(row)}|{row_identity(row)}|{row.get('image_path', '')}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_exact_body(row: dict[str, str]) -> bool:
    return truthy(row.get("body_type_supervised")) and row.get("body_type") not in {"", "unknown"}


def is_occluded_or_truncated(row: dict[str, str]) -> bool:
    level = str(row.get("occlusion_level") or "").strip().lower()
    return truthy(row.get("occluded")) or truthy(row.get("truncated")) or level not in {"", "0", "none", "unknown"}


def is_night(row: dict[str, str]) -> bool:
    lighting = str(row.get("lighting") or "").strip().lower()
    return truthy(row.get("night")) or truthy(row.get("low_light")) or lighting in {"night", "low_light", "low-light", "official_night_low_light"}


def is_small(row: dict[str, str]) -> bool:
    return truthy(row.get("small_target")) or str(row.get("vehicle_size") or "").strip().lower() == "small"


def admissible_night_family(row: dict[str, str]) -> bool:
    """Only the publisher's unambiguous mixed-car class matches an existing coarse loss.

    L-VAD class 2 merges trucks and buses. The current truck-family loss excludes bus,
    so admitting that class would fabricate a narrower target.
    """
    return row.get("coarse_body_family") == "car" and row.get("source_label") == "mixed_car_family"


def prepare_exact_replay(row: dict[str, str]) -> dict[str, str]:
    copied = dict(row)
    copied["stage153_source_coarse_body_family"] = copied.get("coarse_body_family", "")
    copied["coarse_body_family"] = ""
    copied["stage153_origin"] = "stage83_exact_replay"
    copied["sample_weight"] = "1.0"
    return copied


def diverse_take(rows: list[dict[str, str]], limit: int, seed: str) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[group_identity(row)].append(row)
    queues = {
        group: deque(sorted(items, key=lambda item: deterministic_key(item, seed)))
        for group, items in grouped.items()
    }
    order = sorted(queues, key=lambda group: hashlib.sha256(f"{seed}|{group}".encode()).hexdigest())
    selected: list[dict[str, str]] = []
    while len(selected) < limit and order:
        next_order: list[str] = []
        for group in order:
            if len(selected) >= limit:
                break
            queue = queues[group]
            if queue:
                selected.append(queue.popleft())
            if queue:
                next_order.append(group)
        order = next_order
    return selected


class BKNode:
    def __init__(self, value: int):
        self.value = value
        self.children: dict[int, BKNode] = {}


class HammingTree:
    def __init__(self) -> None:
        self.root: BKNode | None = None

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = BKNode(value)
            return
        node = self.root
        while True:
            distance = (value ^ node.value).bit_count()
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKNode(value)
                return
            node = child

    def contains_within(self, value: int, radius: int) -> bool:
        if self.root is None:
            return False
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = (value ^ node.value).bit_count()
            if distance <= radius:
                return True
            low, high = distance - radius, distance + radius
            pending.extend(child for edge, child in node.children.items() if low <= edge <= high)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--night-manifest", type=Path, required=True)
    parser.add_argument("--expected-night-sha256", required=True)
    parser.add_argument("--night-report", type=Path, required=True)
    parser.add_argument("--expected-night-report-sha256", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--exact-per-class", type=int, default=1750)
    parser.add_argument("--occluded-per-eligible-class", type=int, default=525)
    parser.add_argument("--maximum-cross-split-dhash", type=int, default=4)
    args = parser.parse_args()

    inputs = [args.base_manifest, args.night_manifest, args.night_report]
    if any(not path.is_file() for path in inputs):
        raise FileNotFoundError([str(path) for path in inputs if not path.is_file()])
    if args.output_manifest.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Stage153 manifest evidence")
    actual_hashes = [sha256_file(path) for path in inputs]
    expected_hashes = [
        args.expected_base_sha256.lower(),
        args.expected_night_sha256.lower(),
        args.expected_night_report_sha256.lower(),
    ]
    if actual_hashes != expected_hashes:
        raise RuntimeError(f"pinned input SHA mismatch: {actual_hashes}")
    if any(marker in str(path).lower() for marker in FROZEN_MARKERS for path in inputs):
        raise RuntimeError("forbidden test or frozen marker in inputs")

    night_report = json.loads(args.night_report.read_text(encoding="utf-8"))
    policy = night_report.get("policy", {})
    if not (
        night_report.get("status") == "pass"
        and policy.get("all_outputs_train_only") is True
        and policy.get("source_validation_payloads_read") == 0
        and policy.get("source_test_payloads_read") == 0
        and policy.get("fine_body_labels_fabricated") is False
        and policy.get("frozen_video_used") is False
    ):
        raise RuntimeError("night supplement policy is not admissible")

    base_fields, base_rows = load_csv(args.base_manifest)
    night_fields, night_rows = load_csv(args.night_manifest)
    validation_rows = [dict(row) for row in base_rows if row.get("split") == "validation"]
    exact_train = [row for row in base_rows if row.get("split") == "train" and is_exact_body(row)]
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in exact_train:
        by_class[row["body_type"]].append(row)
    if len(by_class) < 9:
        raise RuntimeError(f"insufficient exact classes: {sorted(by_class)}")

    selected_exact: list[dict[str, str]] = []
    class_counts: Counter[str] = Counter()
    selected_occluded: Counter[str] = Counter()
    for label in sorted(by_class):
        candidates = by_class[label]
        if len(candidates) < args.exact_per_class:
            raise RuntimeError(f"class {label} has only {len(candidates)} exact rows")
        occluded = [row for row in candidates if is_occluded_or_truncated(row)]
        wanted_occluded = min(args.occluded_per_eligible_class, len(occluded), args.exact_per_class)
        picked_occluded = diverse_take(occluded, wanted_occluded, f"stage153|{label}|occluded")
        picked_ids = {row_identity(row) for row in picked_occluded}
        remaining = [row for row in candidates if row_identity(row) not in picked_ids]
        picked = picked_occluded + diverse_take(
            remaining,
            args.exact_per_class - len(picked_occluded),
            f"stage153|{label}|replay",
        )
        if len(picked) != args.exact_per_class:
            raise RuntimeError(f"class {label} selection shortfall")
        for row in picked:
            copied = prepare_exact_replay(row)
            selected_exact.append(copied)
            class_counts[label] += 1
            selected_occluded[label] += int(is_occluded_or_truncated(copied))

    prepared_night: list[dict[str, str]] = []
    rejected_ambiguous_commercial_rows = 0
    for row in night_rows:
        if row.get("split") != "train":
            raise RuntimeError("night supplement contains non-train row")
        if row.get("coarse_body_family") not in {"car", "truck"}:
            raise RuntimeError("night supplement contains invalid coarse family")
        if truthy(row.get("body_type_supervised")) or row.get("body_type") not in {"", "unknown"}:
            raise RuntimeError("night supplement fabricated exact body truth")
        if not admissible_night_family(row):
            rejected_ambiguous_commercial_rows += 1
            continue
        copied = dict(row)
        copied.update({
            "stage153_origin": "lvad_real_night_coarse",
            "formal_train_eligible": "true",
            "review_status": "approved_coarse_night_train_only",
            "small_target": "true",
            "vehicle_size": "small",
            "night": "true",
            "lighting": "official_night_low_light",
            "sample_weight": "1.0",
        })
        prepared_night.append(copied)

    train_rows = selected_exact + prepared_night
    train_groups = {group_identity(row) for row in train_rows}
    validation_groups = {group_identity(row) for row in validation_rows}
    group_overlap = (train_groups & validation_groups) - {""}
    train_sha = [row_identity(row) for row in train_rows]
    validation_sha = [row_identity(row) for row in validation_rows]
    exact_overlap = (set(train_sha) & set(validation_sha)) - {""}
    if len(train_sha) != len(set(train_sha)):
        raise RuntimeError("duplicate train image identities")

    validation_tree = HammingTree()
    invalid_dhash = 0
    for row in validation_rows:
        value = str(row.get("dhash64") or row.get("crop_dhash64") or "").strip().lower()
        if len(value) == 16:
            validation_tree.add(int(value, 16))
        else:
            invalid_dhash += 1
    cross_near = 0
    for row in train_rows:
        value = str(row.get("dhash64") or row.get("crop_dhash64") or "").strip().lower()
        if len(value) != 16:
            invalid_dhash += 1
            continue
        cross_near += int(validation_tree.contains_within(int(value, 16), args.maximum_cross_split_dhash))

    all_rows = train_rows + validation_rows
    night_count = sum(is_night(row) for row in train_rows)
    small_count = sum(is_small(row) for row in train_rows)
    occluded_count = sum(is_occluded_or_truncated(row) for row in train_rows)
    quotas = {
        "night_rows": night_count,
        "night_fraction": night_count / len(train_rows),
        "small_rows": small_count,
        "small_fraction": small_count / len(train_rows),
        "occluded_or_truncated_rows": occluded_count,
        "occluded_or_truncated_fraction": occluded_count / len(train_rows),
    }
    failures: list[str] = []
    if group_overlap:
        failures.append(f"train/validation group overlap: {len(group_overlap)}")
    if exact_overlap:
        failures.append(f"train/validation exact SHA overlap: {len(exact_overlap)}")
    if cross_near:
        failures.append(f"train rows near validation dHash<={args.maximum_cross_split_dhash}: {cross_near}")
    if invalid_dhash:
        failures.append(f"invalid dHash rows: {invalid_dhash}")
    if quotas["night_fraction"] < 0.30:
        failures.append("night fraction below 0.30")
    if quotas["small_fraction"] < 0.20:
        failures.append("small fraction below 0.20")
    if quotas["occluded_or_truncated_fraction"] < 0.15:
        failures.append("occluded/truncated fraction below 0.15")

    report = {
        "schema_version": "stage153-lvad-body-repair-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_training_manifest_ready" if not failures else "fail_closed",
        "inputs": {
            "base_manifest": str(args.base_manifest.resolve()),
            "base_manifest_sha256": actual_hashes[0],
            "night_manifest": str(args.night_manifest.resolve()),
            "night_manifest_sha256": actual_hashes[1],
            "night_report": str(args.night_report.resolve()),
            "night_report_sha256": actual_hashes[2],
        },
        "selection": {
            "exact_per_class": args.exact_per_class,
            "occluded_per_eligible_class": args.occluded_per_eligible_class,
            "exact_class_counts": dict(sorted(class_counts.items())),
            "selected_occluded_by_class": dict(sorted(selected_occluded.items())),
            "group_diverse_round_robin": True,
        },
        "output": {
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "exact_replay_rows": len(selected_exact),
            "real_night_coarse_rows": len(prepared_night),
            "rejected_ambiguous_truck_bus_rows": rejected_ambiguous_commercial_rows,
            "coarse_family_counts": dict(sorted(Counter(row["coarse_body_family"] for row in prepared_night).items())),
            "quotas": quotas,
        },
        "leakage": {
            "train_validation_group_overlap": len(group_overlap),
            "train_validation_exact_sha_overlap": len(exact_overlap),
            "train_rows_near_validation_dhash": cross_near,
            "maximum_cross_split_dhash": args.maximum_cross_split_dhash,
            "invalid_dhash_rows": invalid_dhash,
        },
        "policy": {
            "night_rows_coarse_only": True,
            "exact_replay_coarse_metadata_cleared": True,
            "mixed_truck_bus_rows_excluded": True,
            "commercial_union_loss_not_fabricated": True,
            "fine_labels_fabricated": False,
            "source_validation_payloads_read": 0,
            "source_test_payloads_read": 0,
            "stage148_test_reused": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "training_started": False,
            "deployment_performed": False,
        },
        "failures": failures,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        print(json.dumps({"status": "fail_closed", "failures": failures}, ensure_ascii=False))
        return 2

    fields = list(base_fields)
    for field in [*night_fields, "stage153_origin", "stage153_source_coarse_body_family", "formal_train_eligible", "small_target", "vehicle_size", "sample_weight"]:
        if field not in fields:
            fields.append(field)
    with args.output_manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    report["output"].update({
        "manifest": str(args.output_manifest.resolve()),
        "manifest_sha256": sha256_file(args.output_manifest),
    })
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["output"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
