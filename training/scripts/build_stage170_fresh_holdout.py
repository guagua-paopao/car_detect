#!/usr/bin/env python3
"""Build and seal a fresh Stage160-policy holdout after Stage169 authorization.

The builder performs no model inference.  It refuses to read candidate source
manifests until the immutable Stage169 component gate authorizes construction.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
import numpy as np


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "frozen_video")
KNOWN_BODY = {"sedan", "suv", "mpv", "van", "pickup", "truck", "bus", "light_truck", "heavy_truck", "other"}
KNOWN_COLOR = {"black", "white", "gray", "silver", "red", "blue", "green", "yellow", "brown", "other"}
RARE_BODY = {"pickup", "truck"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-state", type=Path, required=True)
    parser.add_argument("--expected-authorization-state-sha256", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, action="append", required=True)
    parser.add_argument("--expected-source-manifest-sha256", action="append", required=True)
    parser.add_argument("--reference-manifest", type=Path, action="append", required=True)
    parser.add_argument("--expected-reference-manifest-sha256", action="append", required=True)
    parser.add_argument("--allowed-image-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, role: str) -> None:
    if not path.is_file() or sha256(path).lower() != expected.lower():
        raise RuntimeError(f"immutable {role} SHA256 mismatch: {path}")


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def valid_hex(value: str, width: int) -> bool:
    return len(value) == width and all(character in "0123456789abcdefABCDEF" for character in value)


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class BKTree:
    def __init__(self) -> None:
        self.root: list[Any] | None = None

    def add(self, value: int, group: str) -> None:
        if self.root is None:
            self.root = [value, {group}, {}]
            return
        node = self.root
        while True:
            distance = hamming(value, node[0])
            if distance == 0:
                node[1].add(group)
                return
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = [value, {group}, {}]
                return
            node = child

    def conflict(self, value: int, radius: int, group: str) -> bool:
        if self.root is None:
            return False
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = hamming(value, node[0])
            if distance <= radius and any(existing != group for existing in node[1]):
                return True
            low, high = distance - radius, distance + radius
            stack.extend(child for edge, child in node[2].items() if low <= edge <= high)
        return False


def load_authorization(path: Path, expected_sha256: str) -> dict[str, Any]:
    require_sha(path, expected_sha256, "Stage169 authorization state")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("status") != "integrated_candidate_pass_stage160_holdout_build_authorized":
        raise RuntimeError("Stage169 did not pass and authorize a fresh holdout build")
    authorization = state.get("authorization", {})
    if authorization.get("stage160_new_holdout_build") is not True:
        raise RuntimeError("Stage169 fresh-holdout authorization is missing")
    for key in ("independent_test_inference", "onnx_backend_gates", "frozen_video_replay", "deployment"):
        if authorization.get(key) is not False:
            raise RuntimeError(f"unsafe Stage169 authorization: {key}")
    for key in ("test_accessed", "frozen_video_used", "production_model_modified", "backend_gates_run", "deployment_performed"):
        if state.get(key) is not False:
            raise RuntimeError(f"unsafe Stage169 state: {key}")
    return state


def row_group(row: dict[str, str]) -> str:
    source = row.get("source_dataset") or row.get("source")
    camera = row.get("camera_id") or row.get("camera") or "unknown-camera"
    global_identity = (
        row.get("vehicle_id") or row.get("identity_id")
        or row.get("advertisement_id") or row.get("vehicle_group")
    )
    global_track = (
        row.get("track_key") or row.get("track_group") or row.get("source_group")
        or row.get("stage159_validation_group") or row.get("stage157_group")
    )
    local_track = row.get("track_id")
    if not source or not (global_identity or global_track or local_track):
        raise RuntimeError("candidate row lacks source and vehicle/track/advertisement identity")
    if global_identity:
        return f"{source}|identity:{global_identity}"
    if global_track:
        return f"{source}|track:{global_track}"
    return f"{source}|{camera}|track:{local_track}"


def digest_fields(row: dict[str, str]) -> tuple[str, str]:
    exact = (row.get("sha256") or row.get("crop_sha256") or row.get("source_image_sha256") or "").lower()
    perceptual = (row.get("dhash64") or row.get("crop_dhash64") or "").lower()
    if not valid_hex(exact, 64) or not valid_hex(perceptual, 16):
        raise RuntimeError("candidate/reference row lacks valid SHA256 or dHash64")
    return exact, perceptual


def resolve_image(row: dict[str, str], manifest: Path, allowed_roots: list[Path]) -> Path:
    raw = Path(row.get("image_path", ""))
    resolved = raw.resolve() if raw.is_absolute() else (manifest.parent / raw).resolve()
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise RuntimeError(f"candidate image escapes allowed roots: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def dhash64(path: Path) -> str:
    with Image.open(path) as image:
        image.load()
        values = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    bits = values[:, 1:] > values[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def explicit_truth(row: dict[str, str]) -> bool:
    evidence = " ".join(
        str(row.get(field, ""))
        for field in ("ground_truth_source", "review_method", "annotation_source", "label_provenance")
    ).strip().lower()
    if not evidence or any(marker in evidence for marker in ("teacher", "pseudo", "model_prediction", "self_train")):
        return False
    return True


def post_build_authorization(passed: bool) -> dict[str, bool]:
    return {
        "one_time_independent_test_inference": passed,
        "onnx_backend_gates": False,
        "frozen_video_replay": False,
        "deployment": False,
    }


def hard_flags(row: dict[str, str]) -> tuple[bool, bool, bool, bool]:
    lighting = str(row.get("lighting", "")).lower()
    night = truthy(row.get("night")) or lighting in {"night", "low_light", "low-light"}
    small = str(row.get("vehicle_size", "")).lower() == "small"
    obstructed = any(truthy(row.get(field)) for field in ("occluded", "truncated", "overlap", "overlapped")) or str(row.get("occlusion_level", "")).lower() in {"medium", "high", "severe"}
    adverse_weather = str(row.get("weather", "")).lower() in {"rain", "rainy", "fog", "foggy", "snow", "storm"}
    blurred = str(row.get("blur", "")).lower() not in {"", "none", "false", "0", "clear", "unknown"}
    return night, small, obstructed, night or small or obstructed or adverse_weather or blurred


def load_reference_index(paths: Iterable[Path], expected_hashes: Iterable[str]) -> tuple[set[str], set[str], BKTree, list[dict[str, Any]]]:
    paths = list(paths)
    expected_hashes = list(expected_hashes)
    if len(paths) != len(expected_hashes):
        raise RuntimeError("reference manifest paths and SHA256 pins differ in count")
    groups: set[str] = set()
    exact: set[str] = set()
    tree = BKTree()
    evidence = []
    for path, expected in zip(paths, expected_hashes):
        if any(marker in str(path).lower() for marker in FROZEN_MARKERS):
            raise RuntimeError(f"forbidden reference manifest: {path}")
        require_sha(path, expected, "reference manifest")
        rows = 0
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                try:
                    group = row_group(row)
                    groups.add(group)
                except RuntimeError:
                    group = f"reference:{path.name}:{rows}"
                try:
                    row_sha, row_dhash = digest_fields(row)
                except RuntimeError:
                    continue
                exact.add(row_sha)
                tree.add(int(row_dhash, 16), group)
        evidence.append({"path": str(path.resolve()), "sha256": sha256(path), "rows": rows})
    return groups, exact, tree, evidence


def audit_sources(paths: list[Path], expected_hashes: list[str], allowed_roots: list[Path], reference_groups: set[str], exact: set[str], tree: BKTree) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if len(paths) != len(expected_hashes):
        raise RuntimeError("source manifest paths and SHA256 pins differ in count")
    accepted: list[dict[str, str]] = []
    rejections: Counter[str] = Counter()
    source_evidence = []
    for manifest, expected in zip(paths, expected_hashes):
        require_sha(manifest, expected, "candidate source manifest")
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise RuntimeError(f"source manifest lacks a header: {manifest}")
            rows = list(reader)
        source_evidence.append({"path": str(manifest.resolve()), "sha256": sha256(manifest), "rows": len(rows)})
        for row in rows:
            searchable = " ".join(str(value).lower() for value in row.values())
            if any(marker in searchable for marker in FROZEN_MARKERS):
                raise RuntimeError("frozen marker found in candidate source row")
            if str(row.get("review_status", "")).lower() != "approved" or not explicit_truth(row):
                rejections["unapproved_or_nonexplicit_truth"] += 1
                continue
            if not row.get("source_license") or str(row.get("license_verified", "")).lower() not in {"true", "yes", "1"}:
                rejections["license_not_verified"] += 1
                continue
            body_supervised = truthy(row.get("body_type_supervised"))
            color_supervised = truthy(row.get("color_supervised"))
            if not body_supervised and not color_supervised:
                rejections["no_supervised_attribute"] += 1
                continue
            if body_supervised and row.get("body_type") not in KNOWN_BODY:
                rejections["invalid_body_truth"] += 1
                continue
            if color_supervised and row.get("color") not in KNOWN_COLOR:
                rejections["invalid_color_truth"] += 1
                continue
            group = row_group(row)
            if group in reference_groups:
                rejections["historical_group_overlap"] += 1
                continue
            row_sha, row_dhash = digest_fields(row)
            if row_sha in exact:
                rejections["historical_or_candidate_exact_overlap"] += 1
                continue
            image = resolve_image(row, manifest, allowed_roots)
            if sha256(image) != row_sha or dhash64(image) != row_dhash:
                rejections["image_digest_mismatch"] += 1
                continue
            value = int(row_dhash, 16)
            if tree.conflict(value, 4, group):
                rejections["historical_or_cross_group_dhash_le_4"] += 1
                continue
            exact.add(row_sha)
            tree.add(value, group)
            output = dict(row)
            output["image_path"] = str(image)
            output["split"] = "test"
            output["stage170_group"] = group
            accepted.append(output)
    return accepted, {"sources": source_evidence, "rejections": dict(rejections)}


def summarize_support(rows: list[dict[str, str]], policy: dict[str, Any]) -> dict[str, Any]:
    body_rows = Counter()
    body_groups: defaultdict[str, set[str]] = defaultdict(set)
    color_rows = Counter()
    color_groups: defaultdict[str, set[str]] = defaultdict(set)
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    hard = Counter()
    for row in rows:
        group = row["stage170_group"]
        grouped[group].append(row)
        if truthy(row.get("body_type_supervised")):
            body_rows[row["body_type"]] += 1
            body_groups[row["body_type"]].add(group)
        if truthy(row.get("color_supervised")):
            color_rows[row["color"]] += 1
            color_groups[row["color"]].add(group)
        for key, flag in zip(("night", "small", "obstructed", "complex"), hard_flags(row)):
            hard[key] += int(flag)
    ordered_tracks = 0
    truth_conflicts = 0
    for group_rows in grouped.values():
        order_values = [row.get("frame_index") or row.get("frame_id") or row.get("timestamp") or row.get("source_frame_id") for row in group_rows]
        if len(group_rows) >= 3 and all(value not in (None, "") for value in order_values) and len(set(order_values)) >= 3:
            ordered_tracks += 1
        body_truths = {
            row["body_type"] for row in group_rows if truthy(row.get("body_type_supervised"))
        }
        color_truths = {
            row["color"] for row in group_rows if truthy(row.get("color_supervised"))
        }
        truth_conflicts += int(len(body_truths) > 1 or len(color_truths) > 1)
    minimum = policy["minimum_support"]
    scenes = policy["hard_scene_minimums"]
    total = len(rows)
    gates: dict[str, bool] = {
        "body_total": sum(body_rows.values()) >= int(minimum["body_total_known_truth_rows"]),
        "color_total": sum(color_rows.values()) >= int(minimum["color_total_known_truth_rows"]),
        "ordered_tracks": ordered_tracks >= int(minimum["real_tracks_with_at_least_three_ordered_frames"]),
        "night_fraction": total > 0 and hard["night"] / total >= float(scenes["night_or_low_light_fraction"]),
        "small_fraction": total > 0 and hard["small"] / total >= float(scenes["small_target_fraction"]),
        "obstructed_fraction": total > 0 and hard["obstructed"] / total >= float(scenes["occluded_truncated_or_overlap_fraction"]),
        "complex_fraction": total > 0 and hard["complex"] / total >= float(scenes["complex_scene_fraction"]),
        "group_truth_consistency": truth_conflicts == 0,
    }
    for label in sorted(KNOWN_BODY):
        rare = label in RARE_BODY
        row_key = "body_rows_for_each_rare_pickup_or_generic_truck" if rare else "body_rows_per_common_class"
        group_key = "body_groups_for_each_rare_pickup_or_generic_truck" if rare else "body_groups_per_common_class"
        gates[f"body_rows:{label}"] = body_rows[label] >= int(minimum[row_key])
        gates[f"body_groups:{label}"] = len(body_groups[label]) >= int(minimum[group_key])
    for label in sorted(KNOWN_COLOR):
        gates[f"color_rows:{label}"] = color_rows[label] >= int(minimum["color_rows_per_class"])
        gates[f"color_groups:{label}"] = len(color_groups[label]) >= int(minimum["color_groups_per_class"])
    return {
        "rows": total,
        "groups": len(grouped),
        "body_rows": dict(sorted(body_rows.items())),
        "body_groups": {key: len(value) for key, value in sorted(body_groups.items())},
        "color_rows": dict(sorted(color_rows.items())),
        "color_groups": {key: len(value) for key, value in sorted(color_groups.items())},
        "ordered_tracks_with_at_least_three_frames": ordered_tracks,
        "groups_with_truth_conflicts": truth_conflicts,
        "hard_scene_counts": dict(hard),
        "hard_scene_fractions": {key: value / total if total else 0.0 for key, value in hard.items()},
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.output_root.exists() or Path(str(args.output_root) + ".state.json").exists():
        raise FileExistsError("refusing to overwrite Stage170 evidence")
    # Authorization and policy pins are deliberately checked before opening any
    # candidate source manifest.
    authorization = load_authorization(args.authorization_state, args.expected_authorization_state_sha256)
    require_sha(args.policy, args.expected_policy_sha256, "Stage160 policy")
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    if policy.get("status") != "policy_ready_data_access_locked":
        raise RuntimeError("Stage160 policy is not the locked ready policy")
    allowed_roots = [root.resolve() for root in args.allowed_image_root]
    if any(not root.is_dir() for root in allowed_roots):
        raise RuntimeError("an allowed image root is missing")
    reference_groups, exact, tree, references = load_reference_index(
        args.reference_manifest, args.expected_reference_manifest_sha256
    )
    rows, source_audit = audit_sources(
        args.source_manifest,
        args.expected_source_manifest_sha256,
        allowed_roots,
        reference_groups,
        exact,
        tree,
    )
    support = summarize_support(rows, policy)
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest = args.output_root / "attribute_manifest.stage170-fresh-holdout.csv"
    if rows:
        write_csv(manifest, rows)
    report = {
        "schema_version": "stage170-fresh-holdout-build-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_fresh_holdout_built_and_sealed" if rows and support["all_gates_pass"] else "rejected_fresh_holdout_requirements_not_met",
        "authorization_state": str(args.authorization_state.resolve()),
        "authorization_state_sha256": sha256(args.authorization_state),
        "stage169_candidate_status": authorization["status"],
        "policy": str(args.policy.resolve()),
        "policy_sha256": sha256(args.policy),
        "references": references,
        "source_audit": source_audit,
        "support": support,
        "output_manifest": str(manifest.resolve()) if rows else None,
        "output_manifest_sha256": sha256(manifest) if rows else None,
        "integrity": {
            "historical_group_overlap": 0,
            "historical_or_candidate_exact_overlap": 0,
            "historical_or_cross_group_dhash_distance_le_4": 0,
        },
        "authorization": post_build_authorization(bool(rows) and support["all_gates_pass"]),
        "execution_policy": {
            "test_inference_run": False,
            "test_used_for_selection": False,
            "threshold_or_temperature_search": False,
            "stage148_reused": False,
            "stage155_reused": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "backend_gates_run": False,
            "deployment_performed": False,
        },
    }
    report_path = args.output_root / "stage170-fresh-holdout-build-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state = {
        "schema_version": "stage170-fresh-holdout-state-v1",
        "created_at": report["created_at"],
        "status": "fresh_holdout_sealed_pending_one_time_inference" if report["status"].startswith("pass_") else report["status"],
        "manifest": report["output_manifest"],
        "manifest_sha256": report["output_manifest_sha256"],
        "report": str(report_path.resolve()),
        "report_sha256": sha256(report_path),
        "authorization": report["authorization"],
        **report["execution_policy"],
    }
    state_path = Path(str(args.output_root) + ".state.json")
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sums = args.output_root / "SHA256SUMS"
    entries = [(report_path, report_path.name), (state_path, state_path.name)]
    if rows:
        entries.insert(0, (manifest, manifest.name))
    sums.write_text("".join(f"{sha256(path)}  {name}\n" for path, name in entries), encoding="utf-8")
    print(json.dumps({"status": state["status"], "rows": len(rows), "failed_gates": [key for key, value in support["gates"].items() if not value]}, ensure_ascii=False))
    return 0 if state["status"] == "fresh_holdout_sealed_pending_one_time_inference" else 2


if __name__ == "__main__":
    raise SystemExit(main())
