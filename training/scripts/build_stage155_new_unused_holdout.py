#!/usr/bin/env python3
"""Build the one-time Stage155 holdout without model inference or threshold tuning."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import tarfile
from typing import Any, Iterable

import numpy as np
from PIL import Image


MIO_MAP = {
    "articulated_truck": "heavy_truck",
    "bus": "bus",
    "pickup_truck": "light_truck",
    "single_unit_truck": "heavy_truck",
    "work_van": "van",
}
MIO_QUOTAS = {
    "articulated_truck": 350,
    "bus": 350,
    "pickup_truck": 350,
    "single_unit_truck": 250,
    "work_van": 350,
}
MIO_MIN_COUNTS = {
    "articulated_truck": 20,
    "bus": 20,
    "pickup_truck": 100,
    "single_unit_truck": 0,
    "work_van": 15,
}
FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48", "36_48", "frozen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mio-archive", type=Path, required=True)
    parser.add_argument("--expected-mio-archive-sha256", required=True)
    parser.add_argument("--vehicle-rear-archive", type=Path, required=True)
    parser.add_argument("--expected-vehicle-rear-archive-sha256", required=True)
    parser.add_argument("--vehicle-rear-plan", type=Path, required=True)
    parser.add_argument("--expected-vehicle-rear-plan-sha256", required=True)
    parser.add_argument("--vehicle-rear-metadata", type=Path, required=True)
    parser.add_argument("--expected-vehicle-rear-metadata-sha256", required=True)
    parser.add_argument("--vehicle-rear-readme", type=Path, required=True)
    parser.add_argument("--expected-vehicle-rear-readme-sha256", required=True)
    parser.add_argument("--vehicle-rear-license", type=Path, required=True)
    parser.add_argument("--expected-vehicle-rear-license-sha256", required=True)
    parser.add_argument("--reference-manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, role: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    actual = sha256_file(path)
    if actual != expected.lower():
        raise RuntimeError(f"{role} SHA256 mismatch: expected={expected} actual={actual}")
    return actual


def valid_hex(value: str, width: int) -> bool:
    return len(value) == width and all(ch in "0123456789abcdefABCDEF" for ch in value)


def distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class BKTree:
    def __init__(self) -> None:
        self.root: list[Any] | None = None

    def add(self, value: int, tag: str) -> None:
        if self.root is None:
            self.root = [value, {tag}, {}]
            return
        node = self.root
        while True:
            delta = distance(value, node[0])
            if delta == 0:
                node[1].add(tag)
                return
            child = node[2].get(delta)
            if child is None:
                node[2][delta] = [value, {tag}, {}]
                return
            node = child

    def conflict(self, value: int, radius: int, tag: str) -> bool:
        if self.root is None:
            return False
        stack = [self.root]
        while stack:
            node = stack.pop()
            delta = distance(value, node[0])
            if delta <= radius and any(existing != tag for existing in node[1]):
                return True
            low, high = delta - radius, delta + radius
            stack.extend(child for edge, child in node[2].items() if low <= edge <= high)
        return False


def dhash64(image: Image.Image) -> str:
    values = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    bits = values[:, 1:] > values[:, :-1]
    number = 0
    for bit in bits.reshape(-1):
        number = (number << 1) | int(bit)
    return f"{number:016x}"


def image_stats(payload: bytes) -> dict[str, Any]:
    with Image.open(BytesIO(payload)) as image:
        image.load()
        rgb = image.convert("RGB")
        gray = np.asarray(rgb.convert("L"), dtype=np.float32)
        width, height = rgb.size
        horizontal = np.diff(gray, axis=1)
        vertical = np.diff(gray, axis=0)
        edge_variance = float(horizontal.var() + vertical.var())
        mean = float(gray.mean())
        return {
            "width": width,
            "height": height,
            "gray_mean": mean,
            "gray_stddev": float(gray.std()),
            "edge_variance": edge_variance,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "dhash64": dhash64(rgb),
            "lighting": "night" if mean < 55 else ("low_light" if mean < 85 else "daylight"),
            "night": mean < 55,
            "vehicle_size": "small" if min(width, height) < 96 or width * height < 20000 else ("medium" if width * height < 80000 else "large"),
        }


def manifests_are_safe(paths: Iterable[Path]) -> None:
    for path in paths:
        searchable = str(path).lower()
        if any(marker in searchable for marker in FROZEN_MARKERS):
            raise RuntimeError(f"forbidden reference manifest: {path}")


def load_reference_index(paths: list[Path]) -> tuple[set[str], BKTree, set[str], list[dict[str, Any]]]:
    exact: set[str] = set()
    tree = BKTree()
    used_mio: set[str] = set()
    evidence: list[dict[str, Any]] = []
    manifests_are_safe(paths)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = 0
        mio_rows = 0
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows += 1
                sha = (row.get("sha256") or row.get("crop_sha256") or row.get("source_sha256") or "").lower()
                dhash = (row.get("dhash64") or row.get("crop_dhash64") or "").lower()
                if valid_hex(sha, 64):
                    exact.add(sha)
                if valid_hex(dhash, 16):
                    tree.add(int(dhash, 16), "reference")
                searchable = " ".join((row.get("source_dataset", ""), row.get("image_path", ""), row.get("source_manifest", ""))).lower()
                if "mio" in searchable:
                    value = row.get("source_frame_id", "") or row.get("source_image_id", "") or row.get("image_path", "")
                    match = re.search(r"(?<!\d)(\d{8})(?!\d)", value)
                    if match:
                        used_mio.add(match.group(1))
                        mio_rows += 1
        evidence.append({"path": str(path), "sha256": sha256_file(path), "rows": rows, "mio_rows": mio_rows})
    return exact, tree, used_mio, evidence


def deterministic_key(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}:{value}".encode()).hexdigest()


def load_mio_ground_truth(archive: tarfile.TarFile) -> dict[str, str]:
    handle = archive.extractfile("gt_train.csv")
    if handle is None:
        raise RuntimeError("MIO archive lacks gt_train.csv")
    truth: dict[str, str] = {}
    for raw in handle:
        parts = raw.decode("utf-8").strip().split(",")
        if len(parts) == 2:
            truth[parts[0]] = parts[1]
    if len(truth) != 519164:
        raise RuntimeError(f"unexpected MIO ground-truth rows: {len(truth)}")
    return truth


def audit_mio(
    archive_path: Path,
    used_mio: set[str],
    exact: set[str],
    tree: BKTree,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accepted_by_class: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    rejections = Counter()
    with tarfile.open(archive_path) as archive:
        truth = load_mio_ground_truth(archive)
        available = defaultdict(list)
        for image_id, source_label in truth.items():
            if source_label in MIO_MAP and image_id not in used_mio:
                available[source_label].append(image_id)
        for source_label, quota in MIO_QUOTAS.items():
            candidates = sorted(available[source_label], key=lambda item: deterministic_key("stage155-mio-pool", item))
            pool_limit = min(len(candidates), max(quota * 4, quota + 100))
            for image_id in candidates[:pool_limit]:
                member = f"train/{source_label}/{image_id}.jpg"
                extracted = archive.extractfile(member)
                if extracted is None:
                    rejections["missing_member"] += 1
                    continue
                payload = extracted.read()
                try:
                    stats = image_stats(payload)
                except Exception:
                    rejections["decode_failure"] += 1
                    continue
                if stats["sha256"] in exact:
                    rejections["reference_exact"] += 1
                    continue
                value = int(stats["dhash64"], 16)
                tag = f"mio:{image_id}"
                if tree.conflict(value, 4, tag):
                    rejections["reference_or_cross_candidate_dhash_le_4"] += 1
                    continue
                exact.add(stats["sha256"])
                tree.add(value, tag)
                accepted_by_class[source_label].append({
                    "source_image_id": image_id,
                    "source_label": source_label,
                    "body_type": MIO_MAP[source_label],
                    "payload_member": member,
                    **stats,
                })
    selected: list[dict[str, Any]] = []
    for source_label, quota in MIO_QUOTAS.items():
        pool = accepted_by_class[source_label]
        effective_quota = min(quota, len(pool))
        if effective_quota < MIO_MIN_COUNTS[source_label]:
            counts = {key: len(value) for key, value in accepted_by_class.items()}
            raise RuntimeError(
                f"MIO independent minimum failed for {source_label}: {effective_quota} < {MIO_MIN_COUNTS[source_label]}; accepted={counts}"
            )
        low = sorted((row for row in pool if row["gray_mean"] < 85), key=lambda row: deterministic_key("stage155-mio-low", row["source_image_id"]))
        target_low = min(len(low), math.ceil(effective_quota * 0.30))
        chosen = low[:target_low]
        chosen_ids = {row["source_image_id"] for row in chosen}
        remainder = sorted((row for row in pool if row["source_image_id"] not in chosen_ids), key=lambda row: deterministic_key("stage155-mio-final", row["source_image_id"]))
        chosen.extend(remainder[: effective_quota - len(chosen)])
        selected.extend(chosen)
    report = {
        "used_historical_mio_ids": len(used_mio),
        "target_caps": MIO_QUOTAS,
        "independent_minimums": MIO_MIN_COUNTS,
        "accepted_pool_counts": {key: len(value) for key, value in accepted_by_class.items()},
        "selected_counts": dict(Counter(row["source_label"] for row in selected)),
        "selected_low_light_or_night": sum(row["gray_mean"] < 85 for row in selected),
        "selected_small": sum(row["vehicle_size"] == "small" for row in selected),
        "rejections": dict(rejections),
    }
    return selected, report


def vehicle_rear_type(model: str) -> str | None:
    value = " ".join(model.lower().split())
    rules = (
        ("bus", ("busscar", "marcopolo", "volare")),
        ("heavy_truck", ("5.150",)),
        ("light_truck", ("hilux", "amarok", "courier", "saveiro", "ranger", "strada", "montana", "frontier", "bandeirante")),
        ("van", ("kombi", "master", "kangoo", "doblo", "ducato", "fiorino")),
        ("suv", ("ix35", "pajero", "outlander", "sorento", "ecosport", "crossfox", "3008")),
        ("mpv", (" idea ", "meriva", "livina", "picasso", "spacefox")),
        ("sedan", (" sedan", "corolla", "civic", "logan", "symbol", "prisma", "malibu", "cobalt", "fluence", "lancer", "versa", "320i", "408", "siena", "cruze", "voyage", "classic", "c4l", "e250")),
    )
    padded = f" {value} "
    matches = [label for label, patterns in rules if any(pattern in padded for pattern in patterns)]
    return matches[0] if len(matches) == 1 else None


def vehicle_rear_color(value: str) -> str:
    normalized = value.strip().lower()
    mapping = {
        "gray": "silver_gray", "silver": "silver_gray", "grey": "silver_gray",
        "yellow": "yellow_orange", "orange": "yellow_orange",
        "brown": "brown_beige", "beige": "brown_beige",
    }
    result = mapping.get(normalized, normalized)
    allowed = {"black", "white", "silver_gray", "red", "blue", "green", "yellow_orange", "brown_beige", "other"}
    if result not in allowed:
        raise RuntimeError(f"unsupported Vehicle-Rear color in sealed plan: {value}")
    return result


def vehicle_rear_metadata_lookup(path: Path, wanted: set[str]) -> dict[str, dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[str, dict[str, str]] = {}
    for records in raw.values():
        for record in records:
            if not isinstance(record, list) or len(record) < 7:
                continue
            attributes = [item for item in record[5:7] if isinstance(item, dict)]
            for position, members in enumerate(record[:4]):
                if not isinstance(members, list):
                    continue
                current = attributes[0] if position < 2 and attributes else (attributes[-1] if attributes else {})
                for member in members:
                    if member in wanted:
                        lookup[member] = {key: str(current.get(key, "")).strip() for key in ("brand", "model", "year", "color")}
    return lookup


def read_selected_tar_payloads(archive_path: Path, wanted: set[str]) -> dict[str, bytes]:
    """Read selected members with one sequential pass, including for gzip TAR files."""
    payloads: dict[str, bytes] = {}
    with tarfile.open(archive_path, mode="r|*") as archive:
        for member in archive:
            if member.name not in wanted:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            payloads[member.name] = handle.read()
            if len(payloads) == len(wanted):
                break
    missing = wanted - set(payloads)
    if missing:
        example = sorted(missing)[:3]
        raise RuntimeError(f"archive lacks {len(missing)} selected members; examples={example}")
    return payloads


def audit_vehicle_rear(
    archive_path: Path,
    plan_path: Path,
    metadata_path: Path,
    exact: set[str],
    tree: BKTree,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with plan_path.open("r", encoding="utf-8", newline="") as handle:
        future = [row for row in csv.DictReader(handle) if row.get("split") == "future_holdout"]
    if len(future) != 750 or len({row["track_group"] for row in future}) != 158:
        raise RuntimeError("Vehicle-Rear future holdout contract changed")
    wanted = {row["tar_member"] for row in future}
    metadata = vehicle_rear_metadata_lookup(metadata_path, wanted)
    if set(metadata) != wanted:
        raise RuntimeError(f"Vehicle-Rear metadata coverage mismatch: {len(metadata)} != {len(wanted)}")
    payloads = read_selected_tar_payloads(archive_path, wanted)
    accepted: list[dict[str, Any]] = []
    rejections = Counter()
    supervised_tracks: dict[str, str] = {}
    for source in sorted(future, key=lambda row: deterministic_key("stage155-vehicle-rear", row["tar_member"])):
        payload = payloads[source["tar_member"]]
        try:
            stats = image_stats(payload)
        except Exception:
            rejections["decode_failure"] += 1
            continue
        tag = f"vr:{source['track_group']}"
        if stats["sha256"] in exact:
            rejections["reference_exact"] += 1
            continue
        value = int(stats["dhash64"], 16)
        if tree.conflict(value, 4, tag):
            rejections["reference_or_cross_track_dhash_le_4"] += 1
            continue
        exact.add(stats["sha256"])
        tree.add(value, tag)
        attributes = metadata[source["tar_member"]]
        body_type = vehicle_rear_type(attributes["model"])
        if body_type:
            previous = supervised_tracks.setdefault(source["track_group"], body_type)
            if previous != body_type:
                raise RuntimeError("conflicting deterministic body mapping inside Vehicle-Rear track")
        accepted.append({
            "source": source,
            "attributes": attributes,
            "body_type": body_type or "unknown",
            "body_type_supervised": bool(body_type),
            "payload": payload,
            **stats,
        })
    if len(accepted) < 650:
        raise RuntimeError(f"too few decontaminated Vehicle-Rear rows: {len(accepted)}")
    report = {
        "planned_rows": len(future),
        "planned_tracks": len({row["track_group"] for row in future}),
        "accepted_rows": len(accepted),
        "accepted_tracks": len({row["source"]["track_group"] for row in accepted}),
        "color_counts": dict(Counter(row["source"]["color"] for row in accepted)),
        "body_counts": dict(Counter(row["body_type"] for row in accepted if row["body_type_supervised"])),
        "body_supervised_rows": sum(row["body_type_supervised"] for row in accepted),
        "body_supervised_tracks": len(supervised_tracks),
        "low_light_or_night_rows": sum(row["gray_mean"] < 85 for row in accepted),
        "small_rows": sum(row["vehicle_size"] == "small" for row in accepted),
        "rejections": dict(rejections),
        "mapping_policy": "closed deterministic make/model rules; every unmatched or ambiguous model remains unknown",
    }
    return accepted, report


def write_payloads_and_manifest(
    output_root: Path,
    mio_archive_path: Path,
    vehicle_rear_archive_path: Path,
    mio_rows: list[dict[str, Any]],
    vehicle_rear_rows: list[dict[str, Any]],
) -> tuple[Path, int]:
    images = output_root / "images"
    mio_dir, vr_dir = images / "mio", images / "vehicle_rear"
    mio_dir.mkdir(parents=True, exist_ok=False)
    vr_dir.mkdir(parents=True, exist_ok=False)
    output_rows: list[dict[str, Any]] = []
    with tarfile.open(mio_archive_path) as archive:
        for row in mio_rows:
            handle = archive.extractfile(row["payload_member"])
            if handle is None:
                raise RuntimeError("selected MIO member disappeared")
            payload = handle.read()
            if hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise RuntimeError("selected MIO payload hash changed")
            destination = mio_dir / f"{row['source_image_id']}.jpg"
            destination.write_bytes(payload)
            output_rows.append({
                "image_path": str(destination), "split": "test", "holdout_role": "new_unused_body",
                "body_type": row["body_type"], "body_type_supervised": "true", "color": "unknown", "color_supervised": "false",
                "source_dataset": "MIO-TCD-Classification-2017", "source_label": row["source_label"],
                "source_license": "CC-BY-NC-SA-4.0", "research_only": "true", "deployment_eligible": "false",
                "review_status": "approved", "review_method": "official_gt+unused_id+exact_and_dhash_le_4_decontamination",
                "camera_id": "unknown", "video_id": "unknown", "track_group": f"stage155_mio_{row['source_image_id']}",
                "source_frame_id": row["source_image_id"], "crop_quality": "good", "viewpoint": "unknown",
                "lighting": row["lighting"], "night": str(row["night"]).lower(), "occluded": "unknown", "truncated": "unknown", "blur": "unknown",
                "vehicle_size": row["vehicle_size"], "occlusion_level": "unknown", **{key: row[key] for key in ("width", "height", "gray_mean", "gray_stddev", "edge_variance", "sha256", "dhash64")},
                "source_model_hash": "", "source_member_hash": hashlib.sha256(row["payload_member"].encode()).hexdigest(),
            })
    for row in vehicle_rear_rows:
        source = row["source"]
        payload = row["payload"]
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise RuntimeError("selected Vehicle-Rear payload hash changed")
        token = hashlib.sha256(source["tar_member"].encode()).hexdigest()[:24]
        destination = vr_dir / f"{token}.png"
        destination.write_bytes(payload)
        output_rows.append({
                "image_path": str(destination), "split": "test", "holdout_role": "new_unused_color_and_conservative_body",
                "body_type": row["body_type"], "body_type_supervised": str(row["body_type_supervised"]).lower(),
                "color": vehicle_rear_color(source["color"]), "color_supervised": "true", "source_dataset": "Vehicle-Rear",
                "source_label": source["source_color"], "source_license": "Apache-2.0 repository scope; research-only dataset audit",
                "research_only": "true", "deployment_eligible": "false", "review_status": "approved",
                "review_method": "official_registration_metadata+sealed_vehicle_group+exact_and_dhash_le_4_decontamination+closed_model_type_map",
                "camera_id": source["camera_id"], "video_id": source["video_id"], "track_group": source["track_group"],
                "source_frame_id": token, "crop_quality": "good", "viewpoint": "rear", "lighting": row["lighting"],
                "night": str(row["night"]).lower(), "occluded": "unknown", "truncated": "unknown", "blur": "unknown",
                "vehicle_size": row["vehicle_size"], "occlusion_level": "unknown", **{key: row[key] for key in ("width", "height", "gray_mean", "gray_stddev", "edge_variance", "sha256", "dhash64")},
                "source_model_hash": hashlib.sha256(f"{row['attributes']['brand']}|{row['attributes']['model']}|{row['attributes']['year']}".encode()).hexdigest(),
                "source_member_hash": hashlib.sha256(source["tar_member"].encode()).hexdigest(),
        })
    manifest = output_root / "attribute_manifest.stage155-new-unused-holdout.csv"
    fields = list(output_rows[0])
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(output_rows)
    return manifest, len(output_rows)


def write_license_evidence(output_root: Path, mio_archive: Path, vehicle_rear_readme: Path, vehicle_rear_license: Path) -> dict[str, Any]:
    evidence = output_root / "license_evidence"
    evidence.mkdir(exist_ok=False)
    with tarfile.open(mio_archive) as archive:
        handle = archive.extractfile("README.txt")
        if handle is None:
            raise RuntimeError("MIO archive lacks embedded README license evidence")
        mio_payload = handle.read()
    paths = {
        "mio_readme": evidence / "MIO-TCD-README.txt",
        "vehicle_rear_readme": evidence / "Vehicle-Rear-OFFICIAL_README.md",
        "vehicle_rear_license": evidence / "Vehicle-Rear-OFFICIAL_LICENSE",
    }
    paths["mio_readme"].write_bytes(mio_payload)
    paths["vehicle_rear_readme"].write_bytes(vehicle_rear_readme.read_bytes())
    paths["vehicle_rear_license"].write_bytes(vehicle_rear_license.read_bytes())
    return {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in paths.items()}


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite Stage155 evidence: {args.output_root}")
    for path in (args.mio_archive, args.vehicle_rear_archive, args.vehicle_rear_plan, args.vehicle_rear_metadata,
                 args.vehicle_rear_readme, args.vehicle_rear_license, *args.reference_manifest):
        if any(marker in str(path).lower() for marker in FROZEN_MARKERS):
            raise RuntimeError(f"forbidden frozen marker in input: {path}")
    inputs = {
        "mio_archive_sha256": require_hash(args.mio_archive, args.expected_mio_archive_sha256, "MIO archive"),
        "vehicle_rear_archive_sha256": require_hash(args.vehicle_rear_archive, args.expected_vehicle_rear_archive_sha256, "Vehicle-Rear archive"),
        "vehicle_rear_plan_sha256": require_hash(args.vehicle_rear_plan, args.expected_vehicle_rear_plan_sha256, "Vehicle-Rear plan"),
        "vehicle_rear_metadata_sha256": require_hash(args.vehicle_rear_metadata, args.expected_vehicle_rear_metadata_sha256, "Vehicle-Rear metadata"),
        "vehicle_rear_readme_sha256": require_hash(args.vehicle_rear_readme, args.expected_vehicle_rear_readme_sha256, "Vehicle-Rear README"),
        "vehicle_rear_license_sha256": require_hash(args.vehicle_rear_license, args.expected_vehicle_rear_license_sha256, "Vehicle-Rear license"),
    }
    exact, tree, used_mio, references = load_reference_index(args.reference_manifest)
    mio_rows, mio_report = audit_mio(args.mio_archive, used_mio, exact, tree)
    vr_rows, vr_report = audit_vehicle_rear(args.vehicle_rear_archive, args.vehicle_rear_plan, args.vehicle_rear_metadata, exact, tree)
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest, rows = write_payloads_and_manifest(args.output_root, args.mio_archive, args.vehicle_rear_archive, mio_rows, vr_rows)
    license_evidence = write_license_evidence(args.output_root, args.mio_archive, args.vehicle_rear_readme, args.vehicle_rear_license)
    report = {
        "schema_version": "stage155-new-unused-holdout-build-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass_new_unused_holdout_ready_for_one_time_evaluation", "inputs": inputs, "reference_manifests": references,
        "output": {"manifest": str(manifest), "manifest_sha256": sha256_file(manifest), "rows": rows},
        "mio": mio_report, "vehicle_rear": vr_report, "license_evidence": license_evidence,
        "integrity": {"historical_exact_overlap": 0, "historical_or_cross_group_dhash_distance_le_4": 0, "stage148_test_reused": False},
        "policy": {"split": "test", "threshold_search_allowed": False, "model_selection_allowed": False, "research_only": True,
                   "test_accessed": True, "frozen_video_used": False, "production_model_modified": False,
                   "backend_gates_run": False, "deployment_performed": False},
    }
    report_path = args.output_root / "stage155-new-unused-holdout-build-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state = {
        "schema_version": "stage155-new-unused-holdout-state-v1", "created_at": report["created_at"],
        "status": "holdout_built_pending_one_time_evaluation", "manifest": str(manifest),
        "manifest_sha256": report["output"]["manifest_sha256"], "report": str(report_path), "report_sha256": sha256_file(report_path),
        **report["policy"],
    }
    state_path = Path(str(args.output_root) + ".state.json")
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sums = args.output_root / "SHA256SUMS"
    sums.write_text(
        f"{sha256_file(manifest)}  {manifest.name}\n{sha256_file(report_path)}  {report_path.name}\n{sha256_file(state_path)}  {state_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": state["status"], "rows": rows, "mio": mio_report["selected_counts"], "vehicle_rear": vr_report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
