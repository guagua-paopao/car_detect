#!/usr/bin/env python3
"""Stream-audit Vehicle-Rear before extraction or training use."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath


TEXT_SUFFIXES = {".json", ".csv", ".txt", ".md", ".yaml", ".yml"}
MEDIA_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mkv"}
LICENSE_MARKERS = ("license", "licence", "copying", "readme")
RESTRICTIVE_MARKERS = (
    "non-commercial",
    "noncommercial",
    "academic research only",
    "research purposes only",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def summarize_json(value) -> dict:
    if isinstance(value, dict):
        return {"type": "object", "keys": sorted(map(str, value.keys()))[:100]}
    if isinstance(value, list):
        summary = {"type": "array", "length": len(value)}
        if value and isinstance(value[0], dict):
            summary["first_item_keys"] = sorted(map(str, value[0].keys()))[:100]
        return summary
    return {"type": type(value).__name__}


def collect_attribute_values(value, output: Counter[str], prefix: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key).lower()
            full_key = f"{prefix}.{key_text}" if prefix else key_text
            if any(token in key_text for token in ("color", "colour", "type", "class")):
                if isinstance(child, (str, int, float, bool)):
                    output[f"{full_key}={child}"] += 1
            collect_attribute_values(child, output, full_key)
    elif isinstance(value, list):
        for child in value[:1000]:
            collect_attribute_values(child, output, prefix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-json-samples", type=int, default=100)
    parser.add_argument("--max-text-bytes", type=int, default=5 * 1024 * 1024)
    parser.add_argument("--expected-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_hash = sha256(args.archive)
    if args.expected_sha256 and archive_hash.lower() != args.expected_sha256.lower():
        raise RuntimeError("archive SHA256 does not match the download report")

    suffix_counts: Counter[str] = Counter()
    top_counts: Counter[str] = Counter()
    attribute_values: Counter[str] = Counter()
    license_texts: dict[str, str] = {}
    json_summaries: dict[str, dict] = {}
    text_samples: dict[str, str] = {}
    metadata_members: list[dict[str, int | str]] = []
    unsafe_members: list[str] = []
    members = 0
    regular_files = 0
    total_uncompressed_bytes = 0
    with tarfile.open(args.archive, mode="r|gz") as archive:
        for member in archive:
            members += 1
            if not safe_member_name(member.name):
                unsafe_members.append(member.name)
            parts = PurePosixPath(member.name).parts
            if parts:
                top_counts[parts[0]] += 1
            if not member.isfile():
                continue
            regular_files += 1
            total_uncompressed_bytes += member.size
            suffix = PurePosixPath(member.name).suffix.lower() or "<none>"
            suffix_counts[suffix] += 1
            lower_name = member.name.lower()
            is_license_candidate = any(marker in lower_name for marker in LICENSE_MARKERS)
            sample_json = suffix == ".json" and len(json_summaries) < args.max_json_samples
            sample_text = suffix in {".txt", ".csv", ".xml", ".ini"} and len(text_samples) < 100
            if suffix in TEXT_SUFFIXES or suffix in {".xml", ".ini"}:
                if len(metadata_members) < 2000:
                    metadata_members.append({"name": member.name, "bytes": member.size})
            if sample_json and member.size > args.max_text_bytes:
                json_summaries[member.name] = {
                    "status": "skipped_size",
                    "bytes": member.size,
                }
            if not (is_license_candidate or sample_json or sample_text):
                continue
            if member.size > args.max_text_bytes:
                if is_license_candidate:
                    license_texts[member.name] = f"<skipped: {member.size} bytes>"
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            raw = extracted.read(args.max_text_bytes + 1)
            if len(raw) > args.max_text_bytes:
                continue
            text = raw.decode("utf-8", errors="replace")
            if is_license_candidate:
                license_texts[member.name] = text[:20000]
            if sample_text:
                text_samples[member.name] = text[:4000]
            if sample_json:
                try:
                    value = json.loads(text)
                except json.JSONDecodeError as error:
                    json_summaries[member.name] = {"status": "invalid_json", "error": str(error)}
                else:
                    json_summaries[member.name] = {"status": "parsed", **summarize_json(value)}
                    collect_attribute_values(value, attribute_values)

    combined_license = "\n".join(license_texts.values()).lower()
    restrictions = [marker for marker in RESTRICTIVE_MARKERS if marker in combined_license]
    archive_specific_license_found = any(
        "license" in name.lower() or "licence" in name.lower() or "copying" in name.lower()
        for name in license_texts
    )
    report = {
        "schema_version": "vehicle-rear-archive-audit-v1",
        "status": "pass_quarantine_audit" if not unsafe_members else "fail_unsafe_member_paths",
        "archive": str(args.archive.resolve()),
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": archive_hash,
        "members": members,
        "regular_files": regular_files,
        "total_uncompressed_bytes": total_uncompressed_bytes,
        "top_level_counts": dict(top_counts.most_common()),
        "suffix_counts": dict(suffix_counts.most_common()),
        "unsafe_members": unsafe_members[:100],
        "license_candidates": sorted(license_texts),
        "metadata_members": metadata_members,
        "archive_specific_license_found": archive_specific_license_found,
        "restrictive_markers_found": restrictions,
        "sample_json_summaries": json_summaries,
        "sample_text": text_samples,
        "sample_attribute_values": dict(attribute_values.most_common(500)),
        "policy": {
            "archive_extracted": False,
            "training_allowed": False,
            "reason": "quarantine audit only; repository-license applicability, dataset terms, labels, privacy and deduplication must be verified before import",
            "frozen_video_used": False,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "archive_sha256", "members", "regular_files",
        "total_uncompressed_bytes", "suffix_counts", "license_candidates",
        "archive_specific_license_found", "restrictive_markers_found", "policy"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
