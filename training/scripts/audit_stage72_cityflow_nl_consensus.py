#!/usr/bin/env python3
"""Audit conservative train-only color/type consensus in CityFlow-NL.

This script audits the Apache-2.0 annotation work only.  It does not download
or authorize the underlying CityFlow images.  A label is emitted only when at
least two primary descriptions each contain one unambiguous requested label,
all usable descriptions agree, and no description contains multiple vehicle
colors/types that could refer to surrounding traffic.

The resulting labels are train-only pseudo-label candidates.  They are never
independent validation truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "36-48s", "36_48s", "frozen_video")
COLOR_PATTERNS = {
    "black": re.compile(r"\bblack\b", re.I),
    "white": re.compile(r"\bwhite\b", re.I),
    "gray": re.compile(r"\bgr[ae]y\b", re.I),
    "silver": re.compile(r"\bsilver\b", re.I),
    "blue": re.compile(r"\bblue\b", re.I),
    "red": re.compile(r"\bred\b", re.I),
    "brown": re.compile(r"\bbrown\b", re.I),
    "green": re.compile(r"\bgreen\b", re.I),
    "yellow": re.compile(r"\byellow\b", re.I),
    # Distinct colors are recognized as conflicts, not folded into a target.
    "non_target_color": re.compile(
        r"\b(?:beige|tan|orange|gold|bronze|purple|pink|magenta|cyan|turquoise)\b",
        re.I,
    ),
}
BODY_PATTERNS = {
    "sedan": re.compile(r"\bsedan\b", re.I),
    "suv": re.compile(r"\b(?:suv|crossover)\b", re.I),
    "mpv": re.compile(r"\b(?:mpv|mini[ -]?van|minivan)\b", re.I),
    "van": re.compile(r"(?<!mini)\bvan\b", re.I),
    "pickup": re.compile(r"\bpick[ -]?up(?:\s+truck)?\b", re.I),
    "truck": re.compile(r"(?<!pick-up )(?<!pickup )\btruck\b", re.I),
    "bus": re.compile(r"\bbus\b", re.I),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_single_label(text: str, patterns: dict[str, re.Pattern[str]]) -> str | None:
    labels = {label for label, pattern in patterns.items() if pattern.search(text)}
    if any(label.startswith("non_target_") for label in labels):
        return None
    return next(iter(labels)) if len(labels) == 1 else None


def consensus(descriptions: list[str], patterns: dict[str, re.Pattern[str]], minimum: int) -> tuple[str, int, str]:
    usable = [label for text in descriptions if (label := exact_single_label(text, patterns))]
    distinct = set(usable)
    if len(distinct) > 1:
        return "unknown", 0, "conflicting_usable_descriptions"
    if len(usable) < minimum:
        return "unknown", len(usable), "insufficient_unambiguous_descriptions"
    return usable[0], len(usable), "accepted_unanimous_usable_descriptions"


def camera_ids(frames: list[str]) -> list[str]:
    found: set[str] = set()
    for frame in frames:
        match = re.search(r"(?:^|[/\\])(c\d{3})(?:[/\\])", frame, re.I)
        if match:
            found.add(match.group(1).lower())
    return sorted(found)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--expected-annotations-sha256", required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--expected-license-sha256", required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--output-labels", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--minimum-agreement", type=int, default=2)
    parser.add_argument("--expected-track-count", type=int, default=2155)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_labels.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite CityFlow-NL audit evidence")
    if not 2 <= args.minimum_agreement <= 3:
        raise ValueError("minimum agreement must be 2 or 3")

    annotation_sha = sha256(args.annotations)
    license_sha = sha256(args.license)
    if annotation_sha.lower() != args.expected_annotations_sha256.lower():
        raise RuntimeError("annotation SHA256 mismatch")
    if license_sha.lower() != args.expected_license_sha256.lower():
        raise RuntimeError("license SHA256 mismatch")
    license_text = args.license.read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise RuntimeError("expected Apache-2.0 license evidence")

    raw = args.annotations.read_text(encoding="utf-8")
    lowered = raw.lower()
    frozen = [marker for marker in FROZEN_MARKERS if marker in lowered]
    if frozen:
        raise RuntimeError(f"frozen asset marker found: {frozen}")
    tracks = json.loads(raw)
    if not isinstance(tracks, dict):
        raise RuntimeError("CityFlow-NL annotations must be a track dictionary")
    if len(tracks) != args.expected_track_count:
        raise RuntimeError(
            f"unexpected training track count: {len(tracks)} != {args.expected_track_count}"
        )

    counters: Counter[str] = Counter()
    labels: list[dict[str, object]] = []
    frame_paths: set[str] = set()
    observations: set[tuple[str, tuple[float, ...]]] = set()
    for track_id, track in sorted(tracks.items()):
        descriptions = track.get("nl")
        frames = track.get("frames")
        boxes = track.get("boxes")
        if not isinstance(descriptions, list) or len(descriptions) != 3:
            raise RuntimeError(f"track {track_id}: expected exactly three primary descriptions")
        if not all(isinstance(item, str) and item.strip() for item in descriptions):
            raise RuntimeError(f"track {track_id}: empty primary description")
        if not isinstance(frames, list) or not frames or len(frames) != len(boxes or []):
            raise RuntimeError(f"track {track_id}: frames/boxes are missing or misaligned")
        current_observations = {
            (str(frame), tuple(float(value) for value in box))
            for frame, box in zip(frames, boxes)
        }
        overlap = observations.intersection(current_observations)
        if overlap:
            raise RuntimeError(
                f"identical frame-and-box observation occurs in multiple training tracks: {next(iter(overlap))}"
            )
        observations.update(current_observations)
        frame_paths.update(str(frame) for frame in frames)

        color, color_votes, color_reason = consensus(
            descriptions, COLOR_PATTERNS, args.minimum_agreement
        )
        body, body_votes, body_reason = consensus(
            descriptions, BODY_PATTERNS, args.minimum_agreement
        )
        counters[f"color:{color}"] += 1
        counters[f"body:{body}"] += 1
        counters[f"color_reason:{color_reason}"] += 1
        counters[f"body_reason:{body_reason}"] += 1
        labels.append(
            {
                "track_id": track_id,
                "split": "train",
                "color": color,
                "color_consensus_descriptions": color_votes,
                "body_type": body,
                "body_consensus_descriptions": body_votes,
                "frame_count": len(frames),
                "camera_ids": camera_ids([str(frame) for frame in frames]),
                "annotation_source": "CityFlow-NL primary descriptions",
                "annotation_license": "Apache-2.0 annotation work; underlying images separately gated",
                "eligibility": "train-only_pseudo-label_candidate_pending_image-package-terms-and-pixel-audit",
            }
        )

    args.output_labels.parent.mkdir(parents=True, exist_ok=True)
    with args.output_labels.open("x", encoding="utf-8", newline="\n") as handle:
        for item in labels:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    accepted_color = sum(item["color"] != "unknown" for item in labels)
    accepted_body = sum(item["body_type"] != "unknown" for item in labels)
    report = {
        "schema_version": "attribute-stage72-cityflow-nl-consensus-audit-v1",
        "status": "pass_annotations_only_images_not_authorized_or_accessed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": "https://github.com/fredfung007/CityFlow-NL",
        "repository_commit": args.repository_commit,
        "annotations": {
            "path": str(args.annotations.resolve()),
            "sha256": annotation_sha,
            "tracks": len(tracks),
            "unique_frame_paths": len(frame_paths),
            "unique_frame_box_observations": len(observations),
        },
        "license": {
            "path": str(args.license.resolve()),
            "sha256": license_sha,
            "scope": "repository annotation and code work; underlying CityFlow images require separate package audit",
        },
        "consensus": {
            "minimum_agreement": args.minimum_agreement,
            "accepted_color_tracks": accepted_color,
            "accepted_body_tracks": accepted_body,
            "color_counts": dict(
                sorted((key.split(":", 1)[1], value) for key, value in counters.items() if key.startswith("color:"))
            ),
            "body_counts": dict(
                sorted((key.split(":", 1)[1], value) for key, value in counters.items() if key.startswith("body:"))
            ),
            "reason_counts": dict(
                sorted((key, value) for key, value in counters.items() if "_reason:" in key)
            ),
        },
        "output_labels": {
            "path": str(args.output_labels.resolve()),
            "sha256": sha256(args.output_labels),
            "rows": len(labels),
        },
        "policy": {
            "primary_descriptions_only": True,
            "multi_color_or_multi_type_description_rejected": True,
            "conflicting_usable_descriptions_rejected": True,
            "images_accessed": False,
            "training_rows_imported": 0,
            "validation_truth_claimed": False,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["consensus"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
