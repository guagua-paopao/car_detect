#!/usr/bin/env python3
"""Build a train-only Stage163 body hard-class view without duplicating samples."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


FROZEN_MARKER_RE = re.compile(
    r"vcas_rtsp_demo_60s|(?:^|[/_\\-])36[-_]48(?:$|[/_.\\-])|"
    r"(?:^|[/_\\-])stage(?:148|155)(?:$|[/_.\\-])",
    re.IGNORECASE,
)

# Stage162 validation showed that SUV, MPV and other are the dominant static
# errors.  These factors apply only to existing train rows; validation/test
# rows, labels and image membership remain untouched.
BODY_MULTIPLIERS = {
    "suv": 2.40,
    "mpv": 2.80,
    "other": 3.50,
    "pickup": 1.80,
    "van": 1.30,
    "truck": 1.15,
    "light_truck": 1.10,
    "heavy_truck": 1.10,
    "sedan": 1.00,
    "bus": 1.00,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def main() -> int:
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    source_sha = sha256(args.source)
    if source_sha.lower() != args.expected_source_sha256.lower():
        raise RuntimeError(f"source manifest SHA256 mismatch: {source_sha}")
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite Stage163 evidence")

    with args.source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("source manifest is empty")
    fields = list(rows[0])
    for field in ("stage163_body_multiplier", "stage163_focus_reason"):
        if field not in fields:
            fields.append(field)

    changed = 0
    counts: Counter[str] = Counter()
    weighted_mass: defaultdict[str, float] = defaultdict(float)
    split_counts: Counter[str] = Counter()
    for row in rows:
        for key in ("image_path", "source_video", "video_id", "source_frame_id", "track_group"):
            value = str(row.get(key, ""))
            if FROZEN_MARKER_RE.search(value):
                raise RuntimeError(f"frozen/test marker found: {key}={value}")
        split = str(row.get("split") or "train").strip().lower()
        split_counts[split] += 1
        body = str(row.get("body_type") or "").strip().lower()
        supervised = truthy(row.get("body_type_supervised"))
        factor = BODY_MULTIPLIERS.get(body, 1.0) if split == "train" and supervised else 1.0
        try:
            original = float(row.get("sample_weight") or 1.0)
        except ValueError:
            original = 1.0
        new_weight = max(0.10, min(10.0, original * factor))
        if factor != 1.0:
            changed += 1
            counts[body] += 1
            weighted_mass[body] += new_weight
        row["sample_weight"] = f"{new_weight:.6f}"
        row["stage163_body_multiplier"] = f"{factor:.6f}"
        row["stage163_focus_reason"] = "stage162_hard_class" if factor != 1.0 else "unchanged"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "schema_version": "stage163-body-hardclass-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "source_sha256": source_sha,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "train_only_reweighted_rows": changed,
        "class_multipliers": BODY_MULTIPLIERS,
        "reweighted_train_counts": dict(sorted(counts.items())),
        "reweighted_train_weighted_mass": dict(sorted(weighted_mass.items())),
        "policy": "weights only; no duplication, relabeling, split change, test access or validation-to-train transfer",
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "reweighted": changed, "output_sha256": report["output_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
