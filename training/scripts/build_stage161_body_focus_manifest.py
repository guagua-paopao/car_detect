#!/usr/bin/env python3
"""Create a train-only body hard-class weighting view without duplicating images."""

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
BODY_MULTIPLIERS = {
    "suv": 1.80,
    "mpv": 2.00,
    "van": 1.50,
    "other": 1.80,
    "pickup": 1.50,
    "truck": 1.30,
    "light_truck": 1.20,
    "heavy_truck": 1.20,
    "sedan": 1.00,
    "bus": 1.00,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--expected-source-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def supervised(row: dict[str, str]) -> bool:
    return str(row.get("body_type_supervised", "")).strip().lower() in {"1", "true", "yes"}


def main() -> int:
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    actual = sha256(args.source)
    if actual.lower() != args.expected_source_sha256.lower():
        raise RuntimeError(f"source manifest SHA256 mismatch: {actual}")
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite Stage161 evidence")

    rows = list(csv.DictReader(args.source.open(newline="", encoding="utf-8")))
    if not rows:
        raise RuntimeError("source manifest is empty")
    fields = list(rows[0])
    if "stage161_body_focus_multiplier" not in fields:
        fields.append("stage161_body_focus_multiplier")
    counts: Counter[str] = Counter()
    weighted_mass: defaultdict[str, float] = defaultdict(float)
    changed = 0
    for row in rows:
        for key in ("image_path", "source_video", "video_id", "source_frame_id", "track_group"):
            value = str(row.get(key, ""))
            if FROZEN_MARKER_RE.search(value):
                raise RuntimeError(f"frozen/test marker found in source row: {key}={row.get(key)}")
        body = str(row.get("body_type", "")).strip()
        factor = BODY_MULTIPLIERS.get(body, 1.0) if supervised(row) and row.get("split", "train") == "train" else 1.0
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
        row["stage161_body_focus_multiplier"] = f"{factor:.6f}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "stage161-body-focus-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "source_sha256": actual,
        "output": str(args.output),
        "rows": len(rows),
        "train_only_reweighted_rows": changed,
        "class_multipliers": BODY_MULTIPLIERS,
        "reweighted_train_counts": dict(sorted(counts.items())),
        "reweighted_train_weighted_mass": dict(sorted(weighted_mass.items())),
        "policy": "weights only; no image duplication, no label fabrication, validation rows unchanged",
        "test_accessed": False,
        "stage148_test_reused": False,
        "stage155_holdout_reused": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "report": str(args.report), "rows": len(rows), "reweighted": changed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
