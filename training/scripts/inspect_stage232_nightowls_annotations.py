#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def compact(value, depth: int = 0):
    if depth >= 3:
        return type(value).__name__
    if isinstance(value, dict):
        return {key: compact(item, depth + 1) for key, item in list(value.items())[:20]}
    if isinstance(value, list):
        return [compact(item, depth + 1) for item in value[:3]]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    doc = json.loads(args.annotations.read_text(encoding="utf-8"))
    out = {"root_type": type(doc).__name__}
    if isinstance(doc, dict):
        out["root_keys"] = list(doc)
        out["root_value_types"] = {key: type(value).__name__ for key, value in doc.items()}
        for key, value in doc.items():
            if isinstance(value, list):
                out[f"{key}_length"] = len(value)
                if value:
                    out[f"{key}_first"] = compact(value[0])
            elif isinstance(value, dict):
                out[f"{key}_length"] = len(value)
                out[f"{key}_first_items"] = compact(value)
        images = doc.get("images")
        if isinstance(images, list) and images and isinstance(images[0], dict):
            for field in ("daytime", "recordings_id", "height", "width"):
                counts = Counter(str(row.get(field)) for row in images)
                out[f"images_{field}_counts"] = dict(counts.most_common())
            timestamps = sorted(int(row["timestamp"]) for row in images if row.get("timestamp") is not None)
            if timestamps:
                gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
                out["timestamp"] = {
                    "min": timestamps[0],
                    "max": timestamps[-1],
                    "median_gap": sorted(gaps)[len(gaps) // 2] if gaps else None,
                    "gaps_over_1000": sum(gap > 1000 for gap in gaps),
                    "gaps_over_10000": sum(gap > 10000 for gap in gaps),
                    "gaps_over_100000": sum(gap > 100000 for gap in gaps),
                    "largest_gaps": sorted(gaps, reverse=True)[:20],
                }
    elif isinstance(doc, list):
        out["length"] = len(doc)
        if doc:
            out["first"] = compact(doc[0])
            if isinstance(doc[0], dict):
                out["first_keys"] = list(doc[0])
                for key in doc[0]:
                    out[f"first_field_{key}_types"] = Counter(type(row.get(key)).__name__ for row in doc[:100] if isinstance(row, dict))
    print(json.dumps(out, ensure_ascii=False, indent=2, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
