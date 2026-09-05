#!/usr/bin/env python3
"""Report attribute manifest split, supervision, source, license, and class distributions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-paths", action="store_true")
    args = parser.parse_args()
    counters = {
        "split": Counter(), "source": Counter(), "license": Counter(),
        "body": Counter(), "color": Counter(), "split_body": Counter(),
        "split_color": Counter(), "source_body": Counter(), "source_color": Counter(),
        "split_source": Counter(), "hard_tags": Counter(),
    }
    rows = body_supervised = color_supervised = 0
    missing_paths: list[str] = []
    missing_path_count = 0
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            if args.check_paths:
                image_path = row.get("image_path", "")
                if not image_path or not (args.manifest.parent / image_path).is_file():
                    missing_path_count += 1
                    if len(missing_paths) < 50:
                        missing_paths.append(image_path)
            split = row.get("split", "unknown") or "unknown"
            source = row.get("source_dataset", "unknown") or "unknown"
            license_name = row.get("source_license", "unknown") or "unknown"
            counters["split"][split] += 1
            counters["source"][source] += 1
            counters["license"][license_name] += 1
            counters["split_source"][f"{split}|{source}"] += 1
            if truthy(row.get("body_type_supervised")) and row.get("body_type") not in {"", "unknown"}:
                body_supervised += 1
                body = row["body_type"]
                counters["body"][body] += 1
                counters["split_body"][f"{split}|{body}"] += 1
                counters["source_body"][f"{source}|{body}"] += 1
            if truthy(row.get("color_supervised")) and row.get("color") not in {"", "unknown"}:
                color_supervised += 1
                color = row["color"]
                counters["color"][color] += 1
                counters["split_color"][f"{split}|{color}"] += 1
                counters["source_color"][f"{source}|{color}"] += 1
            for tag in str(row.get("hard_mining_tags", "")).split(";"):
                if tag:
                    counters["hard_tags"][tag] += 1
    report = {
        "schema_version": "attribute-manifest-distribution-v1",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "rows": rows,
        "body_supervised_rows": body_supervised,
        "color_supervised_rows": color_supervised,
        "frozen_video_used": False,
        "paths_checked": args.check_paths,
        "missing_path_count": missing_path_count,
        "missing_path_sample": missing_paths,
        **{name: dict(sorted(counter.items())) for name, counter in counters.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
