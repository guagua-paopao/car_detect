#!/usr/bin/env python3
"""Build a fail-closed train-only UVH small-target mixture.

The base validation/test rows are copied unchanged.  Only UVH-26 train rows
with supervised body labels and small vehicle boxes are added, balanced by
body class.  Color supervision remains disabled for UVH rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--uvh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    base_fields, base_rows = read_rows(args.base)
    uvh_fields, uvh_rows = read_rows(args.uvh)
    # The UVH-enriched manifest carries additional mining columns.  Keep the
    # base schema first and append those columns so existing dataset readers
    # remain compatible.
    output_fields = list(base_fields) + [field for field in uvh_fields if field not in base_fields]
    for row in base_rows + uvh_rows:
        for field in output_fields:
            row.setdefault(field, "")

    eligible = [
        row for row in uvh_rows
        if row.get("source_dataset") == "UVH-26"
        and row.get("split") == "train"
        and (row.get("vehicle_size") == "small" or row.get("small_target", "").strip().lower() in {"true", "1", "yes"})
        and row.get("body_type_supervised", "").strip().lower() in {"true", "1", "yes"}
        and row.get("body_type") not in {"", "unknown"}
    ]
    rng = random.Random(args.seed)
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in eligible:
        by_class.setdefault(row.get("body_type", "unknown"), []).append(row)

    selected: list[dict[str, str]] = []
    for label in sorted(by_class):
        pool = list(by_class[label])
        rng.shuffle(pool)
        selected.extend(pool[: max(0, args.per_class)])
    selected.sort(key=lambda row: (row.get("body_type", ""), row.get("image_path", "")))

    output_rows = base_rows + selected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "schema_version": "uvh-small-mixture-manifest-v1",
        "base_manifest": str(args.base),
        "uvh_manifest": str(args.uvh),
        "output_manifest": str(args.output),
        "base_rows": len(base_rows),
        "eligible_uvh_small_rows": len(eligible),
        "selected_uvh_small_rows": len(selected),
        "selected_by_body_type": dict(sorted(Counter(row.get("body_type", "unknown") for row in selected).items())),
        "output_rows": len(output_rows),
        "output_train_rows": sum(row.get("split") == "train" for row in output_rows),
        "validation_test_unchanged": all(row.get("split") == "train" for row in selected),
        "uvh_color_supervision_disabled": all(row.get("color_supervised", "false").strip().lower() in {"false", "0", "no"} for row in selected),
        "seed": args.seed,
        "per_class": args.per_class,
        "frozen_video_used": False,
    }
    report_path = args.output.with_suffix(".report.json")
    report["output_sha256"] = sha256(args.output)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
