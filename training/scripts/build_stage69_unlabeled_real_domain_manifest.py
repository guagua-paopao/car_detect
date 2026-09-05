#!/usr/bin/env python3
"""Build a train-only, explicitly unknown real-domain consistency manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "frozen_video")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def build_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], Counter[str]]:
    counters: Counter[str] = Counter()
    output: list[dict[str, str]] = []
    for source in rows:
        if source.get("split") != "train":
            counters["excluded_non_train"] += 1
            continue
        if source.get("review_status") != "approved":
            counters["excluded_not_approved"] += 1
            continue
        joined = " ".join(str(value).lower() for value in source.values())
        if any(marker in joined for marker in FROZEN_MARKERS):
            raise RuntimeError("frozen-video marker found in real-domain train row")
        row = dict(source)
        row["body_type"] = "unknown"
        row["color"] = "unknown"
        row["body_type_supervised"] = "false"
        row["color_supervised"] = "false"
        row["pseudo_label"] = "false"
        row["pseudo_label_confidence"] = ""
        row["coarse_body_family"] = ""
        row["stage69_unlabeled_policy"] = "train_only_explicit_unknown_no_class_target"
        output.append(row)
        counters["accepted"] += 1
        counters[f"source:{row.get('source_dataset', 'unknown')}"] += 1
        counters["night"] += int(truthy(row.get("night")))
        counters["small"] += int(truthy(row.get("small_target")) or row.get("vehicle_size") == "small")
        counters["occluded_or_truncated"] += int(truthy(row.get("occluded")) or truthy(row.get("truncated")))
    return output, counters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite Stage69 evidence: {path}")
    if args.output.parent.resolve() != args.input.parent.resolve():
        raise RuntimeError("output must share the input manifest root so relative image paths remain valid")
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    output, counters = build_rows(rows)
    if not output:
        raise RuntimeError("no eligible real-domain train rows")
    fields += ["stage69_unlabeled_policy"] if "stage69_unlabeled_policy" not in fields else []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(output)
    invalid = sum(
        row.get("split") != "train"
        or row.get("body_type") != "unknown"
        or row.get("color") != "unknown"
        or row.get("body_type_supervised") != "false"
        or row.get("color_supervised") != "false"
        for row in output
    )
    report = {
        "schema_version": "attribute-stage69-unlabeled-real-domain-manifest-v1",
        "status": "pass" if invalid == 0 else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "input_rows": len(rows),
        "output_rows": len(output),
        "counters": dict(sorted(counters.items())),
        "invalid_rows": invalid,
        "source_groups": len({row.get("source_group") or row.get("track_group") or row.get("video_id") for row in output}),
        "policy": {
            "class_predictions_used_as_targets": False,
            "only_augmentation_consistency_allowed": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False
        }
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output_rows": len(output), "output_sha256": report["output_sha256"]}))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
