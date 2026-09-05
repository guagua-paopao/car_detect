#!/usr/bin/env python3
"""Create validation-only taxonomy-v2 views without inventing fine labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


FROZEN_MARKERS = ("vcas_rtsp_demo_60s", "frozen_video")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def supervised(row: dict[str, str], head: str) -> bool:
    return str(row.get(f"{head}_supervised", "true")).strip().lower() not in {"false", "0", "no"}


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("source must be NAME=PATH")
    return name.strip(), Path(raw_path)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite Stage64 validation views")
    if len({name for name, _ in args.source}) != len(args.source):
        raise RuntimeError("validation source names must be unique")

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    body_labels = set(map(str, labels["body_types"]))
    color_labels = set(map(str, labels["colors"]))
    if not {"truck", "light_truck", "heavy_truck", "unknown"}.issubset(body_labels):
        raise RuntimeError("labels are not taxonomy-v2 truck hierarchy labels")
    if not {"gray", "silver", "unknown"}.issubset(color_labels):
        raise RuntimeError("labels are not taxonomy-v2 fine-color labels")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    reports = []
    for name, source in args.source:
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            source_rows = [row for row in reader if row.get("split") == "validation"]
        if not source_rows:
            raise RuntimeError(f"{name} has no validation rows")
        for required in ("image_path", "body_type", "color", "split"):
            if required not in fieldnames:
                raise RuntimeError(f"{name} is missing {required}")
        extra_fields = [
            "taxonomy_v2_original_body_type", "taxonomy_v2_original_color",
            "taxonomy_v2_validation_view", "taxonomy_v2_source_manifest",
        ]
        output_fields = fieldnames + [field for field in extra_fields if field not in fieldnames]
        rows = []
        counts = Counter()
        missing_images = []
        for row in source_rows:
            marker_text = " ".join(str(value).lower() for value in row.values())
            if any(marker in marker_text for marker in FROZEN_MARKERS):
                raise RuntimeError(f"{name} contains a frozen-video marker")
            image = Path(row["image_path"])
            if not image.is_absolute():
                image = (source.parent / image).resolve()
            if not image.is_file():
                missing_images.append(str(image))
                continue
            output = dict(row)
            original_body = str(row.get("body_type", "unknown") or "unknown")
            original_color = str(row.get("color", "unknown") or "unknown")
            output["image_path"] = str(image)
            output["taxonomy_v2_original_body_type"] = original_body
            output["taxonomy_v2_original_color"] = original_color
            output["taxonomy_v2_validation_view"] = name
            output["taxonomy_v2_source_manifest"] = str(source.resolve())
            if original_body not in body_labels or original_body == "unknown" or not supervised(row, "body_type"):
                output["body_type"] = "unknown"
                output["body_type_supervised"] = "false"
                counts["body_demoted_to_unknown"] += 1
            else:
                output["body_type_supervised"] = "true"
                counts["body_exact_supervised"] += 1
            if original_color not in color_labels or original_color == "unknown" or not supervised(row, "color"):
                output["color"] = "unknown"
                output["color_supervised"] = "false"
                counts["color_demoted_to_unknown"] += 1
            else:
                output["color_supervised"] = "true"
                counts["color_exact_supervised"] += 1
            if "track_body_truth" in output_fields:
                truth = str(row.get("track_body_truth", "unknown") or "unknown")
                output["track_body_truth"] = truth if truth in body_labels else "unknown"
            if "track_color_truth" in output_fields:
                truth = str(row.get("track_color_truth", "unknown") or "unknown")
                output["track_color_truth"] = truth if truth in color_labels else "unknown"
            rows.append(output)
        if missing_images:
            raise RuntimeError(f"{name} has {len(missing_images)} missing images; first={missing_images[0]}")
        output_path = args.output_dir / f"{name}.validation-taxonomy-v2.csv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        reports.append({
            "name": name,
            "source": str(source.resolve()),
            "source_sha256": sha256(source),
            "output": str(output_path.resolve()),
            "output_sha256": sha256(output_path),
            "rows": len(rows),
            "counts": dict(sorted(counts.items())),
            "missing_images": 0,
        })

    report = {
        "schema_version": "stage64-taxonomy-v2-validation-views-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "labels": str(args.labels.resolve()),
        "labels_sha256": sha256(args.labels),
        "sources": reports,
        "policy": {
            "validation_only": True,
            "merged_or_unsupported_truth_demoted_not_guessed": True,
            "exact_gray_silver_truth_preserved_when_present": True,
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, report)
    print(json.dumps({"status": "pass", "sources": len(reports), "report": str(args.report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
