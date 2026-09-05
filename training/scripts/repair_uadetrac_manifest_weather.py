#!/usr/bin/env python3
"""Normalize UA-DETRAC's historical ``sence_weather`` typo without changing labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weather_index(roots: list[Path]) -> dict[str, str]:
    result = {}
    for root in roots:
        for path in sorted(root.rglob("*.xml")):
            xml_root = ET.parse(path).getroot()
            sequence = str(xml_root.attrib.get("name", path.stem)).strip().upper()
            attributes = xml_root.find("sequence_attribute")
            if attributes is None:
                result[sequence] = "unknown"
                continue
            result[sequence] = str(
                attributes.attrib.get("weather", attributes.attrib.get("sence_weather", "unknown"))
            ).strip().lower()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--xml-root", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite an existing repaired manifest or report")
    index = weather_index(args.xml_root)
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    changed = 0
    labels_changed = 0
    original_labels = [(row.get("body_type"), row.get("color")) for row in rows]
    for row in rows:
        sequence = str(row.get("source_group") or row.get("video_id", "").split(":")[-1]).upper()
        weather = index.get(sequence, "unknown")
        old_weather = row.get("weather", "unknown")
        if weather != old_weather:
            changed += 1
        row["weather"] = weather
        row["night"] = str(weather == "night").lower()
        if weather == "night":
            row["lighting"] = "night"
        tags = {tag for tag in str(row.get("hard_mining_tags", "")).split(";") if tag}
        if weather == "night":
            tags.add("night")
        else:
            tags.discard("night")
        row["hard_mining_tags"] = ";".join(sorted(tags))
        row["hard_score"] = min(8, 2 * len(tags))
        method = str(row.get("review_method", ""))
        if "sence_weather_normalized" not in method:
            row["review_method"] = method + "+sence_weather_normalized"
    labels_changed = sum(
        original != (row.get("body_type"), row.get("color"))
        for original, row in zip(original_labels, rows)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "uadetrac-weather-repair-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if rows and labels_changed == 0 and "unknown" not in set(index.values()) else "fail",
        "frozen_video_used": False,
        "source_manifest": str(args.input),
        "source_manifest_sha256": sha256(args.input),
        "output_manifest": str(args.output),
        "output_manifest_sha256": sha256(args.output),
        "rows": len(rows),
        "weather_rows_changed": changed,
        "attribute_labels_changed": labels_changed,
        "sequence_weather_counts": dict(sorted(Counter(index.values()).items())),
        "row_weather_counts": dict(sorted(Counter(row["weather"] for row in rows).items())),
        "color_ground_truth_available": any(row.get("track_color_truth") not in {"", "unknown"} for row in rows),
        "training_use_authorized": False,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
