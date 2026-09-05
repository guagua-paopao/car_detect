#!/usr/bin/env python3
"""Build a train-only complex-scene weighted body manifest without duplication."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
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


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def complex_factors(row: dict[str, str]) -> list[tuple[str, float]]:
    factors: list[tuple[str, float]] = []
    lighting = str(row.get("lighting") or "").strip().lower()
    weather = str(row.get("weather") or "").strip().lower()
    occlusion = str(row.get("occlusion_level") or "").strip().lower()
    size = str(row.get("vehicle_size") or "").strip().lower()
    if size == "small" or truthy(row.get("small_target")):
        factors.append(("small", 1.35))
    if truthy(row.get("night")) or truthy(row.get("low_light")) or lighting in {"night", "low_light"}:
        factors.append(("night_low_light", 1.35))
    if lighting in {"backlight", "strong_backlight"}:
        factors.append(("backlight", 1.20))
    if truthy(row.get("occluded")) or truthy(row.get("truncated")) or occlusion not in {"", "none", "clear", "unknown"}:
        factors.append(("occlusion_truncation", 1.30))
    if truthy(row.get("blur")):
        factors.append(("blur", 1.25))
    if weather not in {"", "clear", "normal", "unknown"}:
        factors.append(("adverse_weather", 1.20))
    return factors


def bounded_multiplier(row: dict[str, str]) -> tuple[float, list[str]]:
    factors = complex_factors(row)
    multiplier = 1.0
    for _, value in factors:
        multiplier *= value
    return min(2.50, multiplier), [name for name, _ in factors]


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


def main() -> int:
    args = parse_args()
    if sha256(args.source).lower() != args.expected_source_sha256.lower():
        raise RuntimeError("source manifest SHA256 mismatch")
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite Stage167 evidence")

    with args.source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("source manifest is empty")
    fields = list(rows[0])
    for field in ("stage167_complex_multiplier", "stage167_complex_reasons"):
        if field not in fields:
            fields.append(field)

    train_rows = 0
    train_complex_rows = 0
    signal_counts: Counter[str] = Counter()
    body_counts: Counter[str] = Counter()
    multiplier_counts: Counter[str] = Counter()
    for row in rows:
        for key in ("image_path", "source_video", "video_id", "source_frame_id", "track_group"):
            if FROZEN_MARKER_RE.search(str(row.get(key, ""))):
                raise RuntimeError(f"frozen/test marker found: {key}={row.get(key)}")
        split = str(row.get("split") or "train").strip().lower()
        supervised = truthy(row.get("body_type_supervised"))
        multiplier, reasons = (1.0, [])
        if split == "train" and supervised:
            train_rows += 1
            multiplier, reasons = bounded_multiplier(row)
            if reasons:
                train_complex_rows += 1
                signal_counts.update(reasons)
                body_counts[str(row.get("body_type") or "unknown").strip().lower()] += 1
        try:
            original = float(row.get("sample_weight") or 1.0)
        except ValueError:
            original = 1.0
        final_weight = max(0.10, min(10.0, original * multiplier))
        row["sample_weight"] = f"{final_weight:.6f}"
        row["stage167_complex_multiplier"] = f"{multiplier:.6f}"
        row["stage167_complex_reasons"] = "+".join(reasons) if reasons else "unchanged"
        if split == "train" and supervised:
            multiplier_counts[f"{multiplier:.6f}"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "schema_version": "stage167-complex-body-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "rows": len(rows),
        "supervised_train_rows": train_rows,
        "complex_reweighted_train_rows": train_complex_rows,
        "complex_reweighted_fraction": train_complex_rows / train_rows if train_rows else 0.0,
        "signal_counts": dict(sorted(signal_counts.items())),
        "body_counts": dict(sorted(body_counts.items())),
        "multiplier_counts": dict(sorted(multiplier_counts.items())),
        "maximum_additional_multiplier": 2.50,
        "final_sample_weight_cap": 10.0,
        "policy": "train-only bounded weighting; no sample duplication, relabeling, split change, test use, threshold change or validation-to-train transfer",
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "rows": len(rows),
        "complex_reweighted_train_rows": train_complex_rows,
        "output_sha256": report["output_sha256"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
