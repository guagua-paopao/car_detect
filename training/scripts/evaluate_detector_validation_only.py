#!/usr/bin/env python3
"""Evaluate a detector only on the validation key and report present classes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    result = YOLO(str(args.model.resolve())).val(
        data=str(args.data.resolve()), split="val", imgsz=args.imgsz, batch=args.batch,
        device=args.device, rect=False, plots=True, project=str(args.project.resolve()), name=args.name,
    )
    names = {int(key): str(value) for key, value in dict(result.names).items()}
    present_indexes = [int(value) for value in result.box.ap_class_index.tolist()]
    maps = result.box.maps.tolist()
    per_class = {names[index]: float(maps[index]) for index in present_indexes}
    absent = [names[index] for index in sorted(names) if index not in set(present_indexes)]
    payload = {
        "schema_version": "1.0",
        "evaluated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "split": "validation",
        "imgsz": args.imgsz,
        "metrics": {
            "map50": float(result.box.map50), "map50_95": float(result.box.map),
            "precision_mean": float(result.box.mp), "recall_mean": float(result.box.mr),
            "map50_95_per_present_class": per_class, "absent_classes": absent,
        },
        "safety": {"test_accessed": False, "frozen_video_used": False},
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii", newline="\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
