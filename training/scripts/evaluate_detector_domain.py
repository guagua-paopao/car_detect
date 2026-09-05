#!/usr/bin/env python3
"""Evaluate a detector on a named held-out domain and persist JSON metrics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--rect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use rectangular batches. Disable for static square deployment parity.",
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(str(args.model.resolve()))
    result = model.val(
        data=str(args.data.resolve()),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        rect=args.rect,
        plots=True,
        project=str(args.project.resolve()),
        name=args.name,
    )
    names = result.names
    per_class = {
        str(names[index]): float(value)
        for index, value in enumerate(result.box.maps.tolist())
    }
    payload = {
        "schema_version": "1.0",
        "evaluated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "rect": args.rect,
        "metrics": {
            "map50": float(result.box.map50),
            "map50_95": float(result.box.map),
            "precision_mean": float(result.box.mp),
            "recall_mean": float(result.box.mr),
            "map50_95_per_class": per_class,
        },
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
