"""Write reproducible SHA256 evidence for stage-4 candidate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = {}
    for directory in sorted(p for p in args.root.iterdir() if p.is_dir()):
        files = {}
        for name in ("best.pt", "calibrated.onnx", "calibrated.engine", "calibration-v2.json",
                     "hard-test-stratified.json", "onnx-parity.json", "model_card.json"):
            path = directory / name
            if path.exists() and path.is_file() and path.stat().st_size > 0:
                files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if files:
            entries[directory.name] = files
    result = {
        "schema_version": "stage4-sha256-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root),
        "candidates": entries,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
