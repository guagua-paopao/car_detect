#!/usr/bin/env python3
"""Rewrite sealed Stage175 local crop paths to an audited cloud-relative manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PureWindowsPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rewrite_path(value: str, expected_parent: str) -> str:
    windows = PureWindowsPath(value)
    if windows.parent.name.lower() != expected_parent.lower():
        raise ValueError(f"unexpected Stage175 path parent: {value}")
    if windows.name in {"", ".", ".."} or "/" in windows.name or "\\" in windows.name:
        raise ValueError(f"invalid crop basename: {value}")
    return f"{expected_parent}/{windows.name}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--crop-directory", default="crops")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite Stage175 cloud-path evidence")
    input_sha = sha256_file(args.input)
    if input_sha.lower() != args.expected_input_sha256.lower():
        raise RuntimeError("input manifest SHA256 mismatch")
    images_root = args.images_root.resolve()
    if images_root.is_symlink() or not images_root.is_dir():
        raise RuntimeError("images root missing or symbolic")
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if "image_path" not in fields or not rows:
        raise RuntimeError("invalid or empty manifest")
    seen: set[str] = set()
    missing: list[str] = []
    for row in rows:
        rewritten = rewrite_path(row["image_path"], args.crop_directory)
        if rewritten in seen:
            raise RuntimeError(f"duplicate rewritten path: {rewritten}")
        seen.add(rewritten)
        candidate = (images_root / rewritten).resolve()
        try:
            candidate.relative_to(images_root)
        except ValueError as error:
            raise RuntimeError("rewritten crop escapes images root") from error
        if candidate.is_symlink() or not candidate.is_file():
            missing.append(str(candidate))
        row["image_path"] = rewritten
    if missing:
        raise FileNotFoundError(f"missing cloud crops: {missing[:10]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "stage175-cloud-relative-path-rewrite-v1",
        "status": "pass",
        "input": str(args.input.resolve()), "input_sha256": input_sha,
        "output": str(args.output.resolve()), "output_sha256": sha256_file(args.output),
        "images_root": str(images_root), "rows": len(rows), "unique_paths": len(seen),
        "missing_files": 0,
        "policy": {
            "only_image_path_changed": True, "body_and_color_labels_unchanged": True,
            "test_accessed": False, "frozen_video_used": False,
            "production_model_modified": False, "training_started": False,
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path in (args.output, args.report):
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{sha256_file(path)}  {path.name}\n", encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
