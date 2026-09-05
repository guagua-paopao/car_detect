#!/usr/bin/env python3
"""Create a hash-verified VCAS training-result archive without raw datasets."""

from __future__ import annotations

import argparse
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(TRAINING_ROOT))

from src.common import sha256_file, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--manifests-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def files_under(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def main() -> int:
    args = parse_args()
    roots = {
        "runs": args.runs_root.resolve(),
        "artifacts": args.artifacts_root.resolve(),
        "manifests": args.manifests_root.resolve(),
    }
    entries: list[dict[str, object]] = []
    for label, root in roots.items():
        for path in files_under(root):
            entries.append(
                {
                    "path": f"{label}/{path.relative_to(root).as_posix()}",
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not entries:
        raise RuntimeError("no training artifacts were found")

    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "excludes_raw_datasets": True,
        "file_count": len(entries),
        "files": entries,
    }
    manifest_path = roots["manifests"] / "SHA256SUMS.training-results.json"
    write_json(manifest_path, manifest)
    text_path = roots["manifests"] / "SHA256SUMS.training-results.txt"
    with text_path.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(f"{entry['sha256']}  {entry['path']}\n")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for label, root in roots.items():
            archive.add(root, arcname=label, recursive=True)
    archive_summary = {
        "archive": str(output),
        "size": output.stat().st_size,
        "sha256": sha256_file(output),
        "manifest": str(manifest_path),
    }
    write_json(output.with_suffix(output.suffix + ".json"), archive_summary)
    print(json.dumps(archive_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
