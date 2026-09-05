#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import PurePosixPath

from remotezip import RemoteZip


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with RemoteZip(args.url) as archive:
        infos = [entry for entry in archive.infolist() if not entry.is_dir()]
    suffixes = Counter(PurePosixPath(item.filename).suffix.lower() for item in infos)
    top = Counter()
    second = Counter()
    compressed_by_second = defaultdict(int)
    uncompressed_by_second = defaultdict(int)
    for item in infos:
        parts = PurePosixPath(item.filename).parts
        if parts:
            top[parts[0]] += 1
        key = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
        second[key] += 1
        compressed_by_second[key] += item.compress_size
        uncompressed_by_second[key] += item.file_size
    doc = {
        "stage": "stage232-nightowls-remotezip-index-r1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "url": args.url,
        "entry_count": len(infos),
        "total_compressed_bytes": sum(item.compress_size for item in infos),
        "total_uncompressed_bytes": sum(item.file_size for item in infos),
        "suffix_counts": dict(sorted(suffixes.items())),
        "top_level_counts": dict(sorted(top.items())),
        "two_level_groups": [
            {
                "key": key,
                "entries": second[key],
                "compressed_bytes": compressed_by_second[key],
                "uncompressed_bytes": uncompressed_by_second[key],
            }
            for key in sorted(second)
        ],
        "first_entries": [
            {
                "filename": item.filename,
                "compressed_bytes": item.compress_size,
                "uncompressed_bytes": item.file_size,
            }
            for item in infos[:40]
        ],
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({key: doc[key] for key in ("entry_count", "total_compressed_bytes", "suffix_counts", "top_level_counts")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
