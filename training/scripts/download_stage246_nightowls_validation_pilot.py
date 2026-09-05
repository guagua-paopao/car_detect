#!/usr/bin/env python3
"""Download a deterministic NightOwls validation-only pilot by HTTP ranges."""

from __future__ import annotations

import argparse
import binascii
import csv
import hashlib
import json
import os
import shutil
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from remotezip import RemoteZip


OFFICIAL_URL = "https://thor.robots.ox.ac.uk/nightowls/python/nightowls_validation.zip"
ARCHIVE_PREFIX = "nightowls_validation"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_crc32(path: Path) -> int:
    value = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value = binascii.crc32(chunk, value)
    return value & 0xFFFFFFFF


def select_blocks(rows: list[dict], blocks: int, block_size: int) -> list[tuple[int, dict]]:
    rows = sorted(rows, key=lambda item: (int(item.get("timestamp") or 0), int(item["id"])))
    requested = blocks * block_size
    if len(rows) < requested:
        return []
    slack = len(rows) - requested
    starts = [round(index * slack / max(blocks - 1, 1)) + index * block_size for index in range(blocks)]
    selected: list[tuple[int, dict]] = []
    for block_index, start in enumerate(starts):
        for row in rows[start : start + block_size]:
            selected.append((block_index, row))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=OFFICIAL_URL)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--license-html", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--max-compressed-gib", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if args.url != OFFICIAL_URL:
        raise ValueError("only the recorded official NightOwls validation URL is allowed")
    if "test" in args.url.lower() or "training" in args.url.lower():
        raise ValueError("Stage246 is validation-only")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_root}")
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    args.output_root.mkdir(parents=True)
    image_root = args.output_root / "images"
    image_root.mkdir()

    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in annotations["images"]:
        recording = row.get("recordings_id")
        if recording is None or row.get("daytime") != "night":
            continue
        groups[int(recording)].append(row)

    with RemoteZip(args.url) as archive:
        info_by_name = {entry.filename: entry for entry in archive.infolist() if not entry.is_dir()}
        planned: list[dict] = []
        for recording in sorted(groups):
            selected = select_blocks(groups[recording], args.blocks, args.block_size)
            if not selected:
                continue
            for block_index, row in selected:
                member = f"{ARCHIVE_PREFIX}/{row['file_name']}"
                info = info_by_name.get(member)
                if info is None:
                    raise FileNotFoundError(f"archive member missing: {member}")
                relative = Path(f"recording_{recording:02d}") / f"block_{block_index:02d}" / row["file_name"]
                planned.append(
                    {
                        "official_split": "validation",
                        "recording_id": recording,
                        "block_index": block_index,
                        "image_id": int(row["id"]),
                        "timestamp": int(row["timestamp"]),
                        "daytime": row["daytime"],
                        "member": member,
                        "relative_path": relative.as_posix(),
                        "compressed_bytes": int(info.compress_size),
                        "uncompressed_bytes": int(info.file_size),
                        "archive_crc32": f"{info.CRC:08x}",
                    }
                )
        expected = len([rows for rows in groups.values() if len(rows) >= args.blocks * args.block_size]) * args.blocks * args.block_size
        if len(planned) != expected or not planned:
            raise ValueError(f"selection mismatch: planned={len(planned)} expected={expected}")
        compressed_total = sum(item["compressed_bytes"] for item in planned)
        if compressed_total > args.max_compressed_gib * 1024**3:
            raise ValueError(f"selection exceeds compressed budget: {compressed_total}")
        free_bytes = shutil.disk_usage(args.output_root).free
        if free_bytes < compressed_total + 10 * 1024**3:
            raise OSError(f"insufficient free disk: free={free_bytes}, planned={compressed_total}")
        plan = {
            "schema_version": "stage246-nightowls-validation-pilot-plan-v1",
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "source_url": args.url,
            "official_split": "validation",
            "recording_groups": len({item["recording_id"] for item in planned}),
            "blocks_per_recording": args.blocks,
            "frames_per_block": args.block_size,
            "selected_frames": len(planned),
            "compressed_bytes": compressed_total,
            "uncompressed_bytes": sum(item["uncompressed_bytes"] for item in planned),
            "free_bytes_before": free_bytes,
            "selection": planned,
        }
        plan_path = args.output_root / "stage246-download-plan.json"
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    started = time.time()
    counts = {"processed": 0, "downloaded": 0}
    lock = threading.Lock()

    def download_shard(items: list[dict]) -> None:
        with RemoteZip(args.url) as worker_archive:
            for item in items:
                target = image_root / item["relative_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                part = target.with_suffix(target.suffix + ".part")
                if part.exists() or target.exists():
                    raise FileExistsError(f"unexpected pre-existing output: {target}")
                with worker_archive.open(item["member"]) as source, part.open("wb") as sink:
                    shutil.copyfileobj(source, sink, length=4 * 1024 * 1024)
                if part.stat().st_size != item["uncompressed_bytes"] or file_crc32(part) != int(item["archive_crc32"], 16):
                    raise IOError(f"download integrity failure: {item['member']}")
                os.replace(part, target)
                with lock:
                    counts["processed"] += 1
                    counts["downloaded"] += 1
                    if counts["processed"] % 100 == 0 or counts["processed"] == len(planned):
                        print(
                            json.dumps(
                                {
                                    "progress": counts["processed"],
                                    "total": len(planned),
                                    "workers": args.workers,
                                    "elapsed_seconds": round(time.time() - started, 1),
                                }
                            ),
                            flush=True,
                        )

    shards = [planned[index::args.workers] for index in range(args.workers)]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_shard, shard) for shard in shards]
        for future in futures:
            future.result()

    verified: list[dict] = []
    for item in planned:
        path = image_root / item["relative_path"]
        with Image.open(path) as opened:
            opened.verify()
            image_format = opened.format
            image_size = opened.size
        if image_format != "PNG" or image_size != (1024, 640):
            raise ValueError(f"unexpected image contract: {path} {image_format} {image_size}")
        output = dict(item)
        output["local_path"] = str(path.resolve())
        output["sha256"] = file_sha256(path)
        output["group_key"] = f"NightOwls|validation|recording_{item['recording_id']:02d}|block_{item['block_index']:02d}"
        verified.append(output)

    manifest_path = args.output_root / "stage246-nightowls-validation-pilot-manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(verified[0]))
        writer.writeheader()
        writer.writerows(verified)
    report = {
        "schema_version": "stage246-nightowls-validation-pilot-v1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "complete_validation_only_research_only_pixels_unlabeled",
        "source_url": args.url,
        "official_split": "validation",
        "selected_frames": len(verified),
        "recording_groups": len({item["recording_id"] for item in verified}),
        "block_groups": len({item["group_key"] for item in verified}),
        "daytime_counts": dict(Counter(item["daytime"] for item in verified)),
        "stored_bytes": sum((image_root / item["relative_path"]).stat().st_size for item in verified),
        "decode_errors": 0,
        "exact_sha256_duplicates": len(verified) - len({item["sha256"] for item in verified}),
        "license_policy": "non-commercial research-only; holdout non-deployable",
        "inputs": {
            "annotations": str(args.annotations.resolve()),
            "annotations_sha256": file_sha256(args.annotations),
            "license_html": str(args.license_html.resolve()),
            "license_html_sha256": file_sha256(args.license_html),
            "download_plan": str(plan_path.resolve()),
            "download_plan_sha256": file_sha256(plan_path),
        },
        "outputs": {"manifest": str(manifest_path.resolve()), "manifest_sha256": file_sha256(manifest_path)},
        "policy": {
            "training_or_threshold_selection_allowed": False,
            "color_labels_generated": False,
            "test_pixels_opened": 0,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    report_path = args.output_root / "stage246-nightowls-validation-pilot-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sums = args.output_root / "SHA256SUMS"
    with sums.open("w", encoding="utf-8") as handle:
        for path in (plan_path, manifest_path, report_path, args.annotations, args.license_html):
            handle.write(f"{file_sha256(path)}  {path}\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
