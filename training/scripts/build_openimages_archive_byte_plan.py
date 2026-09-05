#!/usr/bin/env python3
"""Pin official Open Images S3 archive bytes for an already-audited image plan."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(image_id: str, url_template: str) -> dict:
    url = url_template.format(image_id=image_id)
    response = requests.head(
        url,
        timeout=(20, 60),
        allow_redirects=True,
        headers={"User-Agent": "vcas-openimages-archive-audit/1.0"},
    )
    if response.status_code != 200:
        raise RuntimeError(f"archive HEAD status {response.status_code}")
    try:
        size = int(response.headers.get("Content-Length", "0"))
    except ValueError as error:
        raise RuntimeError("archive Content-Length is invalid") from error
    etag = response.headers.get("ETag", "").strip('"').lower()
    if size <= 0 or not re.fullmatch(r"[0-9a-f]{32}", etag):
        raise RuntimeError(f"archive size/ETag is not immutable: size={size}, etag={etag!r}")
    return {
        "image_id": image_id,
        "archive_url": url,
        "archive_size_bytes": size,
        "archive_md5_hex": etag,
        "archive_md5_base64": base64.b64encode(bytes.fromhex(etag)).decode("ascii"),
        "archive_etag": etag,
        "archive_accept_ranges": response.headers.get("Accept-Ranges", ""),
        "archive_last_modified": response.headers.get("Last-Modified", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument(
        "--url-template",
        default="https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg",
    )
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.output_plan.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite Open Images archive-byte evidence")
    if not args.url_template.startswith(
        "https://open-images-dataset.s3.amazonaws.com/train/"
    ):
        raise ValueError("archive URL must use the official Open Images S3 train prefix")

    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    source_status = source_report.get("status")
    allowed_statuses = {
        "pass_bounded_plan",
        "pass_bounded_plan_pending_photometric_audit",
    }
    if source_status not in allowed_statuses:
        raise RuntimeError(f"source bounded plan did not pass: {source_status}")
    if source_report.get("output_plan_sha256") != sha256(args.source_plan):
        raise RuntimeError("source bounded plan hash mismatch")
    policy = source_report.get("policy", {})
    if policy.get("frozen_video_used") is not False:
        raise RuntimeError("source bounded plan does not prove frozen-video isolation")
    if not (policy.get("train_only") is True or policy.get("train_split_only") is True):
        raise RuntimeError("source bounded plan is not train-only")

    with args.source_plan.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        image_id = row.get("image_id", "")
        if not image_id:
            raise ValueError("source plan contains an empty image_id")
        if "creativecommons.org/licenses/by/2.0" not in row.get("license_url", ""):
            raise ValueError(f"source plan has an invalid image license for {image_id}")
        by_image[image_id].append(row)
    if not by_image:
        raise RuntimeError("source bounded plan is empty")

    workers = max(1, min(args.workers, 24))
    archive = {}
    failures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(probe, image_id, args.url_template): image_id
            for image_id in sorted(by_image)
        }
        for future in as_completed(futures):
            image_id = futures[future]
            try:
                archive[image_id] = future.result()
            except Exception as error:
                failures[image_id] = f"{type(error).__name__}: {error}"

    output_rows = []
    for row in rows:
        evidence = archive.get(row["image_id"])
        if evidence is None:
            continue
        output_rows.append({
            **row,
            "source_original_size_bytes": row.get("original_size_bytes", ""),
            "source_original_md5_base64": row.get("original_md5_base64", ""),
            "original_size_bytes": str(evidence["archive_size_bytes"]),
            "original_md5_base64": evidence["archive_md5_base64"],
            "archive_url": evidence["archive_url"],
            "archive_md5_hex": evidence["archive_md5_hex"],
            "archive_integrity_source": "official_s3_content_length_and_single_part_etag_md5",
            "archive_last_modified": evidence["archive_last_modified"],
        })
    additions = [
        "source_original_size_bytes",
        "source_original_md5_base64",
        "archive_url",
        "archive_md5_hex",
        "archive_integrity_source",
        "archive_last_modified",
    ]
    output_fields = fields + [field for field in additions if field not in fields]
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    with args.output_plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    archive_bytes = sum(item["archive_size_bytes"] for item in archive.values())
    source_bytes = sum(
        int(image_rows[0].get("original_size_bytes") or 0)
        for image_rows in by_image.values()
    )
    gates = {
        "all_source_images_archive_pinned": len(archive) == len(by_image) and not failures,
        "all_output_rows_retained": len(output_rows) == len(rows),
        "official_s3_train_prefix_only": True,
        "per_image_cc_by_2_license_rechecked": True,
        "source_bounded_plan_hash_verified": True,
    }
    status = source_status if all(gates.values()) else "fail_archive_availability"
    report = {
        "schema_version": "openimages-official-archive-byte-plan-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source_status_preserved": source_status,
        "source_plan": str(args.source_plan.resolve()),
        "source_plan_sha256": sha256(args.source_plan),
        "source_report": str(args.source_report.resolve()),
        "source_report_sha256": sha256(args.source_report),
        "official_archive_url_template": args.url_template,
        "workers": workers,
        "source_unique_images": len(by_image),
        "archive_pinned_unique_images": len(archive),
        "source_rows": len(rows),
        "output_rows": len(output_rows),
        "source_original_bytes": source_bytes,
        "official_archive_bytes": archive_bytes,
        "archive_to_original_byte_ratio": archive_bytes / source_bytes if source_bytes else 0.0,
        "probe_failures": len(failures),
        "probe_failure_examples": dict(list(sorted(failures.items()))[:20]),
        "gates": gates,
        "output_plan": str(args.output_plan.resolve()),
        "output_plan_sha256": sha256(args.output_plan),
        "policy": {
            "source_license_metadata_preserved": True,
            "source_original_integrity_metadata_preserved_separately": True,
            "download_integrity_replaced_only_with_official_archive_content_length_and_etag": True,
            "archive_etag_required_to_be_single_part_md5": True,
            "train_only": True,
            "validation_or_test_used": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "decision": "download only if every bounded source image is pinned; otherwise fail closed",
    }
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": status,
        "images": len(archive),
        "source_images": len(by_image),
        "rows": len(output_rows),
        "archive_bytes": archive_bytes,
        "source_bytes": source_bytes,
        "failures": len(failures),
        "gates": gates,
    }, ensure_ascii=False))
    return 0 if status == source_status else 2


if __name__ == "__main__":
    raise SystemExit(main())
