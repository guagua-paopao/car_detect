from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
import zipfile
from pathlib import Path, PurePosixPath


TRAIN_ANNOTATIONS = {
    "M3OT/Annotations/1/rgb/train_cocoformat.json",
    "M3OT/Annotations/2/rgb/train_cocoformat.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or ".." in path.parts or ":" in path.parts[0]:
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    return str(path)


def is_rgb_train_image(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return (
        len(parts) == 7
        and parts[0] == "M3OT"
        and parts[1] in {"1", "2"}
        and parts[2] == "rgb"
        and parts[3] == "train"
        and parts[5] == "img1"
        and parts[6].lower().endswith(".png")
    )


def selected_member(name: str) -> bool:
    return name in TRAIN_ANNOTATIONS or is_rgb_train_image(name)


def extract_rgb_train(
    archive: Path,
    audit_report: Path,
    output_root: Path,
    *,
    expected_audit_sha256: str,
    expected_images: int,
    minimum_free_bytes: int,
    maximum_output_bytes: int,
) -> dict[str, object]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if not audit_report.is_file():
        raise FileNotFoundError(audit_report)
    actual_audit_sha256 = sha256_file(audit_report)
    if actual_audit_sha256.lower() != expected_audit_sha256.lower():
        raise ValueError("audit report SHA256 mismatch")
    audit = json.loads(audit_report.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise ValueError("archive audit did not pass")
    if audit.get("archive_sha256") != sha256_file(archive):
        raise ValueError("archive SHA256 differs from the sealed audit")
    if audit.get("validation_payload_opened") or audit.get("test_payload_opened"):
        raise ValueError("archive audit reports validation/test payload access")
    if output_root.exists():
        raise FileExistsError(output_root)

    with zipfile.ZipFile(archive, "r") as opened:
        selected: list[tuple[zipfile.ZipInfo, str]] = []
        image_count = 0
        total_bytes = 0
        for info in opened.infolist():
            name = normalized_member_name(info.filename)
            if info.is_dir() or not selected_member(name):
                continue
            selected.append((info, name))
            total_bytes += info.file_size
            if is_rgb_train_image(name):
                image_count += 1
        if image_count != expected_images:
            raise ValueError(f"RGB train image count mismatch: {image_count} != {expected_images}")
        if {name for _, name in selected if name in TRAIN_ANNOTATIONS} != TRAIN_ANNOTATIONS:
            raise ValueError("missing RGB train annotations")
        if total_bytes > maximum_output_bytes:
            raise ValueError(f"selected output exceeds guard: {total_bytes} > {maximum_output_bytes}")
        free_bytes = shutil.disk_usage(output_root.parent).free
        if free_bytes - total_bytes < minimum_free_bytes:
            raise OSError(f"free-space guard failed: {free_bytes} - {total_bytes} < {minimum_free_bytes}")

        staging = output_root.parent / f".{output_root.name}.staging-{uuid.uuid4().hex}"
        if staging.exists():
            raise FileExistsError(staging)
        staging.mkdir(parents=False)
        extracted_bytes = 0
        try:
            for info, name in selected:
                relative = PurePosixPath(name).relative_to("M3OT")
                destination = staging.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with opened.open(info, "r") as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
                if destination.stat().st_size != info.file_size:
                    raise OSError(f"extracted size mismatch: {name}")
                extracted_bytes += info.file_size
            os.replace(staging, output_root)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    return {
        "schema_version": "m3ot-rgb-train-extraction-v1",
        "status": "pass",
        "archive": str(archive),
        "archive_sha256": audit["archive_sha256"],
        "audit_report": str(audit_report),
        "audit_report_sha256": actual_audit_sha256,
        "output_root": str(output_root),
        "rgb_train_images": image_count,
        "rgb_train_annotation_files": len(TRAIN_ANNOTATIONS),
        "extracted_files": len(selected),
        "extracted_bytes": extracted_bytes,
        "zip_crc_verified_for_extracted_members": True,
        "validation_payload_opened": False,
        "test_payload_opened": False,
        "ir_payload_opened": False,
        "training_started": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract only sealed M3OT RGB training images and COCO train annotations.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--expected-audit-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=8631)
    parser.add_argument("--minimum-free-bytes", type=int, default=107374182400)
    parser.add_argument("--maximum-output-bytes", type=int, default=6442450944)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = extract_rgb_train(
        args.archive.resolve(),
        args.audit_report.resolve(),
        args.output_root.resolve(),
        expected_audit_sha256=args.expected_audit_sha256,
        expected_images=args.expected_images,
        minimum_free_bytes=args.minimum_free_bytes,
        maximum_output_bytes=args.maximum_output_bytes,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.with_suffix(args.report.suffix + ".sha256").write_text(
        f"{sha256_file(args.report)}  {args.report.name}\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
