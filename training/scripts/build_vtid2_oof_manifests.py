#!/usr/bin/env python3
"""Build leakage-safe nested holdout manifests for VTID2 label auditing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-prefix", default="attribute_manifest.vtid2-oof-v1")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if args.folds < 3:
        raise ValueError("at least three folds are required for train/validation/audit separation")
    if args.report.exists():
        raise FileExistsError(args.report)
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    with args.source_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [
            dict(row) for row in reader
            if row.get("split") == "train"
            and truthy(row.get("body_type_supervised"))
            and row.get("body_type") not in {"", "unknown"}
        ]
    if "oof_fold" not in fields:
        fields.append("oof_fold")

    by_class: dict[str, list[dict[str, str]]] = {}
    for label in sorted({row["body_type"] for row in rows}):
        selected = [row for row in rows if row["body_type"] == label]
        selected.sort(key=lambda row: hashlib.sha256(row["image_path"].encode("utf-8")).hexdigest())
        by_class[label] = selected
        for index, row in enumerate(selected):
            row["oof_fold"] = str(index % args.folds)
    if any(len(values) < args.folds * 5 for values in by_class.values()):
        raise RuntimeError("a class is too small for stable nested holdout folds")

    outputs = []
    for audit_fold in range(args.folds):
        validation_fold = (audit_fold + 1) % args.folds
        output = dataset_root / f"{args.output_prefix}.fold{audit_fold}.csv"
        if output.exists():
            raise FileExistsError(output)
        rendered = []
        for row in rows:
            item = dict(row)
            fold = int(item["oof_fold"])
            item["split"] = "audit" if fold == audit_fold else "validation" if fold == validation_fold else "train"
            rendered.append(item)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rendered)
        distributions = {
            split: dict(sorted(Counter(row["body_type"] for row in rendered if row["split"] == split).items()))
            for split in ("train", "validation", "audit")
        }
        outputs.append({
            "audit_fold": audit_fold,
            "validation_fold": validation_fold,
            "manifest": str(output),
            "manifest_sha256": sha256(output),
            "rows": len(rendered),
            "split_counts": dict(sorted(Counter(row["split"] for row in rendered).items())),
            "class_distributions": distributions,
            "disjoint_fold_roles": len({audit_fold, validation_fold}) == 2,
        })

    report = {
        "schema_version": "vtid2-oof-manifests-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256(args.source_manifest),
        "source_rows": len(rows),
        "folds": args.folds,
        "assignment": "per-class deterministic SHA256 order then round-robin",
        "outputs": outputs,
        "policy": {
            "each_row_audit_exactly_once": all(
                sum(
                    1 for item in outputs
                    if int(next(row["oof_fold"] for row in rows if row["image_path"] == source["image_path"])) == item["audit_fold"]
                ) == 1
                for source in rows
            ),
            "audit_fold_excluded_from_train_and_validation_for_its_teacher": True,
            "nested_validation_disjoint_from_audit": all(item["disjoint_fold_roles"] for item in outputs),
            "test_split_not_created_or_used": True,
            "frozen_video_not_used": True,
            "production_model_unchanged": True,
            "deployment_not_performed": True,
        },
        "decision": "train one teacher per fold using train only, select by nested validation, then review only that fold's audit rows",
    }
    if not all(report["policy"].values()):
        report["status"] = "fail"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
