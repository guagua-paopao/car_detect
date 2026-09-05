#!/usr/bin/env python3
"""Build fail-closed Stage70 body, color, and unlabeled specialist manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
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


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def resolve_image(row: dict[str, str], root: Path, safety_root: Path) -> str:
    raw = Path(row["image_path"])
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    path.relative_to(safety_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def content_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_sha256") or row.get("sha256") or "").strip().lower()


def perceptual_hash(row: dict[str, str]) -> str:
    return str(row.get("crop_dhash64") or row.get("dhash64") or "").strip().lower()


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


class HammingBKTree:
    """Exact radius search for 64-bit perceptual hashes without quadratic bands."""

    def __init__(self) -> None:
        self.root: list | None = None

    def add(self, value: int, row: dict[str, str]) -> None:
        if self.root is None:
            self.root = [value, [row], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(row)
                return
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = [value, [row], {}]
                return
            node = child

    def find(self, value: int, radius: int, removed_ids: set[int] | None = None) -> dict[str, str] | None:
        if self.root is None:
            return None
        removed = removed_ids or set()
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                candidate = next((row for row in node[1] if id(row) not in removed), None)
                if candidate is not None:
                    return candidate
            low, high = distance - radius, distance + radius
            stack.extend(child for edge, child in node[2].items() if low <= edge <= high)
        return None


def deduplicate(rows: list[dict[str, str]], head: str, threshold: int) -> tuple[list[dict[str, str]], dict]:
    # Validation rows are considered first, so a duplicate train row can never
    # leak into model fitting.  Non-pseudo evidence is preferred within a split.
    ordered = sorted(rows, key=lambda row: (
        0 if row["split"] == "validation" else 1,
        1 if truthy(row.get("pseudo_label")) else 0,
        row.get("stage70_origin", ""), row["image_path"],
    ))
    exact: dict[str, list[dict[str, str]]] = defaultdict(list)
    tree = HammingBKTree()
    kept: list[dict[str, str]] = []
    removed_ids: set[int] = set()
    counts = Counter()
    conflicts = Counter()
    examples: list[dict] = []
    for row in ordered:
        candidate = (
            next((item for item in exact.get(content_hash(row), []) if id(item) not in removed_ids), None)
            if content_hash(row) else None
        )
        value = perceptual_hash(row)
        if candidate is None and len(value) == 16:
            candidate = tree.find(int(value, 16), threshold, removed_ids)
        if candidate is None:
            kept.append(row)
            if content_hash(row):
                exact[content_hash(row)].append(row)
            if len(value) == 16:
                tree.add(int(value, 16), row)
            continue
        counts["duplicate_rows"] += 1
        if candidate["split"] != row["split"]:
            counts["cross_split_source_duplicates"] += 1
        candidate_label = candidate.get(head, "unknown")
        row_label = row.get(head, "unknown")
        if head != "unlabeled" and candidate_label != row_label:
            conflicts[f"{candidate_label}->{row_label}"] += 1
            removed_ids.add(id(candidate))
            if len(examples) < 100:
                examples.append({
                    "kept_then_removed": candidate["image_path"], "removed": row["image_path"],
                    "left_label": candidate_label, "right_label": row_label,
                })
            continue
    output = [row for row in kept if id(row) not in removed_ids]
    return output, {
        "input_rows": len(rows), "output_rows": len(output),
        "counts": dict(sorted(counts.items())),
        "label_conflicts_removed_both": dict(sorted(conflicts.items())),
        "conflict_examples": examples,
    }


def post_split_leaks(rows: list[dict[str, str]], threshold: int, limit: int = 100) -> list[dict]:
    validation = [row for row in rows if row["split"] == "validation"]
    train = [row for row in rows if row["split"] == "train"]
    exact = {content_hash(row): row for row in train if content_hash(row)}
    tree = HammingBKTree()
    for row in train:
        value = perceptual_hash(row)
        if len(value) == 16:
            tree.add(int(value, 16), row)
    examples: list[dict] = []
    for row in validation:
        candidate = exact.get(content_hash(row)) if content_hash(row) else None
        value = perceptual_hash(row)
        if candidate is None and len(value) == 16:
            candidate = tree.find(int(value, 16), threshold)
        if candidate is not None:
            examples.append({"train": candidate["image_path"], "validation": row["image_path"]})
            if len(examples) >= limit:
                break
    return examples


def reset_body(row: dict[str, str]) -> None:
    row["body_type"] = "unknown"
    row["body_type_supervised"] = "false"
    row["coarse_body_family"] = ""


def reset_color(row: dict[str, str]) -> None:
    row["color"] = "unknown"
    row["color_supervised"] = "false"


def write_manifest(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage67-manifest", type=Path, required=True)
    parser.add_argument("--stage67-root", type=Path, required=True)
    parser.add_argument("--ua-manifest", type=Path, required=True)
    parser.add_argument("--ua-root", type=Path, required=True)
    color_source = parser.add_mutually_exclusive_group(required=True)
    color_source.add_argument("--color-pseudo-manifest", type=Path)
    color_source.add_argument(
        "--omit-color-pseudo", action="store_true",
        help="Keep every UA color unknown and build the color specialist from audited truth only.",
    )
    parser.add_argument(
        "--color-audit-report", type=Path,
        help="Required pass-status report whose manifest SHA256 authorizes --color-pseudo-manifest.",
    )
    parser.add_argument("--stage69-unlabeled-manifest", type=Path, required=True)
    parser.add_argument("--stage69-unlabeled-root", type=Path, required=True)
    parser.add_argument("--datasets-safety-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--near-duplicate-hamming", type=int, default=3)
    args = parser.parse_args()
    if not 0 <= args.near_duplicate_hamming <= 7:
        raise ValueError("near-duplicate Hamming threshold must be between 0 and 7")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_root}")
    safety = args.datasets_safety_root.resolve()
    if args.color_pseudo_manifest is not None:
        if args.color_audit_report is None or not args.color_audit_report.is_file():
            raise RuntimeError("a color pseudo manifest requires its pass-status audit report")
        color_audit = json.loads(args.color_audit_report.read_text(encoding="utf-8"))
        if (
            color_audit.get("status") != "pass"
            or color_audit.get("output_manifest_sha256", "").lower()
            != sha256(args.color_pseudo_manifest).lower()
        ):
            raise RuntimeError("color pseudo audit is missing, failed, or does not bind the manifest")
        policy = color_audit.get("policy", {})
        if (
            policy.get("train_split_only") is not True
            or policy.get("validation_or_test_used") is not False
            or policy.get("frozen_video_used") is not False
            or policy.get("model_predictions_replace_proposals") is not False
        ):
            raise RuntimeError("color pseudo audit policy is not fail closed")
    elif args.color_audit_report is not None:
        raise RuntimeError("--color-audit-report cannot be used with --omit-color-pseudo")

    sources = {
        "stage67": (args.stage67_manifest, args.stage67_root.resolve()),
        "ua": (args.ua_manifest, args.ua_root.resolve()),
        "stage69_unlabeled": (args.stage69_unlabeled_manifest, args.stage69_unlabeled_root.resolve()),
    }
    if args.color_pseudo_manifest is not None:
        sources["color_pseudo"] = (args.color_pseudo_manifest, args.ua_root.resolve())
    loaded: dict[str, list[dict[str, str]]] = {}
    all_fields: list[str] = []
    for name, (manifest, _) in sources.items():
        fields, rows = load_rows(manifest)
        loaded[name] = rows
        for field in fields:
            if field not in all_fields:
                all_fields.append(field)
    for field in ("stage70_origin", "stage70_source_manifest", "stage70_specialist"):
        if field not in all_fields:
            all_fields.append(field)

    def prepare(name: str, row: dict[str, str], specialist: str) -> dict[str, str]:
        output = dict(row)
        output["image_path"] = resolve_image(output, sources[name][1], safety)
        output["stage70_origin"] = name
        output["stage70_source_manifest"] = str(sources[name][0].resolve())
        output["stage70_specialist"] = specialist
        return output

    body_rows: list[dict[str, str]] = []
    for name in ("stage67", "ua"):
        for source in loaded[name]:
            if source.get("split") not in {"train", "validation"}:
                continue
            exact = truthy(source.get("body_type_supervised")) and source.get("body_type") != "unknown"
            coarse = source.get("coarse_body_family") in {"car", "truck"}
            if not (exact or coarse):
                continue
            row = prepare(name, source, "body")
            reset_color(row)
            body_rows.append(row)

    color_rows: list[dict[str, str]] = []
    for source in loaded["stage67"]:
        if source.get("split") not in {"train", "validation"}:
            continue
        if not truthy(source.get("color_supervised")) or source.get("color") == "unknown":
            continue
        row = prepare("stage67", source, "color")
        reset_body(row)
        color_rows.append(row)
    for source in loaded.get("color_pseudo", []):
        if source.get("split") != "train":
            raise RuntimeError("color pseudo manifest contains a non-train row")
        if not truthy(source.get("color_supervised")) or not truthy(source.get("pseudo_label")):
            raise RuntimeError("color pseudo manifest contains an unapproved row")
        row = prepare("color_pseudo", source, "color")
        reset_body(row)
        color_rows.append(row)

    unlabeled_rows: list[dict[str, str]] = []
    for name in ("stage69_unlabeled", "ua"):
        for source in loaded[name]:
            if source.get("split") != "train":
                continue
            row = prepare(name, source, "unlabeled")
            reset_body(row); reset_color(row)
            row["pseudo_label"] = "false"
            row["pseudo_label_confidence"] = ""
            unlabeled_rows.append(row)

    body_rows, body_dedup = deduplicate(body_rows, "body_type", args.near_duplicate_hamming)
    color_rows, color_dedup = deduplicate(color_rows, "color", args.near_duplicate_hamming)
    # The consistency stream has only train rows; exact duplicate removal is
    # sufficient because no split boundary can be crossed.
    unlabeled_rows, unlabeled_dedup = deduplicate(unlabeled_rows, "unlabeled", 0)
    for rows in (body_rows, color_rows, unlabeled_rows):
        rows.sort(key=lambda row: (row["split"], row.get("source_dataset", ""), row["image_path"]))
    body_leaks = post_split_leaks(body_rows, args.near_duplicate_hamming)
    color_leaks = post_split_leaks(color_rows, args.near_duplicate_hamming)
    group_leaks: dict[str, list[str]] = {}
    for name, rows in (("body", body_rows), ("color", color_rows)):
        train_groups = {row.get("track_group") or row.get("video_id") for row in rows if row["split"] == "train"}
        validation_groups = {row.get("track_group") or row.get("video_id") for row in rows if row["split"] == "validation"}
        group_leaks[name] = sorted((train_groups & validation_groups) - {None, ""})[:100]
    status = "pass"
    if not body_rows or not color_rows or not unlabeled_rows or body_leaks or color_leaks or any(group_leaks.values()):
        status = "fail"
    if any(row.get("split") == "test" for rows in (body_rows, color_rows, unlabeled_rows) for row in rows):
        status = "fail"
    args.output_root.mkdir(parents=True, exist_ok=True)
    body_path = args.output_root / "attribute_manifest.stage70-body.csv"
    color_path = args.output_root / "attribute_manifest.stage70-color.csv"
    unlabeled_path = args.output_root / "attribute_manifest.stage70-unlabeled.csv"
    write_manifest(body_path, all_fields, body_rows)
    write_manifest(color_path, all_fields, color_rows)
    write_manifest(unlabeled_path, all_fields, unlabeled_rows)
    report = {
        "schema_version": "stage70-specialist-manifests-v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "status": status,
        "eligibility": "research-only_non-deployable",
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256(path), "rows": len(loaded[name])}
                   for name, (path, _) in sources.items()},
        "outputs": {
            "body": {"path": str(body_path), "sha256": sha256(body_path), "rows": len(body_rows),
                     "splits": dict(Counter(row["split"] for row in body_rows)),
                     "labels": dict(Counter(row["body_type"] for row in body_rows if truthy(row.get("body_type_supervised")))),
                     "coarse": dict(Counter(row.get("coarse_body_family") for row in body_rows if row.get("coarse_body_family"))),
                     "dedup": body_dedup},
            "color": {"path": str(color_path), "sha256": sha256(color_path), "rows": len(color_rows),
                      "splits": dict(Counter(row["split"] for row in color_rows)),
                      "labels": dict(Counter(row["color"] for row in color_rows)), "dedup": color_dedup},
            "unlabeled": {"path": str(unlabeled_path), "sha256": sha256(unlabeled_path), "rows": len(unlabeled_rows),
                          "splits": dict(Counter(row["split"] for row in unlabeled_rows)), "dedup": unlabeled_dedup},
        },
        "post_filter_cross_split_near_leaks": {"body": body_leaks, "color": color_leaks},
        "post_filter_group_leaks": group_leaks,
        "policy": {
            "test_rows_imported": False, "frozen_video_used": False,
            "body_and_color_specialists_are_separate": True,
            "color_pseudo_mode": (
                "pass_status_train_only_audit" if args.color_pseudo_manifest is not None
                else "omitted_after_failed_audit_all_ua_colors_unknown"
            ),
            "color_pseudo_rows_train_only": args.color_pseudo_manifest is not None,
            "failed_color_pseudo_never_imported": args.color_pseudo_manifest is None,
            "unlabeled_targets_all_unknown": True,
            "threshold_or_temperature_selected": False,
            "production_model_modified": False, "deployment_performed": False,
        },
    }
    report_path = args.output_root / "stage70-specialist-manifests-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
