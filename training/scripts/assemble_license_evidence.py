#!/usr/bin/env python3
"""Assemble source-license evidence without widening formal-train eligibility."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "training" / "artifacts" / "attribute-domain-v2"
EVIDENCE = ART / "license-evidence"
MANIFEST = ART / "attribute_manifest.formal-candidate.csv"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_by_source() -> dict[str, dict[str, int]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        source = row["source_dataset"]
        bucket = result.setdefault(source, {"rows": 0, "train_rows": 0, "train_eligible_rows": 0})
        bucket["rows"] += 1
        if row["split"] == "train":
            bucket["train_rows"] += 1
            if row["license_train_eligible"] == "true":
                bucket["train_eligible_rows"] += 1
    return result


def main() -> int:
    sources = rows_by_source()
    cards = {}
    for name in ("attribute-domain-v2-dataset-card.json", "vcor-dataset-card.json", "stanford-dataset-card.json"):
        path = EVIDENCE / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            cards[name] = {
                "path": str(path),
                "sha256": digest(path),
                "dataset_version": data.get("dataset_version"),
                "source": data.get("source", {}),
            }

    report = {
        "schema_version": "1.0",
        "status": "license_evidence_audited_formal_scope_unchanged",
        "manifest": str(MANIFEST),
        "manifest_sha256": digest(MANIFEST),
        "sources": sources,
        "evidence_files": cards,
        "decisions": {
            "BMD-45": {
                "formal_train_eligible": True,
                "basis": "CC-BY-4.0 recorded in dataset card and manifest",
            },
            "VCoR": {
                "formal_train_eligible": True,
                "basis": "research-and-education-only; original authors recorded; commercial release remains out of scope",
            },
            "Stanford-Cars-196": {
                "formal_train_eligible": False,
                "basis": "dataset card explicitly says research benchmark and no SPDX license",
            },
            "human": {
                "formal_train_eligible": False,
                "basis": "manifest points to a source dataset card, but no card or explicit license is present in the authoritative dataset directory",
            },
        },
        "release_blockers": [
            "VCAS-STAGE2-COLOR-DOUBLE-REVIEW: VCoR colors require recorded two-person review before formal use",
            "VCAS-STAGE2-LICENSE-APPROVAL: Stanford-Cars-196 and human rows remain eval-only until explicit authorization is recorded",
        ],
    }
    out = EVIDENCE / "license-evidence-report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = EVIDENCE / "README.md"
    summary.write_text(
        "# Attribute-domain-v2 license evidence\n\n"
        "This folder records the source cards and the resulting formal-training decisions. "
        "It does not grant a license or change candidate labels. Stanford-Cars-196 remains "
        "eval-only because its card states research-benchmark terms without an explicit SPDX "
        "license. Human rows remain eval-only because the referenced source card is absent. "
        "VCoR is research-and-education-only and therefore remains limited to the declared "
        "training/evaluation scope; its separate color double-review gate is still open.\n\n"
        "Machine-readable details: `license-evidence-report.json`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "report": str(out), "summary": str(summary)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
