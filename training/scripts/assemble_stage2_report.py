#!/usr/bin/env python3
"""Assemble Stage 2 data-audit evidence and explicit release blocker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "training" / "artifacts" / "attribute-domain-v2"


def load(name: str) -> dict:
    return json.loads((ART / name).read_text(encoding="utf-8"))


def load_optional(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    candidate = load("formal-candidate-report.json")
    validation = load("formal-candidate-validation.json")
    baseline = load("baseline-test.json")
    review_packet = load_optional(ART / "vcor-color-double-review" / "review-packet-report.json")
    license_evidence = load_optional(ART / "license-evidence" / "license-evidence-report.json")
    top_confusions = []
    for actual, values in baseline["body_type"]["confusion"].items():
        for predicted, count in values.items():
            if actual != predicted:
                top_confusions.append({"actual": actual, "predicted": predicted, "count": count})
    top_confusions.sort(key=lambda x: x["count"], reverse=True)
    report = {
        "schema_version": "1.0",
        "stage": "stage2_supplement_real_attribute_data",
        "status": "blocked_pending_required_human_review",
        "production_model_unchanged": True,
        "candidate_manifest": candidate,
        "candidate_validation": validation,
        "hard_example_mining": {
            "priority_field": "hard_example_priority",
            "high_priority_rows": candidate["candidate_counts"]["hard_example_priority"].get("high", 0),
            "body_type_top_confusions_from_independent_test": top_confusions[:15],
        },
        "blockers": [
            {
                "id": "VCAS-STAGE2-COLOR-DOUBLE-REVIEW",
                "severity": "release_blocking",
                "rows": candidate["color_review_queue"]["rows"],
                "source": "VCoR",
                "reason": "source labels are present but the required two-person color review is not recorded",
                "required_action": "complete two-person review; disagreements become unknown or manual-review queue",
            },
            {
                "id": "VCAS-STAGE2-LICENSE-APPROVAL",
                "severity": "formal-train-blocking-for-excluded-sources",
                "rows": candidate["excluded_train_counts"]["rows"],
                "source_counts": candidate["excluded_train_counts"]["source"],
                "reason": "Stanford/human source terms are not explicit enough for formal training",
                "required_action": "record license approval or keep rows eval-only",
            },
        ],
        "review_packet": review_packet,
        "license_evidence": license_evidence,
        "exploratory_training": {
            "status": "stopped_before_completion",
            "experiment": "ATTR-A-224",
            "reason": "stopped when the color double-review blocker was confirmed; checkpoint and log retained on cloud",
            "next_experiment_not_started": "ATTR-B-256",
            "local_artifacts": {
                str(p.relative_to(ART)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((ART / "stage3-exploratory").glob("*") if (ART / "stage3-exploratory").exists() else [])
                if p.is_file()
            },
        },
        "artifacts": {
            "formal_candidate_manifest_sha256": candidate["output_sha256"],
            "candidate_validation": str(ART / "formal-candidate-validation.json"),
            "vcor_review_packet": str(ART / "vcor-color-double-review" / "README.md") if review_packet else None,
            "license_evidence": str(ART / "license-evidence" / "README.md") if license_evidence else None,
            "review_merge_validator": str(ROOT / "training" / "scripts" / "apply_vcor_color_review.py"),
        },
        "next_step_after_unblock": "apply reviewed colors and license approvals, rebuild candidate manifest, rerun validation, then resume ATTR-A-224 from last.pt and run B/C/D/E comparisons",
    }
    (ART / "stage2-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# VCAS Stage 2 — data audit and candidate filtering",
        "",
        "Status: **blocked pending required human review**. Formal model and registry remain unchanged.",
        "",
        "- Frozen evaluation set: train 32,901 / validation 4,000 / test 4,000.",
        f"- Filtered candidate: train `{candidate['candidate_counts']['split'].get('train', 0)}`, validation `{candidate['candidate_counts']['split'].get('validation', 0)}`, test `{candidate['candidate_counts']['split'].get('test', 0)}`.",
        f"- Candidate validation: `{validation['status']}`; approved supervised crops `{validation['attributes']['approved_supervised_crops']}`.",
        f"- Hard-example queue: `{candidate['candidate_counts']['hard_example_priority'].get('high', 0)}` high-priority rows (small/medium/low-light/night or usable crops).",
        "",
        "## Blocking conditions",
        "",
        f"1. `{candidate['color_review_queue']['rows']}` VCoR color rows have source labels but no recorded two-person review. They are retained only as candidate/evaluation evidence and are not release-ready.",
        f"2. `{candidate['excluded_train_counts']['rows']}` train rows from Stanford/human sources remain eval-only because the current license record is not explicit enough for formal training.",
        "",
        "A blinded review packet for all VCoR color rows is ready under `vcor-color-double-review/`: the reviewer queue omits source labels, while the audit key preserves provenance for reconciliation. No labels were modified.",
        "License evidence is recorded under `license-evidence/`; it confirms VCoR's research-and-education-only scope and leaves Stanford/human rows eval-only because their terms are not explicit enough for formal training.",
        "The fail-closed merge validator `training/scripts/apply_vcor_color_review.py` was dry-run against the pending queue and correctly rejected all 7,234 incomplete rows; it will update train only and preserve validation/test labels.",
        "",
        "An exploratory ATTR-A-224 run was stopped at epoch 9 after the blocker was confirmed; `best.pt`, `last.pt`, and `training_log.jsonl` are preserved under `stage3-exploratory/` and the cloud `runs/stage3/ATTR-A-224` directory. ATTR-B-256 was not started. No candidate was deployed.",
        "",
        "After review/approval, rebuild the manifest, rerun the input contract, resume ATTR-A-224, then execute the remaining crop/model/distillation comparisons. Machine-readable evidence: `stage2-report.json` and `formal-candidate-report.json`.",
    ]
    (ART / "stage2-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(ART / "stage2-report.json"), "summary": str(ART / "stage2-summary.md")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
