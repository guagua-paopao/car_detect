#!/usr/bin/env python3
"""Audit the full VCAS goal against current evidence and release gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "training" / "artifacts" / "attribute-domain-v2"


def load(name: str) -> dict:
    path = ART / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "missing", "path": str(path)}


def sha(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    stage1 = load("stage1-report.json")
    stage2 = load("stage2-report.json")
    stage4 = load("stage4-trajectory-fusion-report.json")
    queue_report = load("vcor-color-double-review/review-packet-report.json")
    license_report = load("license-evidence/license-evidence-report.json")
    audit = {
        "schema_version": "1.0",
        "status": "blocked_pending_external_review_and_authorization",
        "production_model_unchanged": True,
        "stages": {
            "stage1": {
                "status": "complete",
                "evidence": "stage1-report.json",
                "baseline_targets": {
                    "body_precision_0.93": False,
                    "body_coverage_0.45": True,
                    "color_precision_0.93": True,
                    "color_coverage_0.25": True,
                },
            },
            "stage2": {
                "status": stage2.get("status", "unknown"),
                "evidence": ["stage2-report.json", "formal-candidate-validation.json", "license-evidence/license-evidence-report.json"],
                "blockers": stage2.get("blockers", []),
            },
            "stage3": {
                "status": "stopped_before_completion",
                "evidence": "stage3-exploratory/",
                "formal_comparisons_complete": False,
                "formal_candidate_deployed": False,
            },
            "stage4": {
                "status": stage4.get("status", "unknown"),
                "evidence": "stage4-trajectory-fusion-report.json",
                "unit_evidence": True,
                "video_stability_rate": "pending",
                "video_label_switch_count": "pending",
            },
            "stage5": {"status": "not_started_release_blocked", "threshold_calibration_complete": False},
            "stage6": {"status": "not_started_release_blocked", "fixed_video_replay_complete": False},
            "stage7": {"status": "not_started_release_blocked", "deployment_allowed": False},
        },
        "external_blockers": {
            "vcor_double_review": {
                "rows": queue_report.get("rows"),
                "status": queue_report.get("status"),
                "queue_sha256": queue_report.get("reviewer_queue_sha256"),
                "filled_rows": 0,
            },
            "license_authorization": license_report.get("decisions", {}),
        },
        "release_invariants": {
            "fixed_video_not_used_for_tuning": True,
            "test_set_not_used_for_training_or_selection": True,
            "unknown_color_not_coerced": True,
            "formal_registry_overwritten": False,
            "tensorrt_parity_for_new_candidate": False,
            "rollback_preserved": True,
        },
        "artifact_hashes": {
            "stage1_report": sha(ART / "stage1-report.json"),
            "stage2_report": sha(ART / "stage2-report.json"),
            "stage4_report": sha(ART / "stage4-trajectory-fusion-report.json"),
            "review_packet_report": sha(ART / "vcor-color-double-review/review-packet-report.json"),
            "license_evidence_report": sha(ART / "license-evidence/license-evidence-report.json"),
        },
        "required_external_action": "complete the blinded VCoR double-review queue and provide explicit Stanford/human authorization or accept eval-only scope",
        "resume_sequence": "merge reviewed train colors -> validate inputs -> resume ATTR-A-224 -> run A/B/C/D/E -> calibrate -> replay fixed video -> TensorRT parity/deployment gates",
    }
    out = ART / "goal-completion-audit.json"
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "path": str(out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
