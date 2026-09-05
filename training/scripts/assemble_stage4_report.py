#!/usr/bin/env python3
"""Record trajectory-level fusion implementation and executable evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "training" / "artifacts" / "attribute-domain-v2"
TEST = ROOT / "out" / "tmp" / "vehicle_cascade_runtime_test.exe"


def sha(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    report = {
        "schema_version": "1.0",
        "stage": "stage4_track_level_attribute_fusion",
        "status": "implementation_and_unit_evidence_pass",
        "production_model_unchanged": True,
        "configuration": {
            "min_valid_observations": 3,
            "max_observations": 5,
            "body_type_threshold": 0.75,
            "color_threshold": 0.70,
            "quality_weighting": "quality_score * prediction_confidence; unknown excluded from numerator",
            "quality_gate": "96x64 minimum, sharpness/exposure/occlusion/truncation checks before queue",
            "hysteresis": "confirmed labels persist; three consecutive threshold-clearing reverse observations required to switch",
        },
        "evidence": {
            "source_header": str(ROOT / "include" / "business" / "vehicle_cascade_runtime.h"),
            "source_implementation": str(ROOT / "src" / "business" / "vehicle_cascade_runtime.cpp"),
            "test_source": str(ROOT / "tests" / "vehicle_cascade_runtime_test.cpp"),
            "test_command": "g++ -std=c++2a vehicle_cascade_runtime.cpp vehicle_model_contract.cpp vehicle_cascade_runtime_test.cpp && vehicle_cascade_runtime_test.exe",
            "test_status": "PASS: M3 vehicle tracking, quality, queue, fusion, and cascade runtime",
            "test_binary": str(TEST),
            "test_binary_sha256": sha(TEST),
            "assertions": [
                "new track starts unknown",
                "body type becomes stable at three observations",
                "color remains unknown until threshold-clearing evidence",
                "single contradictory observation does not flip confirmed color",
                "three consecutive reverse observations switch label and increment label_switches",
                "stale crop sequence is rejected",
            ],
        },
        "video_stability_rate": {
            "status": "pending",
            "reason": "fixed-video replay remains paused while Stage 2 color double-review blocker is unresolved",
        },
    }
    (ART / "stage4-trajectory-fusion-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = """# VCAS Stage 4 — trajectory-level attribute fusion

Implementation and unit evidence pass. The formal model and registry remain unchanged.

- New tracks begin as `unknown`.
- Three valid observations are required before a head can stabilize.
- At most five observations are retained, weighted by crop quality and prediction confidence.
- The quality gate excludes undersized, blurred, exposure-invalid, occluded, or truncated crops before major updates.
- A confirmed label does not change on one contradictory frame; three consecutive threshold-clearing reverse observations are required.
- The C++ runtime test passes, including stale sequence rejection and label-switch hysteresis.

Video-level stability rate and label-switch counts on the fixed 60-second video remain pending because the Stage 2 color double-review blocker still pauses replay and release work.
"""
    (ART / "stage4-trajectory-fusion-summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"status": report["status"], "binary_sha256": report["evidence"]["test_binary_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
