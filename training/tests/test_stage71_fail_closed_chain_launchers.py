from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]


class Stage71FailClosedChainLaunchersTest(unittest.TestCase):
    def test_validation_launcher_is_validation_only_and_pinned(self) -> None:
        text = (
            TRAINING_ROOT / "scripts" / "run_stage71_fail_closed_pair_gate_after_validation.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("run_stage71_fail_closed_pair_gate.py", text)
        self.assertIn("ATTR-STAGE71-STUDENT-PAIR-TRACK-FAIL-CLOSED-V2", text)
        self.assertIn("--expected-v2-sweeper-sha256", text)
        self.assertNotIn("--split test", text)
        self.assertNotIn("vcas_rtsp_demo_60s", text)

    def test_final_test_consumes_v2_state(self) -> None:
        text = (
            TRAINING_ROOT / "scripts" / "run_stage71_final_test_after_fail_closed_v2.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "VALIDATION_STATE=$RUNS/ATTR-STAGE71-STUDENT-PAIR-TRACK-FAIL-CLOSED-V2.state.json",
            text,
        )
        self.assertIn("VALIDATION_SESSION=VCAS-STAGE71-RECOVERED-CHAIN-V2", text)
        self.assertNotIn("vcas_rtsp_demo_60s", text)

    def test_runner_hash_in_validation_launcher_matches_source(self) -> None:
        source = TRAINING_ROOT / "scripts" / "run_stage71_fail_closed_pair_gate.py"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        launcher = (
            TRAINING_ROOT / "scripts" / "run_stage71_fail_closed_pair_gate_after_validation.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(digest, launcher)

    def test_v3_final_test_applies_fixed_v2_gate_before_backend(self) -> None:
        text = (
            TRAINING_ROOT / "scripts" / "run_stage71_final_test_after_fail_closed_v3.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("apply_stage71_fail_closed_v2_test_gate.py", text)
        self.assertIn("evaluate_attribute_track_fusion_fail_closed_v2.py", text)
        self.assertIn("VALIDATION_SESSION=VCAS-STAGE71-RECOVERED-CHAIN-V3", text)
        self.assertNotIn("vcas_rtsp_demo_60s", text)


if __name__ == "__main__":
    unittest.main()
