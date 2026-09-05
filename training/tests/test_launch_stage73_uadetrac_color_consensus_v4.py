from __future__ import annotations

import unittest
from pathlib import Path


class Stage73TrainOnlyV4LauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "launch_stage73_uadetrac_color_consensus_v4.sh"
        ).read_text(encoding="utf-8")

    def test_pins_repaired_script_and_manifest_local_dataset_root(self) -> None:
        self.assertIn("DA2B102D36222C18B3C06981A03EA7FF814598F389BA6A6B049E0B5D35848C83", self.text)
        self.assertIn("SOURCE_ROOT=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage70-uadetrac-training-pool-v2", self.text)
        self.assertIn('--dataset-root "$SOURCE_ROOT"', self.text)
        self.assertIn("ATTR-STAGE73-UA-COLOR-CONSENSUS-V4", self.text)

    def test_pins_rule_teachers_and_quality_gates(self) -> None:
        for token in (
            "BE3F794F0B3FB4D608459110486DEE353B262B56B932F204D6B038DDF051334F",
            "6F651BBA1C62082F13740728C96C82A074FBB3A8ECAF27FAC1F90504C6BB7383",
            "14BF423D277085C7DECEC236F1CC3931274AD778D0101E5D839BE43648254121",
            "--minimum-track-frames 3",
            "--track-decision-agreement 0.60",
            "--minimum-total 500",
            "--minimum-classes 5",
            "--minimum-rows-per-counted-class 30",
        ):
            self.assertIn(token, self.text)

    def test_refuses_overwrite_and_has_no_forbidden_input(self) -> None:
        lowered = self.text.lower()
        self.assertIn('if [[ -e "$OUTPUT_ROOT" ]]', self.text)
        self.assertNotIn("vcas_rtsp_demo_60s", lowered)
        self.assertNotIn("--split test", lowered)
        self.assertNotIn("/test/", lowered)


if __name__ == "__main__":
    unittest.main()
