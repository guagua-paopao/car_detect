from __future__ import annotations

import unittest
from pathlib import Path


class Stage74JointV2LauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "launch_stage74_joint_color_manifest_v2.sh"
        ).read_text(encoding="utf-8")

    def test_pins_weighted_builder_and_v2_output(self) -> None:
        self.assertIn("BEA0E347948F8A8E7610E970F2C105AE47092D8AEEFC8402AC8D9A766B698D26", self.text)
        self.assertIn("stage74-joint-color-v2", self.text)
        self.assertIn("new_rows_explicitly_weighted_as_cctv", self.text)

    def test_pins_all_source_evidence_and_gates(self) -> None:
        for token in (
            "F4576A85019C99A3E287692AEEBC1F470E21113E26A58DB264A2703F502D8A63",
            "B89DDB21C8649A70D085ECAD1811CF4206618B41A453C51ED138B833F6C74E58",
            "CFF3AD598A7AB78155A06CCF7BB87408E97347A971A6A4BF07CA93760721ED6A",
            "--minimum-new-rows 1000",
            "--minimum-new-classes 6",
            "--minimum-new-rows-per-class 30",
        ):
            self.assertIn(token, self.text)

    def test_no_overwrite_or_forbidden_input(self) -> None:
        lowered = self.text.lower()
        self.assertIn('if [[ -e "$OUTPUT_ROOT" ]]', self.text)
        self.assertNotIn("vcas_rtsp_demo_60s", lowered)
        self.assertNotIn("--split test", lowered)


if __name__ == "__main__":
    unittest.main()
