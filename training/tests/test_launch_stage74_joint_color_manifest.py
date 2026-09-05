from __future__ import annotations

import unittest
from pathlib import Path


class Stage74JointLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "launch_stage74_joint_color_manifest.sh"
        ).read_text(encoding="utf-8")

    def test_pins_all_evidence(self) -> None:
        for token in (
            "EC724567C3CD8A6BF82E7198D2E32CA7539E40450046CA108277FA4A1CBC2D14",
            "F4576A85019C99A3E287692AEEBC1F470E21113E26A58DB264A2703F502D8A63",
            "B89DDB21C8649A70D085ECAD1811CF4206618B41A453C51ED138B833F6C74E58",
            "9184ADA81444F7737FC305E9F4CCD9FDE04FE83C25B2F6603A8EDDD59CB0190D",
            "CFF3AD598A7AB78155A06CCF7BB87408E97347A971A6A4BF07CA93760721ED6A",
            "EF3EC1BA288F5023F9CCF40A69A0F615A5B4CEE7863FE9545FD3670D8BA1E7B5",
        ):
            self.assertIn(token, self.text)

    def test_freezes_joint_support_and_near_duplicate_gates(self) -> None:
        for token in (
            "--near-duplicate-hamming 2",
            "--minimum-new-rows 1000",
            "--minimum-new-classes 6",
            "--minimum-new-rows-per-class 30",
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
