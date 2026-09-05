from __future__ import annotations

import unittest
from pathlib import Path


class Stage74BmdSeedLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "launch_stage74_bmd45_seed_color_revalidation.sh"
        ).read_text(encoding="utf-8")

    def test_pins_script_source_rule_and_teachers(self) -> None:
        for token in (
            "8B526E12802EB284FB6B9FEA81CF743FD7B9FE4C151DE06697A49DC5CEBCAA2A",
            "6A9402A4010954D590CA474513C2007583AE07E430C152C74F0CDB6477AB573A",
            "BE3F794F0B3FB4D608459110486DEE353B262B56B932F204D6B038DDF051334F",
            "6F651BBA1C62082F13740728C96C82A074FBB3A8ECAF27FAC1F90504C6BB7383",
            "14BF423D277085C7DECEC236F1CC3931274AD778D0101E5D839BE43648254121",
        ):
            self.assertIn(token, self.text)

    def test_freezes_dedup_and_class_support_gates(self) -> None:
        for token in (
            "--near-duplicate-hamming 4",
            "--minimum-total 300",
            "--minimum-classes 5",
            "--minimum-rows-per-counted-class 20",
        ):
            self.assertIn(token, self.text)

    def test_no_test_or_frozen_input_and_no_overwrite(self) -> None:
        lowered = self.text.lower()
        self.assertIn('if [[ -e "$OUTPUT_ROOT" ]]', self.text)
        self.assertNotIn("vcas_rtsp_demo_60s", lowered)
        self.assertNotIn("--split test", lowered)
        self.assertNotIn("/test/", lowered)


if __name__ == "__main__":
    unittest.main()
