from __future__ import annotations

import unittest
from pathlib import Path


class Stage73ColorRuleLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "launch_stage73_color_rule_validation.sh"
        ).read_text(encoding="utf-8")

    def test_pins_validation_inputs_and_two_teachers(self) -> None:
        for token in (
            "stage70-specialist-manifests-v4-no-color-pseudo/attribute_manifest.stage70-color.csv",
            "C20AFDE14A89494E26B7CB20C98E9C1ABAD3DB7E776C1E3751D8A81CD46A2C9E",
            "--checkpoint \"production=$PRODUCTION\"",
            "--checkpoint \"stage71_color=$STAGE71_COLOR\"",
            "6F651BBA1C62082F13740728C96C82A074FBB3A8ECAF27FAC1F90504C6BB7383",
            "14BF423D277085C7DECEC236F1CC3931274AD778D0101E5D839BE43648254121",
        ):
            self.assertIn(token, self.text)

    def test_refuses_evidence_overwrite_and_preserves_failure_report(self) -> None:
        self.assertIn('if [[ -e "$OUTPUT_ROOT" ]]', self.text)
        self.assertIn("set +e", self.text)
        self.assertIn('if [[ -f "$REPORT" ]]', self.text)
        self.assertIn("fail_closed_no_rule_passed", self.text)

    def test_has_no_test_or_frozen_video_input(self) -> None:
        lowered = self.text.lower()
        self.assertNotIn("vcas_rtsp_demo_60s", lowered)
        self.assertNotIn("--split test", lowered)
        self.assertNotIn("/test/", lowered)


if __name__ == "__main__":
    unittest.main()
