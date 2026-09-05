from __future__ import annotations

import unittest
from pathlib import Path


class Stage73TrainOnlyLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "launch_stage73_uadetrac_color_consensus_v2.sh"
        ).read_text(encoding="utf-8")

    def test_pins_source_rule_teachers_and_script(self) -> None:
        for token in (
            "5B7FD3571EA0ECDFFEF73062B0930E4BADFF63773DC13B33FA44D9DCA8C10ACF",
            "AB1F20AB1EF2E9DA0B26B48D49EFD0D4B9D324A22EC58DCDB2E102B371EBC369",
            "BE3F794F0B3FB4D608459110486DEE353B262B56B932F204D6B038DDF051334F",
            "6F651BBA1C62082F13740728C96C82A074FBB3A8ECAF27FAC1F90504C6BB7383",
            "14BF423D277085C7DECEC236F1CC3931274AD778D0101E5D839BE43648254121",
        ):
            self.assertIn(token, self.text)

    def test_freezes_track_and_class_support_gates(self) -> None:
        for token in (
            "--minimum-track-frames 3",
            "--audit-frames-per-track 5",
            "--track-decision-agreement 0.60",
            "--minimum-total 500",
            "--minimum-classes 5",
            "--minimum-rows-per-counted-class 30",
        ):
            self.assertIn(token, self.text)

    def test_refuses_overwrite_and_has_no_forbidden_input(self) -> None:
        lowered = self.text.lower()
        self.assertIn('if [[ -e "$OUTPUT_ROOT" ]]', self.text)
        self.assertIn("fail_closed_insufficient_train_only_evidence", self.text)
        self.assertNotIn("vcas_rtsp_demo_60s", lowered)
        self.assertNotIn("--split test", lowered)
        self.assertNotIn("/test/", lowered)


if __name__ == "__main__":
    unittest.main()
