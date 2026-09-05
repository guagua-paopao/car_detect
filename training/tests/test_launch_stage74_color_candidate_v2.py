from __future__ import annotations

import unittest
from pathlib import Path


class Stage74ColorCandidateV2LauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "launch_stage74_color_candidate_v2.sh"
        ).read_text(encoding="utf-8")

    def test_pins_runner_matrix_and_new_output(self) -> None:
        self.assertIn("F2975D4FA1BDABFC3FCABA51A4755A824ABF366504AE7CAEE1C4C4F1CB1F2FAD", self.text)
        self.assertIn("0CA063A66B0B064E509334D08F7DD5F4695D7DBEE7FF7B22D1994B3D4695F726", self.text)
        self.assertIn("ATTR-STAGE74-COLOR-CCTV-JOINT-V2", self.text)

    def test_preflights_before_tmux_and_preserves_test_isolation(self) -> None:
        self.assertLess(self.text.index("--preflight-only"), self.text.index("tmux new-session"))
        self.assertNotIn("vcas_rtsp_demo_60s", self.text.lower())
        self.assertNotIn("--split test", self.text.lower())

    def test_refuses_overwrite_and_parallel_duplicate(self) -> None:
        self.assertIn('[[ ! -e "$output" ]]', self.text)
        self.assertIn("a Stage74 color training process is already active", self.text)


if __name__ == "__main__":
    unittest.main()
