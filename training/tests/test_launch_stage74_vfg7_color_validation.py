from __future__ import annotations

import unittest
from pathlib import Path


class Stage74Vfg7ColorValidationLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "launch_stage74_vfg7_color_validation.sh"
        ).read_text(encoding="utf-8")

    def test_pins_candidate_baselines_manifest_and_evaluators(self) -> None:
        for token in (
            "0C17E979C79A9C16978EF46D628650A3D883B311599AF0C24F8B7B3A2264C581",
            "14BF423D277085C7DECEC236F1CC3931274AD778D0101E5D839BE43648254121",
            "6F651BBA1C62082F13740728C96C82A074FBB3A8ECAF27FAC1F90504C6BB7383",
            "5582A65FD02874B6E84776D9E00A883D4AD0CB4FA579AC10C829583EC751BCE3",
            "5A924C27BC927A82C79C158341E4C3188333EDD76D07F4B4CF84A050A19A6899",
            "974A684D34DC45E4572D8EFA5E703BEE2DE486F3B8BA44649A1909BEF37EF2BB",
            "20881E01AEC678E5F59F2A4EEBB851CF9C015CCCA5D4009780024A988459DA90",
        ):
            self.assertIn(token, self.text)

    def test_is_validation_only_and_refuses_overwrite(self) -> None:
        lowered = self.text.lower()
        self.assertIn('[[ ! -e "$OUTPUT_ROOT" ]]', self.text)
        self.assertIn('\\"split\\":\\"validation\\"', self.text)
        self.assertNotIn("vcas_rtsp_demo_60s", lowered)
        self.assertNotIn("--split test", lowered)

    def test_compares_production_stage71_and_stage74(self) -> None:
        for model in ("production=$PRODUCTION", "stage71_color=$STAGE71", "stage74_color=$STAGE74"):
            self.assertIn(model, self.text)


if __name__ == "__main__":
    unittest.main()
