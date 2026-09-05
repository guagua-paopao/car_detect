import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "training" / "scripts" / "launch_stage72_body_taxonomy_v2_builder.sh"


class Stage72BodyLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = LAUNCHER.read_text(encoding="utf-8")

    def test_sha_pins_authoritative_inputs(self) -> None:
        self.assertIn("7dc5379483578db504eea15c8d6920c350e9770aabaed3e025f0ee82a0e7b490", self.text)
        self.assertIn("89c625bd44e28749453b0470488207234d10f6e52ffe343c1f4e70eb49b861bb", self.text)
        self.assertIn("22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f", self.text)
        self.assertIn("sha256sum \"$builder_script\"", self.text)
        self.assertIn("sha256sum \"$input_manifest\"", self.text)
        self.assertIn("sha256sum \"$labels\"", self.text)

    def test_waits_for_training_and_other_builder_io(self) -> None:
        self.assertIn("train_attribute.py", self.text)
        self.assertIn("recover_stage71_", self.text)
        self.assertIn("build_stage72_.*manifest.py", self.text)
        self.assertIn("less than 4 GiB free", self.text)

    def test_runs_background_and_fails_closed_on_existing_evidence(self) -> None:
        self.assertIn('session_name="VCAS-STAGE72B-BODY-TAXONOMY-V2"', self.text)
        self.assertIn("tmux new-session -d", self.text)
        self.assertIn("refusing to overwrite existing Stage72B evidence", self.text)
        self.assertIn("--near-duplicate-hamming 1", self.text)

    def test_scope_excludes_production_test_and_frozen_assets(self) -> None:
        lowered = self.text.lower()
        for forbidden in (
            "vehicle-attr-agent-e-224", "model_registry", "vehicle_analytics.yaml",
            "vcas_rtsp_demo_60s", "36-48", "/test", "rm ",
        ):
            self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
