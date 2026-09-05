import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "training" / "scripts" / "launch_stage72_dvm_fine_color_builder.sh"


class Stage72LauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = LAUNCHER.read_text(encoding="utf-8")

    def test_shell_syntax(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        probe = subprocess.run(
            [bash, "--version"],
            text=True,
            capture_output=True,
            check=False,
        )
        if probe.returncode != 0:
            self.skipTest("local bash/WSL service is unavailable")
        completed = subprocess.run(
            [bash, "-n"],
            input=self.text,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_inputs_and_thresholds_are_sha_pinned(self) -> None:
        self.assertIn(
            'expected_builder_sha="870c8d517ef431cf8453a36067792692062f13fbc618208c271e3f040c58a938"',
            self.text,
        )
        self.assertIn(
            'expected_source_sha="9c7bc1296e7c5d2caa7cabfc8d34b031e1dbdbccb9be355ce36b41d7d9f0910b"',
            self.text,
        )
        for flag, value in (
            ("--minimum-train-rows", "118338"),
            ("--minimum-gray-rows", "1000"),
            ("--minimum-silver-rows", "1000"),
            ("--minimum-yellow-rows", "500"),
            ("--minimum-brown-rows", "500"),
        ):
            self.assertIn(f"{flag} {value}", self.text)

    def test_fails_closed_while_training_or_evidence_exists(self) -> None:
        self.assertIn("train_attribute.py", self.text)
        self.assertIn("recover_stage71_", self.text)
        self.assertIn("refusing to overwrite existing Stage72A evidence", self.text)
        self.assertIn("less than 4 GiB free", self.text)

    def test_scope_excludes_production_test_and_frozen_assets(self) -> None:
        lowered = self.text.lower()
        self.assertNotIn("vehicle-attr-agent-e-224", lowered)
        self.assertNotIn("model_registry", lowered)
        self.assertNotIn("vehicle_analytics.yaml", lowered)
        self.assertNotIn("vcas_rtsp_demo_60s", lowered)
        self.assertNotIn("36-48", lowered)
        self.assertNotIn("/test", lowered)
        self.assertNotIn("rm ", lowered)


if __name__ == "__main__":
    unittest.main()
