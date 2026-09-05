import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_stage80_v2_color_teacher_after_stage77.py"
)
SPEC = importlib.util.spec_from_file_location("stage80", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage77ActionTests(unittest.TestCase):
    def test_waiting_and_running_states_wait(self):
        self.assertEqual(MODULE.stage77_action("waiting_for_stage76"), "wait")
        self.assertEqual(MODULE.stage77_action("running_validation"), "wait")

    def test_only_clean_completion_is_ready(self):
        self.assertEqual(MODULE.stage77_action("complete_validation_only"), "ready")
        self.assertEqual(MODULE.stage77_action("failed_closed"), "fail")


class CommandTests(unittest.TestCase):
    def test_command_arguments_preserve_skip_test_and_omit_metadata(self):
        result = MODULE.command_arguments(
            {
                "candidate_id": "candidate",
                "epochs": 8,
                "skip_test": True,
                "unlabeled_manifest": None,
            }
        )
        self.assertEqual(result, ["--epochs", "8", "--skip-test"])

    def test_false_flags_are_not_emitted(self):
        self.assertEqual(MODULE.command_arguments({"skip_test": False}), [])


if __name__ == "__main__":
    unittest.main()
