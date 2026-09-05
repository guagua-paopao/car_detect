from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_stage108_three_branch_onnx_chain.py"
SPEC = importlib.util.spec_from_file_location("stage108_runner", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Stage108RunnerTest(unittest.TestCase):
    def test_only_backend_eligible_stage107_state_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "status": "pass_backend_eligible",
                        "composite_gates": {"body": True, "color": True},
                    }
                ),
                encoding="utf-8",
            )
            state = root / "state.json"
            state.write_text(
                json.dumps(
                    {
                        "status": "pass_backend_eligible",
                        "backend_eligible": True,
                        "report": str(report),
                        "report_sha256": sha(report),
                        "test_used_for_selection": False,
                        "frozen_video_used": False,
                    }
                ),
                encoding="utf-8",
            )
            path, digest = MODULE.validate_stage107_state(state, sha(state))
            self.assertEqual(path, report)
            self.assertEqual(digest, sha(report))

            unsafe = json.loads(state.read_text(encoding="utf-8"))
            unsafe["frozen_video_used"] = True
            state.write_text(json.dumps(unsafe), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                MODULE.validate_stage107_state(state, sha(state))


if __name__ == "__main__":
    unittest.main()
