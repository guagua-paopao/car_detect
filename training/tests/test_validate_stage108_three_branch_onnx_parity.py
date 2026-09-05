from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_stage108_three_branch_onnx_parity.py"
SPEC = importlib.util.spec_from_file_location("stage108_parity", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage108ParityTest(unittest.TestCase):
    def test_three_exact_routes_are_required(self) -> None:
        report = {
            "status": "pass_onnx_exported_candidate_only",
            "exports": {
                role: {"consumed_output": consumed}
                for role, (_, consumed) in MODULE.ROUTES.items()
            },
            "policy": {"frozen_video_used": False},
        }
        MODULE.validate_export_report(report)
        report["exports"].pop("truck_specialist")
        with self.assertRaisesRegex(RuntimeError, "exactly three"):
            MODULE.validate_export_report(report)

    def test_route_mismatch_and_frozen_use_fail_closed(self) -> None:
        report = {
            "status": "pass_onnx_exported_candidate_only",
            "exports": {
                role: {"consumed_output": consumed}
                for role, (_, consumed) in MODULE.ROUTES.items()
            },
            "policy": {"frozen_video_used": False},
        }
        report["exports"]["color"]["consumed_output"] = "body_type"
        with self.assertRaisesRegex(RuntimeError, "routing output mismatch"):
            MODULE.validate_export_report(report)
        report["exports"]["color"]["consumed_output"] = "color"
        report["policy"]["frozen_video_used"] = True
        with self.assertRaisesRegex(RuntimeError, "frozen video"):
            MODULE.validate_export_report(report)


if __name__ == "__main__":
    unittest.main()
