from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "export_stage108_three_branch_onnx.py"
SPEC = importlib.util.spec_from_file_location("stage108_export", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def passing_report() -> dict:
    return {
        "status": "pass_backend_eligible",
        "composite_gates": {"a": True, "b": True},
        "all_composite_gates_pass": True,
        "policy": {
            "test_accessed": True,
            "test_used_for_selection": False,
            "threshold_or_temperature_search": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "backend_gates_run": False,
            "deployment_performed": False,
        },
    }


class Stage108ExportTest(unittest.TestCase):
    def test_passing_stage107_report_is_accepted(self) -> None:
        MODULE.validate_final_report(passing_report())

    def test_failed_or_unsafe_stage107_report_is_rejected(self) -> None:
        failed = passing_report()
        failed["composite_gates"]["a"] = False
        with self.assertRaisesRegex(RuntimeError, "failed composite"):
            MODULE.validate_final_report(failed)
        unsafe = passing_report()
        unsafe["policy"]["frozen_video_used"] = True
        with self.assertRaisesRegex(RuntimeError, "frozen_video_used"):
            MODULE.validate_final_report(unsafe)

    def test_main_and_specialist_checkpoint_contracts_are_distinct(self) -> None:
        labels = {
            "labels_version": "vehicle-labels-v2-offline-candidate",
            "body_types": ["sedan", "truck", "light_truck", "heavy_truck", "unknown"],
            "colors": ["black", "white", "unknown"],
        }
        main = {
            **labels,
            "body_hierarchy": "truck_family",
        }
        specialist = {
            "labels_version": labels["labels_version"],
            "body_types": ["light_truck", "heavy_truck", "unknown"],
            "colors": ["unknown"],
        }
        MODULE.validate_checkpoint_contract("body_main", main, labels)
        MODULE.validate_checkpoint_contract("color", main, labels)
        MODULE.validate_checkpoint_contract("truck_specialist", specialist, labels)
        specialist["body_types"] = ["truck", "unknown"]
        with self.assertRaisesRegex(RuntimeError, "body contract"):
            MODULE.validate_checkpoint_contract("truck_specialist", specialist, labels)


if __name__ == "__main__":
    unittest.main()
