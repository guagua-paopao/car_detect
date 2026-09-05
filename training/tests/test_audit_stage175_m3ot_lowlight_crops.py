from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_stage175_m3ot_lowlight_crops.py"
SPEC = importlib.util.spec_from_file_location("stage175_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def valid_row() -> dict[str, str]:
    return {
        "split": "train", "body_type": "unknown", "body_type_supervised": "false",
        "coarse_body_family": "", "color": "unknown", "color_supervised": "false",
        "source_dataset": "M3OT", "source_label": "vehicle", "source_license": "CC-BY-4.0",
        "license_train_eligible": "true", "lighting": "night", "low_light": "true",
        "night": "true", "small_target": "true", "pseudo_label": "false",
        "teacher_consensus": "false", "research_only": "true", "deployment_eligible": "false",
        "stage175_origin": "m3ot_official_rgb_train_lowlight_track_capped",
    }


class Stage175AuditTests(unittest.TestCase):
    def test_accepts_unknown_safe_contract(self) -> None:
        self.assertEqual(MODULE.validate_row_contract(valid_row()), [])

    def test_rejects_fabricated_body_truth(self) -> None:
        row = valid_row()
        row["body_type"] = "suv"
        row["body_type_supervised"] = "true"
        self.assertIn("body truth fabricated", MODULE.validate_row_contract(row))

    def test_rejects_inconsistent_night_flag(self) -> None:
        row = valid_row()
        row["night"] = "false"
        self.assertIn("night flag mismatch", MODULE.validate_row_contract(row))


if __name__ == "__main__":
    unittest.main()
