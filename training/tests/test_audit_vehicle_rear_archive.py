from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_vehicle_rear_archive.py"
SPEC = importlib.util.spec_from_file_location("vehicle_rear_audit", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VehicleRearArchiveAuditTest(unittest.TestCase):
    def test_member_path_validation_rejects_traversal_and_absolute_paths(self) -> None:
        self.assertTrue(MODULE.safe_member_name("vehicle-reid/data/file.json"))
        self.assertFalse(MODULE.safe_member_name("../outside.txt"))
        self.assertFalse(MODULE.safe_member_name("/absolute.txt"))

    def test_json_summary_and_attribute_collection_are_bounded(self) -> None:
        value = {"vehicle": {"color": "silver", "type": "truck"}, "items": [{"class": 2}]}
        summary = MODULE.summarize_json(value)
        self.assertEqual(summary["type"], "object")
        output = MODULE.Counter()
        MODULE.collect_attribute_values(value, output)
        self.assertEqual(output["vehicle.color=silver"], 1)
        self.assertEqual(output["vehicle.type=truck"], 1)

    def test_metadata_suffixes_include_annotation_formats(self) -> None:
        self.assertIn(".json", MODULE.TEXT_SUFFIXES)
        self.assertIn(".txt", MODULE.TEXT_SUFFIXES)


if __name__ == "__main__":
    unittest.main()
