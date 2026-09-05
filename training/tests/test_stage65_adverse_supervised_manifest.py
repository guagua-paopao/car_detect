from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TRAINING = Path(__file__).resolve().parents[1]
SCRIPT = TRAINING / "scripts" / "build_stage65_adverse_supervised_manifest.py"
SPEC = importlib.util.spec_from_file_location("stage65_builder", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage65AdverseManifestTest(unittest.TestCase):
    def test_keyed_rejects_duplicate_crop_hash(self) -> None:
        with self.assertRaises(RuntimeError):
            MODULE.keyed([{"crop_sha256": "a"}, {"crop_sha256": "a"}], "test")

    def test_truthy_is_strict(self) -> None:
        self.assertTrue(MODULE.truthy("true"))
        self.assertFalse(MODULE.truthy("pending"))

    def test_source_contains_required_fail_closed_policies(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("official_car_is_partial_family_not_exact_subtype", source)
        self.assertIn("unsupported_merged_color_truth_rejected", source)
        self.assertIn("validation_and_test_membership_preserved", source)
        self.assertIn("frozen_video_used", source)


if __name__ == "__main__":
    unittest.main()
