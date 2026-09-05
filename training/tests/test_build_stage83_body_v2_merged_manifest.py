from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "build_stage83_body_v2_merged_manifest.py"
SPEC = importlib.util.spec_from_file_location("stage83_merge", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage83BodyV2MergeTests(unittest.TestCase):
    def test_group_key_prefers_track_identity(self) -> None:
        row = {"track_group": "track-1", "video_id": "video-1", "source_group": "source-1"}
        self.assertEqual(MODULE.group_key(row), "track_group:track-1")

    def test_hash_helpers_accept_manifest_aliases(self) -> None:
        self.assertEqual(MODULE.content_hash({"crop_sha256": "ABC"}), "abc")
        self.assertEqual(MODULE.content_hash({"image_sha256": "DEF"}), "def")
        self.assertEqual(MODULE.perceptual_hash({"crop_dhash64": "ABCDEF0123456789"}), "abcdef0123456789")

    def test_frozen_markers_are_explicit(self) -> None:
        self.assertIn("vcas_rtsp_demo_60s", MODULE.FROZEN_MARKERS)

    def test_stage82_normalization_preserves_research_scope_and_loader_contract(self) -> None:
        row = {
            "review_status": "approved_research_only",
            "license_train_eligible": "true",
            "source_license": "CC-BY-NC-SA-4.0",
            "body_type": "pickup",
            "coarse_body_family": "truck",
        }
        output = MODULE.normalize_stage82_row(row)
        self.assertEqual(output["review_status"], "approved")
        self.assertEqual(output["research_only"], "true")
        self.assertEqual(output["deployment_eligible"], "false")
        self.assertEqual(output["coarse_body_family"], "")
        self.assertEqual(
            output["stage83_coarse_family_normalization"],
            "pickup_exact_not_truck_family",
        )

    def test_stage82_normalization_rejects_license_drift(self) -> None:
        row = {
            "review_status": "approved_research_only",
            "license_train_eligible": "true",
            "source_license": "unknown",
        }
        with self.assertRaises(ValueError):
            MODULE.normalize_stage82_row(row)

    def test_pickup_coarse_conflict_counter_returns_an_integer(self) -> None:
        rows = [
            {"stage83_origin": "stage82_mio", "body_type": "pickup", "coarse_body_family": "truck"},
            {"stage83_origin": "stage82_mio", "body_type": "pickup", "coarse_body_family": ""},
            {"stage83_origin": "stage72", "body_type": "pickup", "coarse_body_family": "truck"},
        ]
        value = MODULE.pickup_coarse_conflict_count(rows)
        self.assertIsInstance(value, int)
        self.assertEqual(value, 1)


if __name__ == "__main__":
    unittest.main()
