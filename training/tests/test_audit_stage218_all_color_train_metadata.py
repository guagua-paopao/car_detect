import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_stage218_all_color_train_metadata.py"
if not SCRIPT.is_file():
    SCRIPT = Path(__file__).with_name("audit_stage218_all_color_train_metadata.py")
SPEC = importlib.util.spec_from_file_location("stage218", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage218MetadataAuditTests(unittest.TestCase):
    def test_authoritative_truth_requires_positive_and_rejects_teacher(self):
        base = {"color": "red", "color_supervised": "true"}
        self.assertEqual(
            MODULE.classify_color_truth({**base, "review_method": "official_registration_color"}),
            "authoritative",
        )
        self.assertEqual(
            MODULE.classify_color_truth({**base, "review_method": "teacher_consensus"}),
            "model_derived",
        )
        self.assertEqual(MODULE.classify_color_truth(base), "ambiguous")

    def test_scene_proxy_is_not_promoted_to_night(self):
        self.assertEqual(
            MODULE.classify_scene({"lighting": "low_luminance_proxy", "night": "unknown"}),
            "low_luminance_proxy",
        )
        self.assertEqual(MODULE.classify_scene({"lighting": "night"}), "night_metadata_positive")
        self.assertEqual(MODULE.classify_scene({"lighting": "unknown"}), "unknown")

    def test_test_and_holdout_paths_are_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            test_path = root / "independent-test" / "manifest.csv"
            test_path.parent.mkdir()
            test_path.touch()
            skipped, marker = MODULE.should_skip_manifest(test_path, root)
            self.assertTrue(skipped)
            self.assertEqual(marker, "test")

    def test_sha_key_preferred_over_path(self):
        sha = "a" * 64
        key = MODULE.image_key({"sha256": sha, "image_path": "x.jpg"}, Path("manifest.csv"))
        self.assertEqual(key, ("sha256", sha))

    def test_frozen_marker_is_fatal_signal(self):
        self.assertTrue(MODULE.row_has_frozen_marker({"image_path": "/x/vcas_rtsp_demo_60s/a.jpg"}))
        self.assertFalse(MODULE.row_has_frozen_marker({"image_path": "/x/train/a.jpg"}))


if __name__ == "__main__":
    unittest.main()
