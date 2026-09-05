from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_stage170_fresh_holdout.py"
SPEC = importlib.util.spec_from_file_location("stage170", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Stage170FreshHoldoutTests(unittest.TestCase):
    def passing_state(self) -> dict:
        return {
            "status": "integrated_candidate_pass_stage160_holdout_build_authorized",
            "authorization": {
                "stage160_new_holdout_build": True,
                "independent_test_inference": False,
                "onnx_backend_gates": False,
                "frozen_video_replay": False,
                "deployment": False,
            },
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "backend_gates_run": False,
            "deployment_performed": False,
        }

    def test_authorization_accepts_only_safe_stage169_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text(json.dumps(self.passing_state()), encoding="utf-8")
            result = MODULE.load_authorization(path, sha(path))
            self.assertTrue(result["authorization"]["stage160_new_holdout_build"])

    def test_authorization_rejects_inference_permission(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state = self.passing_state()
            state["authorization"]["independent_test_inference"] = True
            path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unsafe Stage169 authorization"):
                MODULE.load_authorization(path, sha(path))

    def test_main_fails_authorization_before_source_manifest_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "state.json"
            state = self.passing_state()
            state["status"] = "integrated_candidate_rejected_fail_closed"
            state["authorization"]["stage160_new_holdout_build"] = False
            state_path.write_text(json.dumps(state), encoding="utf-8")
            arguments = [
                str(SCRIPT),
                "--authorization-state", str(state_path),
                "--expected-authorization-state-sha256", sha(state_path),
                "--policy", str(root / "missing-policy.json"),
                "--expected-policy-sha256", "0" * 64,
                "--source-manifest", str(root / "must-not-be-read.csv"),
                "--expected-source-manifest-sha256", "0" * 64,
                "--reference-manifest", str(root / "missing-reference.csv"),
                "--expected-reference-manifest-sha256", "0" * 64,
                "--allowed-image-root", str(root),
                "--output-root", str(root / "output"),
            ]
            with mock.patch.object(sys, "argv", arguments):
                with self.assertRaisesRegex(RuntimeError, "Stage169 did not pass"):
                    MODULE.main()

    def test_group_contains_source_camera_and_identity(self):
        group = MODULE.row_group({"source_dataset": "road", "camera_id": "c1", "track_id": "t9"})
        self.assertEqual(group, "road|c1|track:t9")
        first = MODULE.row_group({"source_dataset": "road", "camera_id": "c1", "vehicle_id": "v7"})
        second = MODULE.row_group({"source_dataset": "road", "camera_id": "c2", "vehicle_id": "v7"})
        self.assertEqual(first, second)
        historical = MODULE.row_group({
            "source_dataset": "road",
            "camera_id": "c3",
            "stage159_validation_group": "sealed-track-4",
        })
        self.assertEqual(historical, "road|track:sealed-track-4")
        with self.assertRaisesRegex(RuntimeError, "identity"):
            MODULE.row_group({"source_dataset": "road", "camera_id": "c1"})

    def test_unknown_blur_does_not_inflate_complex_quota(self):
        self.assertEqual(MODULE.hard_flags({"blur": "unknown"}), (False, False, False, False))

    def test_success_authorizes_only_one_time_inference(self):
        authorization = MODULE.post_build_authorization(True)
        self.assertTrue(authorization["one_time_independent_test_inference"])
        self.assertFalse(authorization["onnx_backend_gates"])
        self.assertFalse(authorization["frozen_video_replay"])
        self.assertFalse(authorization["deployment"])
        self.assertFalse(any(MODULE.post_build_authorization(False).values()))

    def small_policy(self) -> dict:
        return {
            "minimum_support": {
                "body_total_known_truth_rows": 10,
                "body_rows_per_common_class": 1,
                "body_groups_per_common_class": 1,
                "body_rows_for_each_rare_pickup_or_generic_truck": 1,
                "body_groups_for_each_rare_pickup_or_generic_truck": 1,
                "color_total_known_truth_rows": 10,
                "color_rows_per_class": 1,
                "color_groups_per_class": 1,
                "real_tracks_with_at_least_three_ordered_frames": 0,
            },
            "hard_scene_minimums": {
                "night_or_low_light_fraction": 0.0,
                "small_target_fraction": 0.0,
                "occluded_truncated_or_overlap_fraction": 0.0,
                "complex_scene_fraction": 0.0,
            },
        }

    def complete_taxonomy_rows(self) -> list[dict[str, str]]:
        return [
            {
                "stage170_group": f"group-{index}",
                "body_type": body,
                "body_type_supervised": "true",
                "color": color,
                "color_supervised": "true",
            }
            for index, (body, color) in enumerate(
                zip(sorted(MODULE.KNOWN_BODY), sorted(MODULE.KNOWN_COLOR))
            )
        ]

    def test_support_requires_every_body_and_color_label(self):
        rows = self.complete_taxonomy_rows()
        support = MODULE.summarize_support(rows, self.small_policy())
        self.assertTrue(support["all_gates_pass"])
        without_silver = [row for row in rows if row["color"] != "silver"]
        support = MODULE.summarize_support(without_silver, self.small_policy())
        self.assertFalse(support["gates"]["color_rows:silver"])
        self.assertFalse(support["all_gates_pass"])

    def test_truth_must_be_explicit_and_not_teacher_generated(self):
        self.assertTrue(MODULE.explicit_truth({"ground_truth_source": "official registration metadata"}))
        self.assertFalse(MODULE.explicit_truth({"review_method": "teacher consensus"}))
        self.assertFalse(MODULE.explicit_truth({}))

    def test_group_truth_conflict_fails_closed(self):
        rows = self.complete_taxonomy_rows()
        rows.append({
            **rows[0],
            "body_type": "suv" if rows[0]["body_type"] != "suv" else "sedan",
        })
        support = MODULE.summarize_support(rows, self.small_policy())
        self.assertEqual(support["groups_with_truth_conflicts"], 1)
        self.assertFalse(support["gates"]["group_truth_consistency"])
        self.assertFalse(support["all_gates_pass"])

    def test_near_duplicate_is_allowed_only_inside_same_group(self):
        tree = MODULE.BKTree()
        tree.add(int("0123456789abcdef", 16), "group-a")
        self.assertFalse(tree.conflict(int("0123456789abcdef", 16), 4, "group-a"))
        self.assertTrue(tree.conflict(int("0123456789abcdef", 16), 4, "group-b"))


if __name__ == "__main__":
    unittest.main()
