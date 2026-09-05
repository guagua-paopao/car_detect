from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stage74_color_candidate.py"
SPEC = importlib.util.spec_from_file_location("run_stage74_color_candidate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_usable_color_train_rows_excludes_validation_unknown_and_unsupervised() -> None:
    rows = [
        {"split": "train", "color_supervised": "true", "color": "red"},
        {"split": "validation", "color_supervised": "true", "color": "red"},
        {"split": "train", "color_supervised": "true", "color": "unknown"},
        {"split": "train", "color_supervised": "false", "color": "blue"},
    ]
    assert MODULE.usable_color_train_rows(rows) == [rows[0]]


def test_expected_supervised_quota_honors_explicit_and_scene_weights() -> None:
    rows = [
        {
            "sample_weight": "10",
            "night": "true",
            "small_target": "true",
            "occluded": "false",
            "truncated": "false",
            "stage71_origin": "cctv",
            "hard_score": "0",
        },
        {
            "sample_weight": "0.3",
            "night": "false",
            "small_target": "false",
            "occluded": "false",
            "truncated": "false",
            "stage71_origin": "dvm",
            "hard_score": "0",
        },
    ]
    quota = MODULE.expected_supervised_quota(
        rows,
        {
            "night_sample_weight": 2,
            "small_sample_weight": 1,
            "occlusion_sample_weight": 1,
            "hard_sample_weight": 0,
        },
    )
    expected = 40 / 40.3
    assert abs(quota["night"] - expected) < 1e-12
    assert abs(quota["small"] - expected) < 1e-12
    assert abs(quota["cctv"] - expected) < 1e-12


def test_expected_unlabeled_night_rejects_supervised_rows() -> None:
    rows = [{"split": "train", "body_type": "unknown", "color": "red"}]
    try:
        MODULE.expected_unlabeled_night(rows, 20)
    except RuntimeError as error:
        assert "contains a class target" in str(error)
    else:
        raise AssertionError("supervised unlabeled row was accepted")


def test_verify_report_requires_frozen_validation_projection_and_policy() -> None:
    matrix = {
        "quota_policy": {
            "minimum_color_train_rows": 118338,
            "minimum_new_cctv_rows": 1000,
            "minimum_supported_new_classes": 6,
        }
    }
    report = {
        "status": "pass_joint_research_manifest_available",
        "eligibility": "research-only_non-deployable",
        "output_manifest_sha256": "abc",
        "joint_train_rows": 120670,
        "retained_new_rows": 1618,
        "supported_new_classes": 6,
        "validation_projection_sha256_before": "same",
        "validation_projection_sha256_after": "same",
        "gates": {"group_leaks": 0},
        "policy": {
            "test_accessed": False,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    MODULE.verify_report(report, "abc", matrix)
    report["policy"]["frozen_video_used"] = True
    try:
        MODULE.verify_report(report, "abc", matrix)
    except RuntimeError as error:
        assert "frozen_video_used" in str(error)
    else:
        raise AssertionError("frozen-video policy violation was accepted")
