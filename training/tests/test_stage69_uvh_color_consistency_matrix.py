from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stage69_uvh_color_consistency_matrix.py"
SPEC = importlib.util.spec_from_file_location("stage69_runner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

MATRIX = Path(
    os.environ.get(
        "STAGE69_MATRIX",
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "attribute-domain-v2"
        / "stage69-uvh-color-consistency-matrix.json",
    )
)


def test_command_arguments_emits_boolean_flags_and_skips_metadata() -> None:
    result = MODULE.command_arguments({"candidate_id": "x", "skip_test": True, "epochs": 3, "purpose": "p"})
    assert result == ["--skip-test", "--epochs", "3"]


def test_command_arguments_skips_false_and_none() -> None:
    assert MODULE.command_arguments({"skip_test": False, "teacher_checkpoint": None}) == []


def test_unlabeled_root_covers_all_dataset_sources() -> None:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    consistency_candidates = [item for item in matrix["candidates"] if item.get("unlabeled_manifest")]
    assert consistency_candidates
    assert {item["unlabeled_root"] for item in consistency_candidates} == {
        "/root/autodl-tmp/vcas/datasets"
    }
