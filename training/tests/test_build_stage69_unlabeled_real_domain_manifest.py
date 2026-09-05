from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_stage69_unlabeled_real_domain_manifest.py"
SPEC = importlib.util.spec_from_file_location("stage69_unlabeled", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_demotes_only_approved_train_rows_without_creating_labels() -> None:
    rows = [
        {"split": "train", "review_status": "approved", "body_type": "sedan", "color": "red", "body_type_supervised": "true", "color_supervised": "true", "source_dataset": "real"},
        {"split": "validation", "review_status": "approved", "body_type": "suv", "color": "blue"},
    ]
    output, counters = MODULE.build_rows(rows)
    assert len(output) == 1
    assert output[0]["body_type"] == "unknown"
    assert output[0]["color"] == "unknown"
    assert output[0]["body_type_supervised"] == "false"
    assert output[0]["color_supervised"] == "false"
    assert counters["excluded_non_train"] == 1


def test_frozen_marker_fails_closed() -> None:
    rows = [{"split": "train", "review_status": "approved", "image_path": "vcas_rtsp_demo_60s/frame.jpg"}]
    try:
        MODULE.build_rows(rows)
    except RuntimeError:
        return
    raise AssertionError("frozen marker must fail closed")
