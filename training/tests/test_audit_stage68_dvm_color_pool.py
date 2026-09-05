from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_stage68_dvm_color_pool.py"
SPEC = importlib.util.spec_from_file_location("audit_stage68", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_near_pair_summary_is_exact_through_distance_four() -> None:
    pairs, matched, distances = MODULE.near_pair_summary([0b1111], [0], 4)
    assert pairs == 1
    assert matched == 1
    assert distances == {4: 1}


def test_near_pair_summary_rejects_distance_five() -> None:
    pairs, matched, distances = MODULE.near_pair_summary([0b11111], [0], 4)
    assert pairs == 0
    assert matched == 0
    assert distances == {}


def test_pixel_sha_prefers_crop_sha256() -> None:
    assert MODULE.pixel_sha({"sha256": "a", "crop_sha256": "b"}) == "b"
