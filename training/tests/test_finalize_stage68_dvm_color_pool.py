from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "finalize_stage68_dvm_color_pool.py"
SPEC = importlib.util.spec_from_file_location("finalize_stage68", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_near_index_exactly_covers_hamming_distance_four() -> None:
    index = MODULE.NearIndex(); index.add(0)
    assert index.matches(0b1111) == [0]


def test_near_index_rejects_hamming_distance_five() -> None:
    index = MODULE.NearIndex(); index.add(0)
    assert index.matches(0b11111) == []
