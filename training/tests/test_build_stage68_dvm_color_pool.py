from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_stage68_dvm_color_pool.py"
SPEC = importlib.util.spec_from_file_location("build_stage68_dvm_color_pool", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_train_quota_is_requested_effective_scale() -> None:
    assert sum(MODULE.TRAIN_QUOTAS.values()) == 118338


def test_member_parser_excludes_color_from_group_identity() -> None:
    base = "19586296/resized_DVM_clean/resized_DVM_clean/Abarth/124 Spider/2017"
    black = f"{base}/Black/Abarth$$124 Spider$$2017$$Black$$2_1$$10$$image_1.jpg"
    white = f"{base}/White/Abarth$$124 Spider$$2017$$White$$2_1$$10$$image_2.jpg"
    black_group, black_color = MODULE.parse_member(black)
    white_group, white_color = MODULE.parse_member(white)
    assert black_group == white_group
    assert black_color == "Black"
    assert white_color == "White"


def test_unlisted_and_ambiguous_red_brown_are_fail_closed() -> None:
    assert "Unlisted" not in MODULE.SOURCE_TO_COLOR
    assert "Maroon" not in MODULE.SOURCE_TO_COLOR
    assert "Burgundy" not in MODULE.SOURCE_TO_COLOR


def test_near_duplicate_index_detects_distance_three() -> None:
    index = MODULE.NearDuplicateIndex(3)
    index.add(0x1234567890ABCDEF, "red", "first")
    assert index.match(0x1234567890ABCDE8, "red") == "first"
    assert index.match(0x1234567890ABCDE8, "blue") is None
    assert index.match(0xEDCBA9876F543210, "red") is None
