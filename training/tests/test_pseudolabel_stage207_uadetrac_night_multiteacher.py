from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve()
SIBLING_SCRIPT = HERE.parent / "pseudolabel_stage207_uadetrac_night_multiteacher.py"
SCRIPT = (
    SIBLING_SCRIPT
    if SIBLING_SCRIPT.is_file()
    else HERE.parents[1] / "scripts" / "pseudolabel_stage207_uadetrac_night_multiteacher.py"
)
SPEC = importlib.util.spec_from_file_location("stage207", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_coarse_compatibility_is_fail_closed() -> None:
    assert MODULE.coarse_compatible("car", "sedan")
    assert MODULE.coarse_compatible("bus", "bus")
    assert MODULE.coarse_compatible("van", "van")
    assert not MODULE.coarse_compatible("bus", "sedan")
    assert not MODULE.coarse_compatible("others", "sedan")
    assert not MODULE.coarse_compatible("unknown", "suv")


def test_track_gate_requires_consistent_support() -> None:
    assert MODULE.qualify_track(["sedan", "sedan"], 4, 2, 0.5) == "sedan"
    assert MODULE.qualify_track(["sedan"], 4, 2, 0.5) is None
    assert MODULE.qualify_track(["sedan", "suv"], 4, 2, 0.5) is None
    assert MODULE.qualify_track(["unknown", "unknown"], 2, 2, 1.0) is None
