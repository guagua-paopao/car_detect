from __future__ import annotations

import runpy
from pathlib import Path


test_path = Path(__file__).resolve().parent / "test_pseudolabel_stage207_uadetrac_night_multiteacher.py"
namespace = runpy.run_path(str(test_path))
namespace["test_coarse_compatibility_is_fail_closed"]()
namespace["test_track_gate_requires_consistent_support"]()
print("2 focused tests passed")
