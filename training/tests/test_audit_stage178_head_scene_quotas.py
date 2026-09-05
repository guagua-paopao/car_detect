from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_stage178_head_scene_quotas.py"
SPEC = importlib.util.spec_from_file_location("stage178", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage178Tests(unittest.TestCase):
    def test_scope_selection_supports_partial_heads_and_coarse_body(self) -> None:
        row = {
            "body_type": "unknown", "body_type_supervised": "false",
            "color": "red", "color_supervised": "true", "coarse_body_family": "car",
        }
        self.assertEqual(
            MODULE.selected_scopes(row, {"sedan"}, {"red"}),
            ["loader_union", "body_exact_or_coarse", "color_exact"],
        )

    def test_positive_only_addition_formula_uses_final_denominator(self) -> None:
        from collections import Counter
        result = MODULE.finish(Counter(total=100, effective_adverse=10), 0.30)
        self.assertEqual(result["minimum_all_positive_additions_to_reach_final_target"], 29)


if __name__ == "__main__":
    unittest.main()
