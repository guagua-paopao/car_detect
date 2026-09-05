import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "stage113", SCRIPTS / "build_stage113_domain_balanced_specialist_manifest.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(label, source, split="train", license_name="CC-BY-4.0"):
    return {
        "split": split,
        "body_type": label,
        "body_type_supervised": "true",
        "review_status": "approved",
        "source_dataset": source,
        "source_license": license_name,
        "research_only": "false",
        "sample_weight": "1.0",
    }


class Stage113ManifestTests(unittest.TestCase):
    def test_excludes_noncommercial_and_balances_weight_mass(self):
        rows = [row("light_truck", "BMD-45-RAW") for _ in range(4)]
        rows += [row("heavy_truck", "BMD-45") for _ in range(2)]
        rows += [row("heavy_truck", "MIO-TCD-Classification-2017", license_name="CC-BY-NC-SA-4.0")]
        rows += [row("light_truck", "BMD-45", split="validation")]
        output, summary = MODULE.build(rows)
        self.assertEqual(summary["excluded"]["noncommercial_or_research_only"], 1)
        self.assertEqual(summary["validation_rows_preserved"], 1)
        masses = summary["weighted_class_mass"]
        self.assertAlmostEqual(masses["light_truck"], masses["heavy_truck"], places=5)
        self.assertTrue(all(item.get("source_dataset") != "MIO-TCD-Classification-2017" for item in output))

    def test_test_rows_fail_closed(self):
        with self.assertRaises(RuntimeError):
            MODULE.build([row("light_truck", "BMD-45", split="test")])


if __name__ == "__main__":
    unittest.main()
