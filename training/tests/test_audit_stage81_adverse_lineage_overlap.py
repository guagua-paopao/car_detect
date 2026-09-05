import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_stage81_adverse_lineage_overlap.py"
SPEC = importlib.util.spec_from_file_location("stage81_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


FIELDS = [
    "image_path", "sha256", "stage66_licensed_adverse", "body_type",
    "body_type_supervised", "color", "color_supervised",
]


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class AdverseLineageOverlapTests(unittest.TestCase):
    def test_passes_when_only_missing_adverse_row_is_unsupervised(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            successor = root / "successor.csv"
            supervised = [
                {
                    "image_path": f"a{i}.jpg", "sha256": f"hash{i}", "stage66_licensed_adverse": "true",
                    "body_type": "van", "body_type_supervised": "true", "color": "unknown",
                    "color_supervised": "false",
                }
                for i in range(100)
            ]
            unknown = {
                "image_path": "unknown.jpg", "sha256": "unknownhash", "stage66_licensed_adverse": "true",
                "body_type": "unknown", "body_type_supervised": "false", "color": "unknown",
                "color_supervised": "false",
            }
            write_manifest(source, supervised + [unknown])
            write_manifest(successor, supervised)

            report = MODULE.build_report(source, successor)

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["overlap"]["missing_adverse_rows"], 1)
            self.assertEqual(report["overlap"]["missing_body_supervised_rows"], 0)
            self.assertIs(report["decision"]["reimport_stage66"], False)

    def test_fails_when_supervised_adverse_truth_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            successor = root / "successor.csv"
            rows = [
                {
                    "image_path": f"a{i}.jpg", "sha256": f"hash{i}", "stage66_licensed_adverse": "true",
                    "body_type": "bus", "body_type_supervised": "true", "color": "unknown",
                    "color_supervised": "false",
                }
                for i in range(100)
            ]
            write_manifest(source, rows)
            write_manifest(successor, rows[:-2])

            report = MODULE.build_report(source, successor)

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["overlap"]["missing_body_supervised_rows"], 2)
            self.assertIsNone(report["decision"]["reimport_stage66"])


if __name__ == "__main__":
    unittest.main()
