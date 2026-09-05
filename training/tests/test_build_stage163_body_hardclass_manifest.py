import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_stage163_body_hardclass_manifest.py"
SPEC = importlib.util.spec_from_file_location("stage163_builder", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage163BuilderTests(unittest.TestCase):
    def write_source(self, root: Path, rows: list[dict[str, str]]) -> Path:
        source = root / "source.csv"
        fields = sorted({key for row in rows for key in row})
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return source

    def run_builder(self, rows: list[dict[str, str]]):
        root = Path(tempfile.mkdtemp())
        source = self.write_source(root, rows)
        output, report = root / "out.csv", root / "report.json"
        old = MODULE.parse_args
        MODULE.parse_args = lambda: type("Args", (), {
            "source": source,
            "expected_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output": output,
            "report": report,
        })()
        try:
            MODULE.main()
        finally:
            MODULE.parse_args = old
        with output.open(newline="", encoding="utf-8") as handle:
            output_rows = list(csv.DictReader(handle))
        return output_rows, json.loads(report.read_text(encoding="utf-8"))

    def test_weights_only_supervised_train_rows(self):
        rows, report = self.run_builder([
            {"split": "train", "body_type": "mpv", "body_type_supervised": "true", "sample_weight": "2", "image_path": "safe/a.jpg"},
            {"split": "validation", "body_type": "mpv", "body_type_supervised": "true", "sample_weight": "2", "image_path": "safe/b.jpg"},
            {"split": "train", "body_type": "suv", "body_type_supervised": "false", "sample_weight": "2", "image_path": "safe/c.jpg"},
        ])
        self.assertEqual(rows[0]["sample_weight"], "5.600000")
        self.assertEqual(rows[1]["sample_weight"], "2.000000")
        self.assertEqual(rows[2]["sample_weight"], "2.000000")
        self.assertEqual(report["train_only_reweighted_rows"], 1)
        self.assertFalse(report["test_accessed"])
        self.assertFalse(report["frozen_video_used"])

    def test_frozen_marker_fails_closed(self):
        root = Path(tempfile.mkdtemp())
        source = self.write_source(root, [{"split": "train", "body_type": "suv", "body_type_supervised": "true", "image_path": "demo/vcas_rtsp_demo_60s/frame.jpg"}])
        old = MODULE.parse_args
        MODULE.parse_args = lambda: type("Args", (), {
            "source": source,
            "expected_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output": root / "out.csv",
            "report": root / "report.json",
        })()
        try:
            with self.assertRaises(RuntimeError):
                MODULE.main()
        finally:
            MODULE.parse_args = old


if __name__ == "__main__":
    unittest.main()
