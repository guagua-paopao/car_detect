import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PREP = load_module("stage192_prep", ROOT / "scripts" / "prepare_stage192_tesla_teacher_inputs.py")
MERGE = load_module("stage192_merge", ROOT / "scripts" / "merge_stage192_tesla_teacher_quota.py")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


class Stage192Test(unittest.TestCase):
    def test_prepare_and_merge_keep_heads_separate(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fields = [
                "split", "source_image_name", "crop_path", "stage191_train_eligible",
                "body_type", "body_type_supervised", "color", "color_supervised", "lighting",
            ]
            colors = ("white", "blue", "red", "green")
            rows = []
            for index in range(3026):
                rows.append({
                    "split": "train", "source_image_name": f"image-{index:04d}.jpg",
                    "crop_path": f"/tmp/crop-{index:04d}.jpg",
                    "stage191_train_eligible": "true" if index < 500 else "false",
                    "body_type": "sedan" if index % 2 == 0 else "suv",
                    "body_type_supervised": "true", "color": colors[index % 4],
                    "color_supervised": "true", "lighting": "Dark" if index % 2 else "Light",
                })
            stage191_manifest = root / "stage191.csv"
            body_input, color_input = root / "body-input.csv", root / "color-input.csv"
            write_csv(stage191_manifest, fields, rows)
            old = sys.argv
            try:
                sys.argv = ["prep", "--manifest", str(stage191_manifest), "--body-output", str(body_input), "--color-output", str(color_input)]
                self.assertEqual(PREP.main(), 0)
            finally:
                sys.argv = old
            with body_input.open(encoding="utf-8", newline="") as handle:
                body_rows = list(csv.DictReader(handle))
            self.assertEqual(body_rows[0]["color_supervised"], "false")
            self.assertEqual(body_rows[500]["body_type_supervised"], "false")

            body_fields = fields + ["teacher_consensus", "teacher_consensus_class", "teacher_consensus_confidence"]
            color_fields = fields + ["color_teacher_consensus", "color_teacher_class", "color_teacher_confidence"]
            body_review, color_review = root / "body-review.csv", root / "color-review.csv"
            write_csv(body_review, body_fields, [dict(row, teacher_consensus="accepted" if i < 400 else "rejected", teacher_consensus_class=row["body_type"], teacher_consensus_confidence="0.9") for i, row in enumerate(rows)])
            write_csv(color_review, color_fields, [dict(row, color_teacher_consensus="accepted" if i < 200 else "rejected", color_teacher_class=row["color"], color_teacher_confidence="0.9") for i, row in enumerate(rows)])

            dummy = root / "stage191.json"; dummy.write_text("{}", encoding="utf-8")
            body_report = root / "body.json"; body_report.write_text("{}", encoding="utf-8")
            color_report = root / "color.json"; color_report.write_text("{}", encoding="utf-8")
            stage178 = root / "stage178.json"; stage178.write_text(json.dumps({"states": {"track_balanced_effective_images": {"body_exact": {"total": 1000, "existing_explicit_night": 10, "existing_explicit_low_light": 20}}}}), encoding="utf-8")
            stage179 = root / "stage179.json"; stage179.write_text(json.dumps({"scene_totals": {"track_balanced_effective_representatives": {"total": 800, "existing_explicit_night": 8}}}), encoding="utf-8")
            stage188 = root / "stage188.json"; stage188.write_text(json.dumps({"counts": {"component_train_eligible_rows": 100, "component_train_eligible_lowlight_proxy_body": {"bus": 2, "heavy_truck": 10, "light_truck": 8}}}), encoding="utf-8")
            output_manifest, output_report = root / "merged.csv", root / "merged.json"
            argv = [
                "merge", "--stage191-manifest", str(stage191_manifest), "--stage191-report", str(dummy),
                "--body-manifest", str(body_review), "--body-report", str(body_report),
                "--color-manifest", str(color_review), "--color-report", str(color_report),
                "--stage178-report", str(stage178), "--stage179-report", str(stage179),
                "--stage188-report", str(stage188), "--output-manifest", str(output_manifest),
                "--output-report", str(output_report),
            ]
            old = sys.argv
            try:
                sys.argv = argv
                self.assertEqual(MERGE.main(), 0)
            finally:
                sys.argv = old
            report = json.loads(output_report.read_text(encoding="utf-8"))
            self.assertEqual(report["component"]["body_teacher_accepted"], 400)
            self.assertEqual(report["component"]["color_teacher_accepted"], 200)
            self.assertEqual(report["component"]["confirmed_natural_night_rows"], 0)
            self.assertFalse(report["gates"]["training_authorized"])


if __name__ == "__main__":
    unittest.main()
