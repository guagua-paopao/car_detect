from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_stage172_trafficnight_border_quarantine.py"
SPEC = importlib.util.spec_from_file_location("stage172_quarantine", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_fixture(root: Path, shapes: list[dict[str, object]]) -> tuple[Path, Path]:
    archive = root / "trafficnight.zip"
    document = {"imageWidth": 100, "imageHeight": 80, "shapes": shapes}
    with zipfile.ZipFile(archive, "w") as opened:
        opened.writestr("set/frame.json", json.dumps(document))
    archive_sha = MODULE.sha256_file(archive)
    source_audit = root / "source-audit.json"
    source_audit.write_text(
        json.dumps(
            {
                "status": "fail",
                "archive_sha256": archive_sha,
                "invalid_shapes": 1,
                "failures": ["invalid shapes: 1"],
                "crc_verified": True,
                "invalid_images": 0,
                "invalid_json": 0,
                "unknown_class_counts": {},
                "images_without_labels": 0,
                "labels_without_images": 0,
                "duplicate_normalized_names": 0,
                "frozen_markers": 0,
            }
        ),
        encoding="utf-8",
    )
    return archive, source_audit


class Stage172TrafficNightQuarantineTest(unittest.TestCase):
    def test_quarantines_border_shape_without_clipping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shapes = [
                {"label": "Car", "shape_type": "polygon", "points": [[1, 1], [10, 1], [10, 10], [1, 10]]},
                {"label": "Car", "shape_type": "polygon", "points": [[-2, 2], [5, 2], [5, 8], [0, 8]]},
            ]
            archive, source = write_fixture(root, shapes)
            report = MODULE.audit_border_quarantine(
                archive,
                source,
                expected_archive_sha256=MODULE.sha256_file(archive),
                max_quarantine_fraction=0.6,
                minimum_retained_objects=1,
                minimum_retained_images=1,
            )
            self.assertEqual(report["status"], "pass_quarantine_plan")
            self.assertEqual(report["retained_objects"], 1)
            self.assertEqual(report["quarantined_objects"], 1)
            self.assertEqual(report["quarantine_reason_counts"], {"out_of_bounds": 1})
            self.assertEqual(report["quarantine_objects"][0]["points"][0], [-2.0, 2.0])

    def test_fails_on_unrelated_source_audit_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shapes = [{"label": "Car", "shape_type": "polygon", "points": [[1, 1], [2, 1], [2, 2], [1, 2]]}]
            archive, source = write_fixture(root, shapes)
            document = json.loads(source.read_text(encoding="utf-8"))
            document["failures"].append("invalid images: 1")
            document["invalid_images"] = 1
            source.write_text(json.dumps(document), encoding="utf-8")
            report = MODULE.audit_border_quarantine(
                archive,
                source,
                expected_archive_sha256=MODULE.sha256_file(archive),
                max_quarantine_fraction=0.5,
                minimum_retained_objects=1,
                minimum_retained_images=1,
            )
            self.assertEqual(report["status"], "fail")
            self.assertTrue(any("unrelated" in failure or "exclusively" in failure for failure in report["failures"]))

    def test_fails_when_quarantine_fraction_exceeds_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shapes = [
                {"label": "Car", "shape_type": "polygon", "points": [[1, 1], [2, 1], [2, 2], [1, 2]]},
                {"label": "Car", "shape_type": "polygon", "points": [[-2, 1], [2, 1], [2, 2], [0, 2]]},
            ]
            archive, source = write_fixture(root, shapes)
            report = MODULE.audit_border_quarantine(
                archive,
                source,
                expected_archive_sha256=MODULE.sha256_file(archive),
                max_quarantine_fraction=0.1,
                minimum_retained_objects=1,
                minimum_retained_images=1,
            )
            self.assertEqual(report["status"], "fail")
            self.assertTrue(any("quarantine fraction" in failure for failure in report["failures"]))

    def test_quarantines_non_polygon_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shapes = [
                {"label": "Car", "shape_type": "polygon", "points": [[1, 1], [2, 1], [2, 2], [1, 2]]},
                {"label": "Car", "shape_type": "linestrip", "points": [[3, 3], [4, 3], [4, 4], [3, 4]]},
            ]
            archive, source = write_fixture(root, shapes)
            report = MODULE.audit_border_quarantine(
                archive,
                source,
                expected_archive_sha256=MODULE.sha256_file(archive),
                max_quarantine_fraction=0.6,
                minimum_retained_objects=1,
                minimum_retained_images=1,
            )
            self.assertEqual(report["status"], "pass_quarantine_plan")
            self.assertEqual(report["quarantine_reason_counts"], {"unsupported_shape_type:linestrip": 1})


if __name__ == "__main__":
    unittest.main()
