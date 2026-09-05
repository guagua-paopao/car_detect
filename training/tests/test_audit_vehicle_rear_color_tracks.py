from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_vehicle_rear_color_tracks.py"
SPEC = importlib.util.spec_from_file_location("vehicle_rear_tracks", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VehicleRearColorTrackAuditTest(unittest.TestCase):
    def test_pair_json_aggregates_paths_without_emitting_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pairs.json"
            path.write_text(
                json.dumps({
                    "Set01": [[
                        ["data/x/plate/a.png"],
                        ["data/dataset2/Camera1/Set01/classes_carros/secret-a/1.png"],
                        ["data/x/plate/b.png"],
                        ["data/dataset2/Camera2/Set01/classes_carros/secret-b/2.png"],
                        1,
                        {"color": "Silver"},
                        {"color": "black"},
                    ]]
                }),
                encoding="utf-8",
            )
            report = MODULE.audit_pair_json(path)
            self.assertEqual(report["unique_vehicle_image_paths"], 2)
            self.assertEqual(report["unique_track_groups"], 2)
            self.assertEqual(report["unique_vehicle_image_color_counts"], {"black": 1, "silver": 1})
            self.assertNotIn("secret-a", json.dumps(report))

    def test_unsafe_archive_path_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pairs.json"
            path.write_text(
                json.dumps({"Set01": [[[], ["../escape.png"], [], [], 0, {"color": "red"}, {}]]}),
                encoding="utf-8",
            )
            report = MODULE.audit_pair_json(path)
            self.assertEqual(report["unsafe_paths"], 1)


if __name__ == "__main__":
    unittest.main()
