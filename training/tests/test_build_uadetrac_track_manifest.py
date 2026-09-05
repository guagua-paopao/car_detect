from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_uadetrac_track_manifest.py"


def make_partition(root: Path, sequence: str, weather: str) -> None:
    image_dir = root / f"detrac_1_{sequence}"
    image_dir.mkdir(parents=True)
    frames = []
    for frame_number in range(1, 7):
        image = np.full((120, 200, 3), (20, 40, 180), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        encoded.tofile(str(image_dir / f"img{frame_number:05d}.jpg"))
        frames.append(
            f'''<frame num="{frame_number}"><target_list><target id="1">'''
            f'''<box left="20" top="20" width="80" height="60"/>'''
            f'''<attribute orientation="0" speed="0" trajectory_length="6" '''
            f'''truncation_ratio="0" vehicle_type="bus" color="red"/>'''
            f'''</target></target_list></frame>'''
        )
    xml = (
        f'''<sequence name="{sequence}">'''
        f'''<sequence_attribute camera_state="stable" weather="{weather}"/>'''
        f'''<ignored_region/>'''
        + "".join(frames)
        + "</sequence>"
    )
    (root / f"{sequence}.xml").write_text(xml, encoding="utf-8")


class TrackManifestTest(unittest.TestCase):
    def test_end_to_end_track_windows_and_split_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            training = tmp_path / "training"
            test = tmp_path / "test"
            training.mkdir(); test.mkdir()
            make_partition(training, "MVI_20011", "night")
            make_partition(test, "MVI_40001", "rainy")
            output = tmp_path / "output"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--training-root", str(training),
                    "--test-root", str(test),
                    "--output-root", str(output),
                    "--window", "3",
                    "--windows-per-track", "2",
                    "--workers", "2",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stdout + "\n" + completed.stderr)
            report = json.loads((output / "dataset_report.json").read_text(encoding="utf-8"))
            with (output / "attribute_manifest.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["video_leaks"], [])
            self.assertEqual(report["track_leaks"], [])
            self.assertEqual(report["tracks_parsed"], 2)
            self.assertEqual(len(rows), 12)
            self.assertEqual({row["body_type"] for row in rows}, {"bus"})
            self.assertEqual({row["color"] for row in rows}, {"red"})
            self.assertTrue(all(row["body_type_supervised"] == "true" for row in rows))
            self.assertTrue(all(row["color_supervised"] == "true" for row in rows))
            self.assertEqual(len({row["image_path"] for row in rows}), len(rows))
            self.assertEqual(
                {row["split"] for row in rows if row["video_id"].endswith("MVI_40001")},
                {"test"},
            )
            self.assertTrue(
                all(row["night"] == "true" for row in rows if row["video_id"].endswith("MVI_20011"))
            )


if __name__ == "__main__":
    unittest.main()
