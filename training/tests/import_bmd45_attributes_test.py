from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def write_split(root: Path, name: str, folders: list[str]) -> None:
    split = root / name
    categories = [
        {"id": 0, "name": "Sedan"},
        {"id": 1, "name": "Two-wheeler"},
    ]
    images = []
    annotations = []
    for index, folder in enumerate(folders):
        path = split / folder / f"{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (320, 200), (80 + index, 100, 120)).save(path)
        images.append({"id": index, "file_name": f"{folder}/{index}.png", "width": 320, "height": 200})
        annotations.append({"id": index * 2, "image_id": index, "category_id": 0, "bbox": [40, 30, 160, 100], "iscrowd": 0})
        annotations.append({"id": index * 2 + 1, "image_id": index, "category_id": 1, "bbox": [210, 30, 100, 100], "iscrowd": 0})
    (split / "_annotations.coco.json").write_text(
        json.dumps({"categories": categories, "images": images, "annotations": annotations}),
        encoding="utf-8",
    )


def test_import_bmd45_attributes_sequence_isolation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_split(source, "BMD-45-Train", ["images_000"])
    write_split(source, "BMD-45-Val", ["images_000", "images_001", "images_002"])
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "labels_version": "vehicle-labels-v1",
                "body_types": [
                    "sedan", "suv", "mpv", "van", "pickup", "bus",
                    "light_truck", "heavy_truck", "other", "unknown",
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    script = Path(__file__).resolve().parents[1] / "scripts" / "import_bmd45_attributes.py"
    subprocess.run(
        [
            sys.executable, str(script), "--bmd-root", str(source),
            "--output-root", str(output), "--labels", str(labels),
            "--train-per-class", "5", "--validation-per-class", "5",
            "--test-per-class", "5", "--license-review-ref", "TEST-LICENSE",
        ],
        check=True,
    )
    with (output / "attribute_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["split"] for row in rows} == {"train", "validation", "test"}
    assert all(row["body_type"] == "sedan" for row in rows)
    assert all(row["color_supervised"] == "false" for row in rows)
    assert {row["video_id"] for row in rows if row["split"] == "validation"} == {
        "bmd45_validation_images_000"
    }
    assert {row["video_id"] for row in rows if row["split"] == "test"} == {
        "bmd45_test_images_001", "bmd45_test_images_002"
    }
    assert len(rows) == 4
