import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from training.scripts.audit_stage188_myvid_semantics_decontamination import run


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_only_exact_original_classes_become_component_candidates(tmp_path: Path) -> None:
    semantics = {
        "status": "pass_conservative_mapping",
        "mapping": {
            "0": {"body_type": "unknown", "coarse_body_family": "passenger_vehicle", "body_supervised": False},
            "1": {"body_type": "unknown", "coarse_body_family": "light_commercial_vehicle", "body_supervised": False},
            "2": {"body_type": "light_truck", "coarse_body_family": "truck", "body_supervised": True},
            "3": {"body_type": "heavy_truck", "coarse_body_family": "truck", "body_supervised": True},
            "4": {"body_type": "bus", "coarse_body_family": "bus", "body_supervised": True},
            "5": {"body_type": "unknown", "coarse_body_family": "out_of_scope", "body_supervised": False},
        },
        "policy": {
            "validation_images_opened": False,
            "test_images_opened": False,
            "frozen_video_used": False,
            "color_truth_available": False,
            "confirmed_per_frame_night_truth_available": False,
        },
    }
    semantics_path = tmp_path / "semantics.json"
    semantics_path.write_text(json.dumps(semantics), encoding="utf-8")
    fields = [
        "split", "source_dataset", "source_image", "source_frame_id", "source_video_id",
        "export_augmentation", "box_index", "numeric_class_id", "x_center", "y_center",
        "width", "height", "scene_label", "body_type", "coarse_body_family",
        "body_supervised", "color", "color_supervised",
    ]
    rows = []
    for class_id in range(5):
        image_path = tmp_path / f"frame_{class_id}.jpg"
        image = np.full((80, 100, 3), 30 + class_id * 35, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (80, 60), (200 - class_id * 10, 80, 50), -1)
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(image_path)
        rows.append({
            "split": "train", "source_dataset": "MY-VID-v2", "source_image": str(image_path),
            "source_frame_id": f"frame_{class_id}", "source_video_id": f"video_{class_id}",
            "export_augmentation": "original", "box_index": 0, "numeric_class_id": class_id,
            "x_center": 0.5, "y_center": 0.5, "width": 0.6, "height": 0.5,
            "scene_label": "unknown", "body_type": "unknown", "coarse_body_family": "unknown",
            "body_supervised": "false", "color": "unknown", "color_supervised": "false",
        })
    stage187 = tmp_path / "stage187.csv"
    write_csv(stage187, fields, rows)
    existing = tmp_path / "existing.csv"
    write_csv(existing, ["split", "stage177_dhash64", "body_type", "body_type_supervised", "image_path"], [])
    report = run(stage187, existing, semantics_path, tmp_path / "out.csv", tmp_path / "report.json")
    assert report["status"] == "pass_component_train_only_auxiliary_pending_quota_merge"
    assert report["counts"]["stage187_original_rows"] == 5
    assert report["counts"]["component_train_eligible_body"] == {
        "bus": 1, "heavy_truck": 1, "light_truck": 1
    }
    assert report["counts"]["confirmed_night_truth_rows"] == 0
    assert report["counts"]["color_truth_rows"] == 0
    assert report["gates"]["overall_training_authorized"] is False
