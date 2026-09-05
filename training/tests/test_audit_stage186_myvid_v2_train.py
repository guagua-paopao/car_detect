from pathlib import Path

from PIL import Image

from training.scripts.audit_stage186_myvid_v2_train import run


def test_train_only_six_numeric_classes_stay_quarantined(tmp_path: Path) -> None:
    root = tmp_path / "myvid"
    images, labels = root / "train" / "images", root / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (root / "data.yaml").write_text(
        "names: ['car', 'van', 'medium', 'heavy', 'bus', 'motorcycle']\n",
        encoding="utf-8",
    )
    for class_id in range(6):
        stem = f"scene1_{class_id:06d}_jpg.rf.{class_id:032x}"
        Image.new("RGB", (96, 64), (25 + class_id * 25, 30, 35)).save(images / f"{stem}.jpg")
        (labels / f"{stem}.txt").write_text(f"{class_id} 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    report = run(root, tmp_path / "report.json", tmp_path / "manifest.csv", tmp_path / "contact.jpg")
    assert report["status"] == "pass_semantics_and_scene_truth_pending_with_exclusions"
    assert report["counts"]["valid_box_rows"] == 6
    assert report["counts"]["confirmed_night_frames"] == 0
    assert report["gates"]["training_authorized"] is False
    assert report["scope"]["test_images_opened"] == 0


def test_segmentation_polygons_and_export_variants_share_source_group(tmp_path: Path) -> None:
    root = tmp_path / "myvid"
    images, labels = root / "train" / "images", root / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (root / "data.yaml").write_text(
        "names: ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6']\n",
        encoding="utf-8",
    )
    polygon = "0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"
    for class_id in range(5):
        for suffix in ("", "_flip_h", "_flip_v", "_gray"):
            stem = f"frame_000000_{class_id:05d}{suffix}"
            Image.new("RGB", (96, 64), (25 + class_id * 25, 30, 35)).save(images / f"{stem}.jpg")
            (labels / f"{stem}.txt").write_text(f"{class_id} {polygon}\n", encoding="utf-8")
    report = run(root, tmp_path / "report.json", tmp_path / "manifest.csv", tmp_path / "contact.jpg")
    assert report["status"] == "pass_semantics_and_scene_truth_pending_with_exclusions"
    assert report["counts"]["annotation_formats"] == {"yolo_segmentation_polygon": 20}
    assert report["counts"]["unique_source_frames_before_cross_split_audit"] == 5
    assert report["counts"]["export_augmentation_images"] == {
        "flip_h": 5,
        "flip_v": 5,
        "gray": 5,
        "original": 5,
    }
    assert report["gates"]["all_five_in_scope_vehicle_classes_present_in_train"] is True
    assert report["gates"]["motorcycle_class_present_in_train"] is False
