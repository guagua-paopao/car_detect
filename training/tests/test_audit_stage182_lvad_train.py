from pathlib import Path

from PIL import Image

from training.scripts.audit_stage182_lvad_train import run


def test_numeric_classes_remain_quarantined(tmp_path: Path) -> None:
    root = tmp_path / "lvad"
    images = root / "train" / "images"
    labels = root / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for class_id in range(3):
        stem = f"frame1_00000{class_id}_jpg.rf.{class_id:032x}"
        Image.new("RGB", (64, 48), (20 + class_id * 30, 20, 20)).save(images / f"{stem}.jpg")
        (labels / f"{stem}.txt").write_text(f"{class_id} 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    report = run(
        root,
        tmp_path / "report.json",
        tmp_path / "manifest.csv",
        tmp_path / "contact.jpg",
    )
    assert report["status"] == "pass_source_audit_semantics_pending_with_exclusions"
    assert report["counts"]["valid_box_rows"] == 3
    assert report["gates"]["training_authorized"] is False
    assert report["scope"]["validation_images_opened"] == 0
    assert report["scope"]["frozen_video_used"] is False
