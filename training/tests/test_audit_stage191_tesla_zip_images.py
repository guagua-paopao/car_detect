import zipfile
from pathlib import Path

from training.scripts.audit_stage191_tesla_zip_images import build_burst_groups, safe_image_members


def test_burst_grouping_requires_consecutive_name_and_identical_labels() -> None:
    rows = [
        {"image": "IMG_100.jpg", "color": "White", "lighting": "Dark", "model": "Y", "year_3": "", "year_s": "", "year_x": "", "year_y": "2020"},
        {"image": "IMG_101.jpg", "color": "White", "lighting": "Dark", "model": "Y", "year_3": "", "year_s": "", "year_x": "", "year_y": "2020"},
        {"image": "IMG_102.jpg", "color": "Blue", "lighting": "Dark", "model": "Y", "year_3": "", "year_s": "", "year_x": "", "year_y": "2020"},
        {"image": "IMG_104.jpg", "color": "Blue", "lighting": "Dark", "model": "Y", "year_3": "", "year_s": "", "year_x": "", "year_y": "2020"},
    ]
    groups = build_burst_groups(rows)
    assert groups[0] == groups[1]
    assert groups[1] != groups[2]
    assert groups[2] != groups[3]


def test_zip_member_audit_rejects_duplicate_basenames(tmp_path: Path) -> None:
    archive_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("one/a.jpg", b"one")
        archive.writestr("two/a.jpg", b"two")
    with zipfile.ZipFile(archive_path) as archive:
        try:
            safe_image_members(archive)
        except ValueError as error:
            assert "duplicate image basename" in str(error)
        else:
            raise AssertionError("duplicate basenames must fail closed")
