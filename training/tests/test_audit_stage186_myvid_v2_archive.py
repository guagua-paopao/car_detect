import zipfile
from pathlib import Path

import training.scripts.audit_stage186_myvid_v2_archive as module


def test_split_inventory_without_image_decode(tmp_path: Path) -> None:
    archive = tmp_path / "myvid.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("MYVID/data.yaml", "names: ['a', 'b', 'c', 'd', 'e', 'f']\n")
        for split in ("train", "valid", "test"):
            package.writestr(f"MYVID/{split}/images/example.jpg", b"not-decoded")
            package.writestr(f"MYVID/{split}/labels/example.txt", "0 0.5 0.5 0.5 0.5\n")
    previous_bytes, previous_md5 = module.EXPECTED_BYTES, module.EXPECTED_MD5
    try:
        module.EXPECTED_BYTES = archive.stat().st_size
        module.EXPECTED_MD5 = module.digest(archive, "md5")
        report = module.audit(archive)
    finally:
        module.EXPECTED_BYTES, module.EXPECTED_MD5 = previous_bytes, previous_md5
    assert report["status"] == "pass"
    assert report["archive"]["split_counts"]["train"]["images"] == 1
    assert report["scope"]["image_pixels_decoded"] == 0
    assert report["scope"]["test_images_opened"] == 0
