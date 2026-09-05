from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.scripts.normalize_stage74_joint_color_runtime_paths import (
    projection_digest,
    resolve_runtime_path,
)


class Stage74RuntimePathNormalizerTest(unittest.TestCase):
    def test_new_sources_resolve_from_their_pinned_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ua = root / "ua"
            bmd = root / "bmd"
            base = root / "base"
            for path in (ua, bmd, base):
                path.mkdir()
            ua_path, ua_mode = resolve_runtime_path(
                {"image_path": "crops/a.jpg", "stage74_joint_source": "ua"},
                base_parent=base,
                ua_root=ua,
                bmd_root=bmd,
            )
            bmd_path, bmd_mode = resolve_runtime_path(
                {"image_path": "crops/b.jpg", "stage74_joint_source": "bmd"},
                base_parent=base,
                ua_root=ua,
                bmd_root=bmd,
            )
            self.assertEqual(ua_path, (ua / "crops/a.jpg").resolve())
            self.assertEqual(bmd_path, (bmd / "crops/b.jpg").resolve())
            self.assertEqual(ua_mode, "ua_absolute_normalized")
            self.assertEqual(bmd_mode, "bmd_absolute_normalized")

    def test_base_relative_path_keeps_base_parent_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            base = root / "base"
            path, mode = resolve_runtime_path(
                {"image_path": "../images/a.jpg", "stage74_joint_source": ""},
                base_parent=base,
                ua_root=root / "ua",
                bmd_root=root / "bmd",
            )
            self.assertEqual(path, (root / "images/a.jpg").resolve())
            self.assertEqual(mode, "base_relative_preserved")

    def test_absolute_path_is_not_rebased(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            absolute = (root / "images/a.jpg").resolve()
            path, mode = resolve_runtime_path(
                {"image_path": str(absolute), "stage74_joint_source": "ua"},
                base_parent=root / "base",
                ua_root=root / "ua",
                bmd_root=root / "bmd",
            )
            self.assertEqual(path, absolute)
            self.assertEqual(mode, "already_absolute")

    def test_validation_projection_detects_label_or_path_changes(self) -> None:
        fields = ["image_path", "color"]
        rows = [{"image_path": "a.jpg", "color": "red", "extra": "x"}]
        same = [{"image_path": "a.jpg", "color": "red", "extra": "y"}]
        changed = [{"image_path": "a.jpg", "color": "blue", "extra": "x"}]
        self.assertEqual(projection_digest(rows, fields), projection_digest(same, fields))
        self.assertNotEqual(projection_digest(rows, fields), projection_digest(changed, fields))


if __name__ == "__main__":
    unittest.main()
