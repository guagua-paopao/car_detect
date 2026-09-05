from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_stage66_licensed_adverse_manifest.py"
SPEC = importlib.util.spec_from_file_location("stage66_manifest", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(**changes: str) -> dict[str, str]:
    value = {
        "image_path": "x.jpg", "split": "train", "sha256": "a", "source_license": "CC-BY-4.0",
        "license_train_eligible": "true", "body_type": "sedan", "color": "unknown",
        "body_type_supervised": "true", "color_supervised": "false", "review_status": "approved",
        "track_group": "train-a", "video_id": "", "camera_id": "", "adverse_supervised": "",
    }
    value.update(changes)
    return value


class Stage66LicensedAdverseManifestTest(unittest.TestCase):
    def test_filters_restricted_train_but_preserves_heldout(self) -> None:
        base = [
            row(),
            row(sha256="b", source_license="research-and-education-only; original-authors"),
            row(sha256="c", split="validation", source_license="research-and-education-only; original-authors", track_group="val-c"),
            row(sha256="d", split="test", source_license="", license_train_eligible="", track_group="test-d"),
        ]
        output, audit = MODULE.build(base, [])
        self.assertEqual([item["sha256"] for item in output], ["a", "c", "d"])
        self.assertTrue(audit["evaluation_membership_preserved"])

    def test_maps_official_coarse_truck_to_partial_v1_family(self) -> None:
        base = [row()]
        adverse = [row(
            sha256="z", adverse_supervised="true", body_type="truck", body_type_supervised="true",
            source_license="CC-BY-2.0-image / CC-BY-4.0-annotation", track_group="adverse-z",
        )]
        output, audit = MODULE.build(base, adverse)
        added = output[-1]
        self.assertEqual(added["body_type"], "unknown")
        self.assertEqual(added["body_type_supervised"], "false")
        self.assertEqual(added["coarse_body_family"], "truck")
        self.assertEqual(audit["appended"]["coarse_truck"], 1)

    def test_rejects_duplicate_adverse_sha(self) -> None:
        base = [row(sha256="same")]
        adverse = [row(sha256="same", adverse_supervised="true")]
        output, audit = MODULE.build(base, adverse)
        self.assertEqual(len(output), 1)
        self.assertEqual(audit["append_rejections"]["duplicate_sha256"], 1)


if __name__ == "__main__":
    unittest.main()
