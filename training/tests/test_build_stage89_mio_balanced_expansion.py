from __future__ import annotations

import importlib.util
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "build_stage89_mio_balanced_expansion.py"
SPEC = importlib.util.spec_from_file_location("stage89_mio", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage89MioBalancedExpansionTests(unittest.TestCase):
    def test_default_targets_cap_pickup_and_prioritize_heavy_classes(self) -> None:
        targets = MODULE.DEFAULT_ADDITIONS
        self.assertLessEqual(targets["pickup"], 2300)
        self.assertGreater(targets["heavy_truck"], targets["pickup"])
        self.assertEqual(sum(targets.values()), 11450)

    def test_target_contract_requires_all_and_only_mapped_body_types(self) -> None:
        self.assertEqual(MODULE.validate_targets(dict(MODULE.DEFAULT_ADDITIONS)), MODULE.DEFAULT_ADDITIONS)
        with self.assertRaises(ValueError):
            MODULE.validate_targets({"pickup": 100})

    def test_selection_rank_is_stable_and_seeded(self) -> None:
        first = MODULE.deterministic_rank("123", "bus", "seed-a")
        self.assertEqual(first, MODULE.deterministic_rank("123", "bus", "seed-a"))
        self.assertNotEqual(first, MODULE.deterministic_rank("123", "bus", "seed-b"))

    def test_train_member_name_includes_official_class_directory(self) -> None:
        self.assertEqual(MODULE.train_member_name("00123456", "bus"), "train/bus/00123456.jpg")

    def test_mapping_preserves_generic_single_unit_truck(self) -> None:
        self.assertEqual(MODULE.SOURCE_TO_BODY["single_unit_truck"], "truck")
        self.assertEqual(MODULE.SOURCE_TO_BODY["articulated_truck"], "heavy_truck")
        self.assertNotIn("car", MODULE.SOURCE_TO_BODY)

    def test_tar_members_can_be_indexed_once_for_constant_time_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "tiny.tar"
            payload = Path(temporary) / "1.jpg"
            payload.write_bytes(b"jpeg")
            with tarfile.open(archive, "w") as opened:
                opened.add(payload, arcname="train/1.jpg")
            with tarfile.open(archive, "r:") as opened:
                members = {
                    member.name: member
                    for member in opened.getmembers()
                    if member.isfile() and member.name.startswith("train/")
                }
                self.assertIn("train/1.jpg", members)


if __name__ == "__main__":
    unittest.main()
