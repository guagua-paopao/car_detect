from __future__ import annotations

import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_stage76_dvm_body_pretrain_manifest.py"
SPEC = importlib.util.spec_from_file_location("stage76_dvm_body", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DvmBodyManifestTests(unittest.TestCase):
    def test_advertisement_id_is_recovered_from_official_image_name(self) -> None:
        value = "root/Audi$$RS4 Saloon$$2006$$Black$$7_29$$12$$image_1.jpg"
        self.assertEqual(MODULE.advertisement_id(value), "7_29$$12")
        self.assertEqual(MODULE.advertisement_id("bad.jpg"), "")

    def test_mapping_is_conservative_and_contract_shaped(self) -> None:
        self.assertEqual(MODULE.BODY_MAPPING["Saloon"], "sedan")
        self.assertEqual(MODULE.BODY_MAPPING["SUV"], "suv")
        self.assertEqual(MODULE.BODY_MAPPING["MPV"], "mpv")
        self.assertEqual(MODULE.BODY_MAPPING["Panel Van"], "van")
        self.assertEqual(MODULE.BODY_MAPPING["Hatchback"], "other")
        self.assertNotIn("Manual", MODULE.BODY_MAPPING)

    def test_official_bodytype_table_detects_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "dvm.zip"
            with zipfile.ZipFile(archive, "w") as opened:
                opened.writestr(
                    MODULE.AD_TABLE,
                    "Adv_ID,Bodytype\n1_1$$1,SUV\n1_1$$2,Saloon\n",
                )
            self.assertEqual(
                MODULE.read_bodytypes(archive),
                {"1_1$$1": "SUV", "1_1$$2": "Saloon"},
            )
            with zipfile.ZipFile(archive, "w") as opened:
                opened.writestr(
                    MODULE.AD_TABLE,
                    "Adv_ID,Bodytype\n1_1$$1,SUV\n1_1$$1,Saloon\n",
                )
            with self.assertRaises(RuntimeError):
                MODULE.read_bodytypes(archive)


if __name__ == "__main__":
    unittest.main()
