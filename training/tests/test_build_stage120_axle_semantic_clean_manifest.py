import sys
from pathlib import Path
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_stage120_axle_semantic_clean_manifest import build, semantic_role


def row(**updates):
    base = {
        "split": "train",
        "source_dataset": "InaTRC-v1",
        "source_label": "two_axle_truck",
        "body_type": "light_truck",
        "source_license": "Apache-2.0",
        "research_only": "false",
        "sample_weight": "9.0",
        "annotation_source": "",
        "review_model": "",
    }
    base.update(updates)
    return base


class Stage120SemanticCleanTests(unittest.TestCase):
    def test_only_explicit_axle_or_lcv_contract_is_accepted(self):
        self.assertEqual(semantic_role(row())[0], "inatrc_light")
        self.assertEqual(
            semantic_role(row(source_label="three_axle_truck", body_type="heavy_truck"))[0],
            "inatrc_heavy",
        )
        self.assertEqual(
            semantic_role(
                row(
                    source_dataset="BMD-45-RAW",
                    source_label="",
                    body_type="light_truck",
                    annotation_source="BMD-45 COCO category mapping",
                    review_model="source_category_mapping_v23",
                )
            )[0],
            "bmd_lcv_light",
        )
        self.assertEqual(
            semantic_role(row(source_dataset="BMD-45-RAW", source_label="", body_type="heavy_truck"))[1],
            "generic_bmd_truck_not_axle_verified",
        )

    def test_build_preserves_validation_and_reweights_semantic_rows(self):
        rows = [
            row(split="validation", source_dataset="VFG", source_label="", body_type="light_truck"),
            row(),
            row(source_label="four_axle_truck", body_type="heavy_truck"),
            row(
                source_dataset="BMD-45-RAW",
                source_label="",
                body_type="light_truck",
                annotation_source="BMD-45 COCO category mapping",
                review_model="source_category_mapping_v23",
            ),
            row(source_dataset="BMD-45-RAW", source_label="", body_type="heavy_truck"),
        ]
        output, summary = build(rows)
        self.assertEqual(summary["validation_rows_preserved"], 1)
        self.assertEqual(summary["train_counts"], {"heavy_truck": 1, "light_truck": 2})
        train = [item for item in output if item["split"] == "train"]
        self.assertEqual({item["stage120_semantic_role"] for item in train}, {"inatrc_light", "inatrc_heavy", "bmd_lcv_light"})
        self.assertTrue(all(item["stage120_generic_truck_supervision"] == "false" for item in train))


if __name__ == "__main__":
    unittest.main()
