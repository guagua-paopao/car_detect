from __future__ import annotations

import unittest

from build_stage136_mio_road_domain_axle_manifest import map_mio_row, merge_rows


def row(**updates: str) -> dict[str, str]:
    base = {
        "image_path": "/data/x.jpg",
        "split": "train",
        "body_type": "pickup",
        "body_type_supervised": "true",
        "color": "unknown",
        "color_supervised": "false",
        "source_dataset": "MIO-TCD-Classification-2017",
        "source_label": "pickup_truck",
        "source_license": "CC-BY-NC-SA-4.0",
        "review_status": "approved_research_only",
        "track_group": "mio-1",
        "sha256": "a" * 64,
        "dhash64": "0000000000000000",
        "sample_weight": "1.0",
    }
    base.update(updates)
    return base


class Stage136ManifestTests(unittest.TestCase):
    def test_exact_semantic_mapping_and_research_isolation(self) -> None:
        mapped, reason = map_mio_row(row())
        self.assertEqual(reason, "accepted")
        self.assertEqual(mapped["body_type"], "light_truck")
        self.assertEqual(mapped["research_only"], "true")
        self.assertEqual(mapped["deployment_eligible"], "false")
        heavy, reason = map_mio_row(
            row(body_type="heavy_truck", source_label="articulated_truck")
        )
        self.assertEqual(reason, "accepted")
        self.assertEqual(heavy["body_type"], "heavy_truck")
        self.assertEqual(heavy["sample_weight"], "1.280000")

    def test_rejects_coarse_or_wrong_license_rows(self) -> None:
        self.assertEqual(map_mio_row(row(source_label="single_unit_truck"))[1], "not_axle_eligible")
        self.assertEqual(map_mio_row(row(source_license="CC-BY-4.0"))[1], "license_mismatch")
        self.assertEqual(map_mio_row(row(split="validation"))[1], "non_train")

    def test_merge_deduplicates_against_base_and_candidate(self) -> None:
        base = [
            row(
                split="validation",
                body_type="light_truck",
                source_dataset="VFG",
                source_label="two_axle",
                source_license="CC-BY-4.0",
                review_status="approved",
                track_group="vfg-1",
                sha256="b" * 64,
                dhash64="ffffffffffffffff",
            )
        ]
        candidates = [
            row(),
            row(track_group="mio-2", sha256="c" * 64, dhash64="0000000000000000"),
            row(
                body_type="heavy_truck",
                source_label="articulated_truck",
                track_group="mio-3",
                sha256="d" * 64,
                dhash64="00ff00ff00ff00ff",
            ),
        ]
        output, summary = merge_rows(base, candidates, radius=0)
        self.assertEqual(len(output), 3)
        self.assertEqual(summary["accepted_class_counts"], {"heavy_truck": 1, "light_truck": 1})
        self.assertEqual(summary["counters"]["rejected_near_duplicate"], 1)

    def test_test_rows_fail_closed(self) -> None:
        with self.assertRaises(RuntimeError):
            merge_rows([], [row(split="test")])


if __name__ == "__main__":
    unittest.main()
