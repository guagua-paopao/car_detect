from __future__ import annotations

import json
import unittest
from pathlib import Path


TEMPLATE = Path(__file__).resolve().parents[1] / "artifacts" / "attribute-domain-v2" / "stage71-student-matrix-template.json"


class Stage71StudentMatrixTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix = json.loads(TEMPLATE.read_text(encoding="utf-8"))

    def test_template_cannot_launch_before_teacher_validation(self) -> None:
        self.assertEqual(self.matrix["status"], "template_waiting_for_teacher_validation")
        self.assertEqual(self.matrix["evidence_mode"], "stage71_teacher_distillation")
        self.assertNotIn("teacher_validation_report", self.matrix["immutable_inputs"])
        self.assertNotIn("teacher_pair_id", self.matrix)

    def test_candidate_matrix_covers_sizes_and_foreground_color(self) -> None:
        candidates = self.matrix["candidates"]
        self.assertEqual(len(candidates), 5)
        body_sizes = {item["input_size"] for item in candidates if item["specialist"] == "body"}
        color_sizes = {item["input_size"] for item in candidates if item["specialist"] == "color"}
        self.assertEqual(body_sizes, {224, 256})
        self.assertEqual(color_sizes, {224, 256})
        self.assertTrue(any(item["architecture"] == "mobilenet_v3_large_foreground_dual" for item in candidates))

    def test_every_candidate_preserves_specialist_isolation(self) -> None:
        for item in self.matrix["candidates"]:
            if item["specialist"] == "body":
                self.assertGreater(item["body_loss_weight"], 0)
                self.assertEqual(item["color_loss_weight"], 0)
                self.assertGreater(item["distill_body_weight"], 0)
                self.assertEqual(item["distill_color_weight"], 0)
            else:
                self.assertGreater(item["color_loss_weight"], 0)
                self.assertEqual(item["body_loss_weight"], 0)
                self.assertGreater(item["distill_color_weight"], 0)
                self.assertEqual(item["distill_body_weight"], 0)
            self.assertGreater(item["distill_weight"], 0)


if __name__ == "__main__":
    unittest.main()
