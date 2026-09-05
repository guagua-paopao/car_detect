import json
from pathlib import Path
import tempfile
import unittest

import finalize_stage156_fullscale_body_training as finalizer


class Stage156TrainingFinalizerTests(unittest.TestCase):
    def test_audit_candidate_requires_test_not_run_and_expected_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("best.pt", "last.pt"):
                (root / name).write_bytes(b"weights")
            (root / "metrics.json").write_text(json.dumps({"body_type_accuracy": 0.5}), encoding="utf-8")
            (root / "test_metrics.json").write_text(json.dumps({"status": "not_run"}), encoding="utf-8")
            (root / "model_card.json").write_text(json.dumps({
                "input_size": 288, "resize_mode": "letterbox", "command": ["trainer", "--skip-test"],
                "metrics": {"test": {"status": "not_run"}},
            }), encoding="utf-8")
            audited = finalizer.audit_candidate(root, (288, "letterbox"))
            self.assertEqual(audited["test_status"], "not_run")

    def test_audit_candidate_rejects_test_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("best.pt", "last.pt"):
                (root / name).write_bytes(b"weights")
            (root / "metrics.json").write_text("{}", encoding="utf-8")
            (root / "test_metrics.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
            (root / "model_card.json").write_text(json.dumps({
                "input_size": 256, "resize_mode": "stretch", "command": ["trainer", "--skip-test"],
                "metrics": {"test": {"status": "not_run"}},
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "accessed a test split"):
                finalizer.audit_candidate(root, (256, "stretch"))


if __name__ == "__main__":
    unittest.main()
