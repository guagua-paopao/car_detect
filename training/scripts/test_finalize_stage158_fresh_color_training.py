import json
import tempfile
import unittest
from pathlib import Path

from finalize_stage158_fresh_color_training import audit_candidate


class Stage158FinalizerTests(unittest.TestCase):
    def make_candidate(self, root: Path, command):
        root.mkdir()
        for name in ("best.pt", "last.pt"):
            (root / name).write_bytes(b"checkpoint")
        (root / "metrics.json").write_text("{}", encoding="utf-8")
        (root / "test_metrics.json").write_text(json.dumps({"status": "not_run"}), encoding="utf-8")
        card = {
            "input_size": 256,
            "resize_mode": "stretch",
            "selection_head": "color",
            "command": command,
            "metrics": {"test": {"status": "not_run"}},
        }
        (root / "model_card.json").write_text(json.dumps(card), encoding="utf-8")

    def test_accepts_fresh_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate"
            self.make_candidate(path, ["python", "train_attribute.py", "--skip-test"])
            result = audit_candidate(path, (256, "stretch"))
            self.assertEqual(result["test_status"], "not_run")

    def test_rejects_attribute_checkpoint_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate"
            self.make_candidate(path, ["python", "train_attribute.py", "--skip-test", "--init-checkpoint", "prior.pt"])
            with self.assertRaisesRegex(RuntimeError, "fresh ImageNet"):
                audit_candidate(path, (256, "stretch"))


if __name__ == "__main__":
    unittest.main()
