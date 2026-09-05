from __future__ import annotations

import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_stage70_uadetrac_color_pseudolabels.py"
SPEC = importlib.util.spec_from_file_location("stage70_color", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage70ColorProposalTest(unittest.TestCase):
    def test_unknown_stays_in_agreement_denominator(self) -> None:
        evidence = [
            {"label": "red"}, {"label": "red"}, {"label": "red"},
            {"label": "unknown"}, {"label": "unknown"},
        ]
        label, share, reason = MODULE.choose_track_proposal(evidence, 3, 0.8)
        self.assertEqual(label, "unknown")
        self.assertAlmostEqual(share, 0.6)
        self.assertEqual(reason, "foreground_track_conflict")

    def test_requires_multiframe_high_share(self) -> None:
        evidence = [
            {"label": "blue"}, {"label": "blue"}, {"label": "blue"},
            {"label": "blue"}, {"label": "unknown"},
        ]
        label, share, reason = MODULE.choose_track_proposal(evidence, 3, 0.8)
        self.assertEqual((label, reason), ("blue", "accepted"))
        self.assertAlmostEqual(share, 0.8)

    def test_insufficient_frames_fail_closed(self) -> None:
        label, share, reason = MODULE.choose_track_proposal(
            [{"label": "white"}, {"label": "white"}, {"label": "unknown"}], 3, 0.6
        )
        self.assertEqual((label, share, reason), ("unknown", 0.0, "insufficient_foreground_frames"))

    def test_color_contract_is_sha_pinned_and_checkpoint_ordered(self) -> None:
        colors = ["black", "white", "silver_gray", "unknown"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            path.write_text(json.dumps({"colors": colors}), encoding="utf-8")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded, actual = MODULE.load_color_contract(path, expected)
            self.assertEqual((loaded, actual), (colors, expected))
            MODULE.validate_checkpoint_color_contract("teacher", colors, loaded)
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                MODULE.load_color_contract(path, "0" * 64)
            with self.assertRaisesRegex(RuntimeError, "color label order mismatch"):
                MODULE.validate_checkpoint_color_contract(
                    "teacher", ["black", "white", "gray", "unknown"], loaded
                )


if __name__ == "__main__":
    unittest.main()
