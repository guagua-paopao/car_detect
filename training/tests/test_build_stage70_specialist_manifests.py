from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_stage70_specialist_manifests.py"
FIELDS = [
    "image_path", "body_type", "color", "body_type_supervised", "color_supervised",
    "coarse_body_family", "split", "track_group", "video_id", "source_dataset",
    "sha256", "dhash64", "pseudo_label", "pseudo_label_confidence", "night",
]


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)


def row(image: Path, *, split: str, body: str = "unknown", color: str = "unknown",
        body_supervised: bool = False, color_supervised: bool = False,
        coarse: str = "", pseudo: bool = False, track: str = "t1") -> dict[str, str]:
    data = image.read_bytes()
    import hashlib
    return {
        "image_path": image.name, "body_type": body, "color": color,
        "body_type_supervised": str(body_supervised).lower(),
        "color_supervised": str(color_supervised).lower(), "coarse_body_family": coarse,
        "split": split, "track_group": track, "video_id": track,
        "source_dataset": "fixture", "sha256": hashlib.sha256(data).hexdigest(),
        "dhash64": hashlib.sha256(data).hexdigest()[:16],
        "pseudo_label": str(pseudo).lower(), "pseudo_label_confidence": "0.9" if pseudo else "",
        "night": "false",
    }


class Stage70SpecialistManifestTest(unittest.TestCase):
    def test_hamming_tree_exact_radius(self) -> None:
        tree = __import__("importlib.util").util.spec_from_file_location("stage70_builder", SCRIPT)
        assert tree and tree.loader
        module = __import__("importlib.util").util.module_from_spec(tree)
        tree.loader.exec_module(module)
        index = module.HammingBKTree()
        first = {"image_path": "first"}; second = {"image_path": "second"}
        index.add(int("0000000000000000", 16), first)
        index.add(int("ffffffffffffffff", 16), second)
        self.assertIs(index.find(int("0000000000000003", 16), 2), first)
        self.assertIsNone(index.find(int("0000000000000007", 16), 2))
        self.assertIs(index.find(int("fffffffffffffffe", 16), 1), second)

    def test_head_isolation_and_test_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"; images.mkdir()
            paths = []
            for index in range(7):
                path = images / f"{index}.jpg"; path.write_bytes(f"image-{index}".encode()); paths.append(path)
            stage67 = root / "stage67.csv"
            write_manifest(stage67, [
                row(paths[0], split="train", body="sedan", body_supervised=True, track="s67-body"),
                row(paths[1], split="validation", color="red", color_supervised=True, track="s67-color-val"),
                row(paths[2], split="test", body="suv", body_supervised=True, track="must-not-import"),
            ])
            ua = root / "ua.csv"
            write_manifest(ua, [
                row(paths[3], split="train", coarse="car", track="ua-car"),
                row(paths[4], split="validation", body="van", body_supervised=True, track="ua-van-val"),
            ])
            pseudo = root / "pseudo.csv"
            write_manifest(pseudo, [
                row(paths[5], split="train", color="blue", color_supervised=True, pseudo=True, track="ua-color"),
            ])
            import hashlib
            pseudo_report = root / "pseudo-report.json"
            pseudo_report.write_text(json.dumps({
                "status": "pass",
                "output_manifest_sha256": hashlib.sha256(pseudo.read_bytes()).hexdigest(),
                "policy": {
                    "train_split_only": True, "validation_or_test_used": False,
                    "frozen_video_used": False, "model_predictions_replace_proposals": False,
                },
            }), encoding="utf-8")
            unlabeled = root / "unlabeled.csv"
            write_manifest(unlabeled, [row(paths[6], split="train", track="raw")])
            output = root / "output"
            completed = subprocess.run([
                sys.executable, str(SCRIPT), "--stage67-manifest", str(stage67),
                "--stage67-root", str(images), "--ua-manifest", str(ua), "--ua-root", str(images),
                "--color-pseudo-manifest", str(pseudo), "--color-audit-report", str(pseudo_report),
                "--stage69-unlabeled-manifest", str(unlabeled),
                "--stage69-unlabeled-root", str(images), "--datasets-safety-root", str(root),
                "--output-root", str(output), "--near-duplicate-hamming", "0",
            ], capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report_text = (output / "stage70-specialist-manifests-report.json").read_text(encoding="utf-8")
            try:
                report = json.loads(report_text)
            except json.JSONDecodeError:
                self.fail(repr(report_text[:500]))
            self.assertEqual(report["status"], "pass")
            self.assertFalse(report["policy"]["test_rows_imported"])
            with (output / "attribute_manifest.stage70-body.csv").open(newline="", encoding="utf-8") as handle:
                body = list(csv.DictReader(handle))
            with (output / "attribute_manifest.stage70-color.csv").open(newline="", encoding="utf-8") as handle:
                color = list(csv.DictReader(handle))
            with (output / "attribute_manifest.stage70-unlabeled.csv").open(newline="", encoding="utf-8") as handle:
                raw = list(csv.DictReader(handle))
            self.assertTrue(all(item["color"] == "unknown" and item["color_supervised"] == "false" for item in body))
            self.assertTrue(all(item["body_type"] == "unknown" and not item["coarse_body_family"] for item in color))
            self.assertTrue(all(item["body_type"] == item["color"] == "unknown" for item in raw))
            self.assertFalse(any(item["split"] == "test" for item in body + color + raw))

    def test_failed_color_path_can_only_be_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"; images.mkdir()
            paths = []
            for index in range(4):
                path = images / f"{index}.jpg"; path.write_bytes(f"fallback-{index}".encode()); paths.append(path)
            stage67 = root / "stage67.csv"
            write_manifest(stage67, [
                row(paths[0], split="train", body="sedan", body_supervised=True, track="body"),
                row(paths[1], split="validation", color="red", color_supervised=True, track="color"),
            ])
            ua = root / "ua.csv"
            write_manifest(ua, [row(paths[2], split="train", coarse="car", track="ua")])
            unlabeled = root / "unlabeled.csv"
            write_manifest(unlabeled, [row(paths[3], split="train", track="raw")])
            output = root / "output"
            completed = subprocess.run([
                sys.executable, str(SCRIPT), "--stage67-manifest", str(stage67),
                "--stage67-root", str(images), "--ua-manifest", str(ua), "--ua-root", str(images),
                "--omit-color-pseudo", "--stage69-unlabeled-manifest", str(unlabeled),
                "--stage69-unlabeled-root", str(images), "--datasets-safety-root", str(root),
                "--output-root", str(output), "--near-duplicate-hamming", "0",
            ], capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads(
                (output / "stage70-specialist-manifests-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                report["policy"]["color_pseudo_mode"],
                "omitted_after_failed_audit_all_ua_colors_unknown",
            )
            self.assertTrue(report["policy"]["failed_color_pseudo_never_imported"])


if __name__ == "__main__":
    unittest.main()
