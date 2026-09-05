#!/usr/bin/env python3
"""Compare attribute checkpoints on VFG-7 validation only; never read test images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


THRESHOLDS = [
    0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
    0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86,
    0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()


def select_threshold(sweep: dict, precision_gate: float = 0.93) -> dict:
    rows = [
        {
            "threshold": float(threshold),
            "precision": float(metrics["high_confidence_precision"]),
            "coverage": float(metrics["high_confidence_coverage"]),
            "selected": int(metrics["high_confidence_selected"]),
        }
        for threshold, metrics in sweep.items()
    ]
    passing = [row for row in rows if row["precision"] >= precision_gate and row["selected"] > 0]
    if passing:
        chosen = max(passing, key=lambda row: (row["coverage"], row["precision"], -row["threshold"]))
        return {**chosen, "precision_gate_pass": True}
    chosen = max(rows, key=lambda row: (row["precision"], row["coverage"], row["threshold"]))
    return {**chosen, "precision_gate_pass": False}


def run(command: list[str], log_path: Path) -> None:
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    log_path.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT\n" + result.stdout + "\nSTDERR\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}); see {log_path}")


def compact_metrics(value: dict) -> dict:
    keys = ("evaluated", "selected", "precision", "coverage", "effective_unknown_rate", "correct_over_all")
    return {key: value.get(key) for key in keys}


def compact_static_metrics(value: dict) -> dict:
    return {
        "evaluated": value.get("evaluated"),
        "selected": value.get("high_confidence_selected"),
        "precision": value.get("high_confidence_precision"),
        "coverage": value.get("high_confidence_coverage"),
        "accuracy_without_abstention": value.get("accuracy"),
        "macro_f1_without_abstention": value.get("macro_f1"),
    }


def compact_stability(value: dict) -> dict:
    keys = ("windows", "emitted_predictions", "weighted_stability_rate", "windows_stability_ge_0_95_rate", "label_switches_total", "abstention_rate")
    return {key: value.get(key) for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--model", action="append", required=True, help="NAME=CHECKPOINT")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    splits = {row.get("split") for row in manifest_rows}
    if splits - {"validation", "test"}:
        raise RuntimeError(f"unexpected manifest splits: {splits}")
    if any(str(row.get("training_eligible", "")).lower() != "false" for row in manifest_rows):
        raise RuntimeError("VFG manifest contains a training-eligible row")
    validation_rows = sum(row.get("split") == "validation" for row in manifest_rows)
    withheld_test_rows = sum(row.get("split") == "test" for row in manifest_rows)
    manifest_hash_before = sha256(args.manifest)

    models = []
    for spec in args.model:
        if "=" not in spec:
            raise ValueError(f"invalid --model value: {spec}")
        name, checkpoint = spec.split("=", 1)
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        models.append((name, path))

    summaries = []
    for name, checkpoint in models:
        slug = safe_name(name)
        static_path = args.output_dir / f"{slug}.static-validation.json"
        static_log = args.output_dir / f"{slug}.static-validation.log"
        command = [
            args.python, str(args.scripts_root / "evaluate_attribute_baseline.py"),
            "--manifest", str(args.manifest), "--checkpoint", str(checkpoint),
            "--labels", str(args.labels), "--output", str(static_path),
            "--split", "validation", "--batch-size", str(args.batch_size),
            "--workers", str(args.workers), "--device", "cuda",
            "--type-threshold", "0.75", "--color-threshold", "0.40",
            "--threshold-sweep", *[str(value) for value in THRESHOLDS],
        ]
        if not (args.reuse_existing and static_path.is_file()):
            run(command, static_log)
        static = json.loads(static_path.read_text(encoding="utf-8"))
        body_selection = select_threshold(static["threshold_sweep"]["body_type"])
        color_selection = select_threshold(static["threshold_sweep"]["color"])

        track_path = args.output_dir / f"{slug}.track-validation.json"
        track_log = args.output_dir / f"{slug}.track-validation.log"
        track_command = [
            args.python, str(args.scripts_root / "evaluate_attribute_track_fusion.py"),
            "--manifest", str(args.manifest), "--checkpoint", str(checkpoint),
            "--output", str(track_path), "--split", "validation",
            "--type-threshold", str(body_selection["threshold"]),
            "--color-threshold", str(color_selection["threshold"]),
            "--fusion-windows", "3", "5", "--batch-size", str(args.batch_size),
            "--minimum-window-frames", "3", "--workers", str(args.workers), "--device", "cuda",
        ]
        if not (args.reuse_existing and track_path.is_file()):
            run(track_command, track_log)
        track = json.loads(track_path.read_text(encoding="utf-8"))
        body5 = track["body_type"]["fusion"]["5"]
        color5 = track["color"]["fusion"]["5"]
        summaries.append({
            "model": name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "selected_thresholds": {"body_type": body_selection, "color": color_selection},
            "static_validation": {
                "body_type": compact_static_metrics(static["threshold_sweep"]["body_type"][f"{body_selection['threshold']:.4f}"]),
                "color": compact_static_metrics(static["threshold_sweep"]["color"][f"{color_selection['threshold']:.4f}"]),
            },
            "track_validation": {
                "body_single": compact_metrics(track["body_type"]["single_frame"]),
                "body_fusion5_frame": compact_metrics(body5["frame"]),
                "body_fusion5_final": compact_metrics(body5["final_window"]),
                "body_fusion5_stability": compact_stability(body5["stability"]),
                "color_single": compact_metrics(track["color"]["single_frame"]),
                "color_fusion5_frame": compact_metrics(color5["frame"]),
                "color_fusion5_final": compact_metrics(color5["final_window"]),
                "color_fusion5_stability": compact_stability(color5["stability"]),
                "color_fusion5_night_proxy": {
                    "stability": compact_stability(color5["stability_stratified"]["weather"].get("night", {})),
                    "single": compact_metrics(track["color"]["single_frame_stratified"]["weather"].get("night", {})),
                },
            },
            "artifacts": {
                "static_validation": str(static_path), "static_validation_sha256": sha256(static_path),
                "track_validation": str(track_path), "track_validation_sha256": sha256(track_path),
                "static_log": str(static_log), "track_log": str(track_log),
            },
        })

    baseline = summaries[0]
    baseline_body_coverage = baseline["track_validation"]["body_fusion5_final"]["coverage"] or 0.0
    baseline_color_unknown = baseline["track_validation"]["color_fusion5_stability"]["abstention_rate"] or 0.0
    for item in summaries:
        body_final = item["track_validation"]["body_fusion5_final"]
        color_final = item["track_validation"]["color_fusion5_final"]
        body_stability = item["track_validation"]["body_fusion5_stability"]["weighted_stability_rate"] or 0.0
        color_stability = item["track_validation"]["color_fusion5_stability"]["weighted_stability_rate"] or 0.0
        color_unknown = item["track_validation"]["color_fusion5_stability"]["abstention_rate"] or 0.0
        coverage_gain = ((body_final["coverage"] or 0.0) - baseline_body_coverage) * 100.0
        unknown_reduction = ((baseline_color_unknown - color_unknown) / baseline_color_unknown) if baseline_color_unknown > 0 else 0.0
        gates = {
            "static_body": item["selected_thresholds"]["body_type"]["precision_gate_pass"] and item["selected_thresholds"]["body_type"]["coverage"] >= 0.45,
            "static_color": item["selected_thresholds"]["color"]["precision_gate_pass"] and item["selected_thresholds"]["color"]["coverage"] >= 0.25,
            "track_body": (body_final["precision"] or 0.0) >= 0.93 and (body_final["coverage"] or 0.0) >= 0.45,
            "track_color": (color_final["precision"] or 0.0) >= 0.93 and (color_final["coverage"] or 0.0) >= 0.25,
            "track_stability": body_stability >= 0.95 and color_stability >= 0.95,
            "type_coverage_gain_15pp": item is baseline or coverage_gain >= 15.0,
            "color_unknown_reduction_20pct": item is baseline or unknown_reduction >= 0.20,
        }
        item["comparison_to_production"] = {
            "body_fusion5_final_coverage_gain_percentage_points": coverage_gain,
            "color_fusion5_unknown_relative_reduction": unknown_reduction,
        }
        item["validation_gates"] = gates
        item["validation_screen_status"] = "pass" if all(gates.values()) else "fail_closed"

    if sha256(args.manifest) != manifest_hash_before:
        raise RuntimeError("manifest changed during evaluation")
    report = {
        "schema_version": "vfg7-validation-comparison-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_validation_only",
        "policy": {
            "license": "CC BY-NC 4.0",
            "training_eligible": False,
            "split_used": "validation",
            "withheld_test_rows_not_inferred": withheld_test_rows,
            "test_used_for_threshold_selection": False,
            "frozen_video_used": False,
            "backend_gates_run": False,
            "deployment": "paused_by_user",
            "threshold_selection": "maximum validation coverage subject to precision >= 0.93",
            "lighting": "pixel-derived proxy, not source ground truth",
        },
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_hash_before,
        "validation_rows": validation_rows,
        "withheld_test_rows": withheld_test_rows,
        "models": summaries,
        "decision": "use validation failures and confusion evidence to plan the next separated-head candidate; do not run VFG test",
    }
    report_path = args.output_dir / "comparison-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "report_sha256": sha256(report_path),
        "status": report["status"],
        "models": [
            {"model": item["model"], "status": item["validation_screen_status"], "gates": item["validation_gates"]}
            for item in summaries
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
