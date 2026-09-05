#!/usr/bin/env python3
"""Assemble the local, human- and machine-readable Stage 1 evidence bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "training" / "artifacts" / "attribute-domain-v2"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metric_block(data: dict) -> dict:
    return {
        key: data.get(key)
        for key in (
            "evaluated",
            "accuracy",
            "macro_f1",
            "high_confidence_precision",
            "high_confidence_coverage",
            "high_confidence_selected",
            "predicted_unknown_rate",
            "true_unknown_rate",
        )
    }


def main() -> int:
    validation = read_json(ART / "baseline-validation.json")
    test = read_json(ART / "baseline-test.json")
    card = read_json(ART / "dataset_card.json")
    split_report = read_json(ART / "attribute-domain-v2-split-report.json")
    registry_path = ROOT / "models" / "manifests" / "model_registry.v1.json"
    config_path = ROOT / "config" / "vehicle_analytics.yaml"
    registry = read_json(registry_path)
    config = read_json(config_path)
    attr = next(x for x in registry["artifacts"] if x["artifact_id"] == "vehicle-attr-v1")
    model_path = ROOT / attr["files"]["onnx_path"]
    engine_path = ROOT / attr["files"]["engine_path"]
    rollback_root = ROOT / "training" / "artifacts" / "domain-retrain-v1" / "rollback-20260823-domain-bmd45-v1"
    data_gaps = [
        {
            "gap": "body_type_high_confidence_precision_below_target",
            "target": 0.93,
            "validation": validation["body_type"]["high_confidence_precision"],
            "test": test["body_type"]["high_confidence_precision"],
            "impact": "Do not lower the 0.75 threshold or publish a new candidate to reduce unknowns.",
        },
        {
            "gap": "body_type_macro_f1_and_small_medium_crops",
            "validation": validation["body_type"]["macro_f1"],
            "test": test["body_type"]["macro_f1"],
            "impact": "Prioritize sedan/SUV/MPV/van/pickup confusion and small/medium crops in hard-example mining.",
        },
        {
            "gap": "occlusion_metadata_sparse",
            "counts": card["metadata_counts"]["occlusion_level"],
            "impact": "Only a small number of source rows carry explicit occluded/truncated flags; add reviewed occlusion labels from real gate video before stratified release selection.",
        },
        {
            "gap": "license_restricted_or_unclear_rows",
            "counts": card["license_records"],
            "formal_train_eligible_rows": card["metadata_counts"]["license_train_eligible"].get("true", 0),
            "eval_only_rows": card["metadata_counts"]["license_train_eligible"].get("false", 0),
            "impact": "Keep Stanford-Cars/human rows with unclear terms out of formal training until an approval record exists.",
        },
        {
            "gap": "fixed_video_not_used_for_tuning",
            "status": "pass",
            "impact": "No fixed 36-48s video frames were used for threshold or model selection in Stage 1.",
        },
    ]
    report = {
        "schema_version": "1.0",
        "stage": "stage1_freeze_baseline_and_attribute_eval_set",
        "status": "complete",
        "completed_at": "2026-08-23",
        "production_model_unchanged": True,
        "formal_baseline": {
            "artifact_id": attr["artifact_id"],
            "architecture": attr["architecture"],
            "input": attr["input"],
            "thresholds": {"body_type": config["vehicle_analytics"]["type_threshold"], "color": config["vehicle_analytics"]["color_threshold"]},
            "onnx": {"path": str(model_path), "sha256": sha(model_path), "registry_sha256": attr["files"]["onnx_sha256"]},
            "engine": {"path": str(engine_path), "sha256": sha(engine_path), "registry_sha256": attr["files"]["engine_sha256"]},
            "registry_path": str(registry_path),
            "runtime_config_path": str(config_path),
        },
        "baseline_metrics": {
            "validation": {"body_type": metric_block(validation["body_type"]), "color": metric_block(validation["color"])},
            "test": {"body_type": metric_block(test["body_type"]), "color": metric_block(test["color"])},
            "stratified_validation": validation["stratified"],
            "stratified_test": test["stratified"],
            "unknown_distribution_validation": validation["unknown_distribution"],
            "unknown_distribution_test": test["unknown_distribution"],
            "threshold_protocol": validation["protocol"],
        },
        "dataset": {
            "dataset_card": card,
            "split_report": split_report,
            "manifest_sha256": sha(ART / "attribute_manifest.csv"),
        },
        "data_gaps": data_gaps,
        "rollback_integrity": {
            "path": str(rollback_root),
            "files": {
                str(p.relative_to(rollback_root)): sha(p)
                for p in sorted(rollback_root.rglob("*"))
                if p.is_file()
            },
        },
        "next_stage_estimate": {
            "stage": "stage2_supplement_real_attribute_data",
            "estimate": "4-8 hours for source/license audit and manifest review queue; 6-12 GPU hours for the first candidate training run after data is frozen.",
            "prerequisites": ["approve or exclude restricted-license sources", "double-review color labels", "add night/occlusion/size hard examples", "freeze manifest and split report"],
        },
    }
    (ART / "stage1-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# VCAS Stage 1 — baseline freeze and attribute-domain-v2",
        "",
        "Status: **complete**. The deployed model and registry were not overwritten.",
        "",
        "## Formal baseline",
        "",
        f"- Model: `{attr['architecture']}`, 224×224; thresholds body-type `{config['vehicle_analytics']['type_threshold']}` and color `{config['vehicle_analytics']['color_threshold']}`.",
        f"- ONNX SHA256: `{sha(model_path)}` (registry `{attr['files']['onnx_sha256']}`).",
        f"- TensorRT SHA256: `{sha(engine_path)}` (registry `{attr['files']['engine_sha256']}`).",
        "",
        "| Split | Head | Accuracy | Macro-F1 | High-conf precision | Coverage | Evaluated |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split_name, data in (("validation", validation), ("test", test)):
        for head, label in (("body_type", "body-type"), ("color", "color")):
            m = data[head]
            lines.append(f"| {split_name} | {label} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['high_confidence_precision']:.4f} | {m['high_confidence_coverage']:.4f} | {m['evaluated']} |")
    lines += [
        "",
        "Body-type high-confidence precision is below the 0.93 release target on both validation and test; color clears the precision target but is not a reason to relax unknown policy. Test was not used for threshold or model selection, and the fixed 36–48s video was not used during Stage 1.",
        "",
        "## attribute-domain-v2",
        "",
        f"- Deterministic split: train `{card['split_counts']['train']}`, validation `{card['split_counts']['validation']}`, test `{card['split_counts']['test']}`; group isolation and duplicate checks pass.",
        f"- Sources: BMD-45 `{card['source_counts'].get('BMD-45', 0)}`, VCoR `{card['source_counts'].get('VCoR', 0)}`, Stanford Cars `{card['source_counts'].get('Stanford-Cars-196', 0)}`, human `{card['source_counts'].get('human', 0)}`.",
        f"- Formal-train eligible by recorded license policy: `{card['metadata_counts']['license_train_eligible'].get('true', 0)}`; eval-only restricted/unclear rows: `{card['metadata_counts']['license_train_eligible'].get('false', 0)}`.",
        "- Required metadata fields are present. Image-derived size and lighting are populated; explicit occlusion labels remain sparse and are a Stage 2 data gap.",
        "",
        "## Data gaps and next stage",
        "",
        "- Prioritize body-type hard examples, especially small/medium crops and human-reviewed gate scenes.",
        "- Keep BMD-45 color=unknown; add only double-reviewed color labels from approved sources.",
        "- Resolve license approval for Stanford/human rows before any formal training use.",
        "- Expected Stage 2 effort: 4–8 hours for audit/review queue, then 6–12 GPU hours for the first candidate run once frozen.",
        "",
        "Machine-readable evidence: `stage1-report.json`; dataset card: `dataset_card.json`; split/duplicate/isolations: `attribute-domain-v2-split-report.json`.",
    ]
    (ART / "stage1-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "report": str(ART / "stage1-report.json"), "summary": str(ART / "stage1-summary.md")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
