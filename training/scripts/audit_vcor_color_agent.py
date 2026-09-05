#!/usr/bin/env python3
"""Conservative, agent-only VCoR color audit for Stage 2.

The audit never edits validation/test labels and never invents a color.  A train
row keeps its source color only when the frozen attribute model agrees with the
source label across multiple deterministic image views with high confidence;
otherwise the color head is marked unsupervised (unknown-by-policy).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True, help="ONNX multitask attribute model")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--min-confidence", type=float, default=0.82)
    ap.add_argument("--min-agreement", type=float, default=0.80)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()

    import onnxruntime as ort

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    colors = list(labels["colors"])
    color_index = {name: i for i, name in enumerate(colors)}
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("empty manifest")
    fields = list(rows[0])
    for field in ("color_review_status", "review_method", "review_score", "review_model"):
        if field not in fields:
            fields.append(field)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(args.model), providers=providers)
    input_name = session.get_inputs()[0].name

    # Mild views target exposure/resize sensitivity while retaining the label.
    def views(image: Image.Image) -> list[np.ndarray]:
        base = ImageOps.fit(image.convert("RGB"), (args.input_size, args.input_size), method=Image.Resampling.BILINEAR)
        variants = [base, ImageOps.mirror(base), ImageEnhance.Brightness(base).enhance(0.75), ImageEnhance.Brightness(base).enhance(1.25), ImageEnhance.Contrast(base).enhance(0.80)]
        out = []
        for im in variants:
            a = np.asarray(im, dtype=np.float32) / 255.0
            out.append(np.transpose(a, (2, 0, 1)))
        return out

    audit_indices = [i for i, r in enumerate(rows) if r.get("split") == "train" and r.get("source_dataset") == "VCoR" and r.get("color") not in ("", "unknown")]
    accepted = rejected = 0
    scores: list[float] = []
    distribution = Counter()
    for start in range(0, len(audit_indices), args.batch):
        batch_indices = audit_indices[start:start + args.batch]
        tensors: list[np.ndarray] = []
        spans: list[tuple[int, int]] = []
        for idx in batch_indices:
            row = rows[idx]
            path = (args.manifest.parent / row["image_path"]).resolve()
            with Image.open(path) as im:
                vv = views(im)
            begin = len(tensors)
            tensors.extend(vv)
            spans.append((begin, len(tensors)))
        body_logits, color_logits = session.run(None, {input_name: np.asarray(tensors, dtype=np.float32)})
        probs = softmax(np.asarray(color_logits))
        for idx, (lo, hi) in zip(batch_indices, spans):
            row = rows[idx]
            source = row.get("color", "unknown")
            source_id = color_index.get(source, -1)
            pv = probs[lo:hi]
            pred = pv.argmax(axis=1)
            conf = pv.max(axis=1)
            agreement = float(np.mean(pred == source_id)) if source_id >= 0 else 0.0
            score = float(agreement * float(np.min(conf)))
            accept = source_id >= 0 and agreement >= args.min_agreement and float(np.min(conf)) >= args.min_confidence
            out = dict(row)
            out["review_method"] = "agent_ensemble_audit_v1"
            out["review_model"] = str(args.model)
            out["review_score"] = f"{score:.6f}"
            if accept:
                out["color_review_status"] = "agent_auto_approved"
                accepted += 1
                distribution[source] += 1
            else:
                # Fail closed: do not train on a disputed color.
                out["color"] = "unknown"
                out["color_supervised"] = "false"
                out["color_review_status"] = "agent_auto_rejected_unknown"
                rejected += 1
            rows[idx] = out
            scores.append(score)

    # All non-audited rows keep their original labels and receive explicit status.
    for row in rows:
        if not row.get("review_method"):
            row["review_method"] = row.get("review_method", "frozen_source_manifest")
        if not row.get("color_review_status"):
            row["color_review_status"] = "unknown_by_policy" if row.get("color") in ("", "unknown") else "approved_source_or_human"

    # Hashes and split invariants make the audit reproducible.
    before_eval = {r.get("source_frame_id"): (r.get("body_type"), r.get("color")) for r in rows if r.get("split") in ("validation", "test")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    after = list(csv.DictReader(args.output.open("r", encoding="utf-8-sig", newline="")))
    after_eval = {r.get("source_frame_id"): (r.get("body_type"), r.get("color")) for r in after if r.get("split") in ("validation", "test")}
    report = {
        "schema_version": "1.0",
        "status": "complete",
        "method": "agent_ensemble_audit_v1",
        "policy": {"min_confidence": args.min_confidence, "min_agreement": args.min_agreement, "disagreement": "color=unknown,color_supervised=false", "validation_test_modified": before_eval != after_eval},
        "input_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "output_manifest_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "model": str(args.model),
        "audited_rows": len(audit_indices),
        "accepted": accepted,
        "rejected_to_unknown": rejected,
        "accepted_color_counts": dict(sorted(distribution.items())),
        "score": {"min": min(scores) if scores else None, "mean": float(np.mean(scores)) if scores else None, "p10": float(np.quantile(scores, 0.1)) if scores else None},
        "evaluation_labels_modified": before_eval != after_eval,
        "split_counts": dict(sorted(Counter(r.get("split", "") for r in after).items())),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
