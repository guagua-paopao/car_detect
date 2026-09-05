#!/usr/bin/env python3
"""Rank all unknown train scenes for agent visual review without auto-labeling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from audit_stage237_clip_unknown_night import PROMPTS, sha256, truthy  # noqa: E402


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    result = {}
    for label, fraction in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)):
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
        result[label] = ordered[index]
    result["max"] = ordered[-1]
    return result


def make_contact_sheets(rows: list[dict[str, str]], output_dir: Path, per_page: int = 50) -> list[Path]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite contact sheets: {output_dir}")
    output_dir.mkdir(parents=True)
    font = ImageFont.load_default()
    tile_width, tile_height, caption_height, columns = 240, 170, 50, 5
    outputs = []
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_dataset"]].append(row)
    for source, candidates in sorted(grouped.items()):
        safe_source = "".join(ch if ch.isalnum() else "_" for ch in source).strip("_") or "unknown"
        candidates = sorted(candidates, key=lambda row: (-float(row["adverse_score"]), row["image_path"]))
        for page_index in range(0, len(candidates), per_page):
            page = candidates[page_index : page_index + per_page]
            row_count = (len(page) + columns - 1) // columns
            sheet = Image.new("RGB", (columns * tile_width, row_count * (tile_height + caption_height)), "white")
            draw = ImageDraw.Draw(sheet)
            for tile_index, row in enumerate(page):
                path = Path(row["image_path"])
                with Image.open(path) as opened:
                    image = ImageOps.contain(opened.convert("RGB"), (tile_width, tile_height))
                x = (tile_index % columns) * tile_width
                y = (tile_index // columns) * (tile_height + caption_height)
                sheet.paste(image, (x + (tile_width - image.width) // 2, y + (tile_height - image.height) // 2))
                caption = (
                    f"#{row['source_rank']} score={float(row['adverse_score']):.3f} "
                    f"{row.get('color', 'unknown')}\n{path.name[:34]}"
                )
                draw.multiline_text((x + 3, y + tile_height + 2), caption, fill="black", font=font)
            output = output_dir / f"{safe_source}-page-{page_index // per_page + 1:02d}.jpg"
            sheet.save(output, quality=92)
            outputs.append(output)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-manifest", type=Path, required=True)
    parser.add_argument("--stage237-report", type=Path, required=True)
    parser.add_argument("--model", default="ViT-B-32-quickgelu")
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-ranking", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-contact-sheets", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=100)
    parser.add_argument("--per-source-color", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_ranking.exists() or args.output_report.exists() or args.output_contact_sheets.exists():
        raise FileExistsError("refusing to overwrite Stage238 evidence")
    if sha256(args.weight) != "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af":
        raise RuntimeError("OpenAI CLIP weight SHA256 mismatch")
    stage237 = json.loads(args.stage237_report.read_text(encoding="utf-8"))
    if stage237.get("status") != "complete_fail_closed_semantic_precheck":
        raise RuntimeError("Stage238 requires the preserved Stage237 fail-closed report")

    import open_clip
    import torch
    from torch.utils.data import DataLoader, Dataset

    with args.active_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        active_rows = list(csv.DictReader(handle))
    unknown_rows = [
        row for row in active_rows
        if row.get("split") == "train"
        and truthy(row.get("color_supervised"))
        and truthy(row.get("stage177_effective_representative"))
        and row.get("stage177_scene_label") == "unknown"
        and not row.get("stage177_read_error")
    ]

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=str(args.weight), cache_dir=str(args.cache_dir), weights_only=False
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    device = torch.device(args.device)
    model.to(device).eval()
    class_names = list(PROMPTS)
    with torch.no_grad():
        prototypes = []
        for name in class_names:
            tokens = tokenizer(PROMPTS[name]).to(device)
            features = model.encode_text(tokens, normalize=True)
            prototype = features.mean(dim=0)
            prototypes.append(prototype / prototype.norm())
        text_features = torch.stack(prototypes)

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(unknown_rows)

        def __getitem__(self, index):
            path = Path(unknown_rows[index]["image_path"])
            try:
                with Image.open(path) as opened:
                    image = preprocess(opened.convert("RGB"))
                readable = True
            except Exception:
                image = torch.zeros(3, 224, 224)
                readable = False
            return image, index, readable

    results: list[dict[str, object] | None] = [None] * len(unknown_rows)
    loader = DataLoader(
        Dataset(), batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    with torch.no_grad():
        for images, indexes, readable in loader:
            image_features = model.encode_image(images.to(device, non_blocking=True), normalize=True)
            probabilities = (100.0 * image_features @ text_features.T).softmax(dim=1).cpu()
            for local, index in enumerate(indexes.tolist()):
                probs = {name: float(probabilities[local, class_index]) for class_index, name in enumerate(class_names)}
                results[index] = {
                    "readable": bool(readable[local]),
                    "adverse_score": probs["night"] + probs["low_light"],
                    "predicted_class": max(probs, key=probs.get),
                    "probabilities": probs,
                }

    source_scores: dict[str, list[float]] = defaultdict(list)
    source_read_errors = Counter()
    ranked_by_source: dict[str, list[tuple[float, int, dict[str, object]]]] = defaultdict(list)
    for index, (row, result) in enumerate(zip(unknown_rows, results)):
        source = row.get("source_dataset", "unknown") or "unknown"
        if result is None or not result["readable"]:
            source_read_errors[source] += 1
            continue
        score = float(result["adverse_score"])
        source_scores[source].append(score)
        ranked_by_source[source].append((score, index, result))

    selected = []
    selection_counts = Counter()
    for source, candidates in sorted(ranked_by_source.items()):
        candidates.sort(key=lambda item: (-item[0], unknown_rows[item[1]]["image_path"]))
        seen_groups = set()
        color_counts = Counter()
        source_rank = 0
        for score, index, result in candidates:
            row = unknown_rows[index]
            group = row.get("stage177_scene_group_key") or row.get("stage177_group_key") or row["image_path"]
            color = row.get("color", "unknown") or "unknown"
            if group in seen_groups or color_counts[color] >= args.per_source_color:
                continue
            seen_groups.add(group)
            color_counts[color] += 1
            source_rank += 1
            probs = dict(result["probabilities"])
            output = {
                "source_rank": str(source_rank),
                "image_path": row["image_path"],
                "source_dataset": source,
                "color": color,
                "stage177_scene_group_key": group,
                "stage177_group_key": row.get("stage177_group_key", ""),
                "track_key": row.get("track_key", ""),
                "camera_id": row.get("camera_id", ""),
                "video_id": row.get("video_id", ""),
                "vehicle_size": row.get("vehicle_size", ""),
                "adverse_score": f"{score:.8f}",
                "predicted_class": result["predicted_class"],
                **{f"prob_{key}": f"{float(value):.8f}" for key, value in probs.items()},
                "scene_proposal": "ranked_for_agent_visual_review_not_truth",
                "training_eligible": "false",
            }
            selected.append(output)
            selection_counts[source] += 1
            if source_rank >= args.per_source:
                break

    fields = list(selected[0]) if selected else ["image_path", "source_dataset", "adverse_score"]
    args.output_ranking.parent.mkdir(parents=True, exist_ok=True)
    with args.output_ranking.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    sheets = make_contact_sheets(selected, args.output_contact_sheets)

    report = {
        "schema_version": "stage238-clip-rank-unknown-night-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_ranked_candidates_pending_agent_visual_audit",
        "stage237_fail_closed_preserved": True,
        "stage237_calibration_auroc": stage237.get("calibration_auroc"),
        "automatic_scene_labeling_authorized": False,
        "unknown_effective_rows_scanned": len(unknown_rows),
        "read_errors": sum(source_read_errors.values()),
        "read_errors_by_source": dict(sorted(source_read_errors.items())),
        "source_score_quantiles": {source: quantiles(scores) for source, scores in sorted(source_scores.items())},
        "ranked_visual_candidates": len(selected),
        "ranked_visual_candidate_source_counts": dict(sorted(selection_counts.items())),
        "selection": {
            "maximum_per_source": args.per_source,
            "maximum_per_source_color": args.per_source_color,
            "one_per_scene_group": True,
            "ranking_only_no_threshold": True,
        },
        "model": args.model,
        "weight": str(args.weight.resolve()),
        "weight_sha256": sha256(args.weight),
        "inputs": {
            "active_manifest": str(args.active_manifest.resolve()),
            "active_manifest_sha256": sha256(args.active_manifest),
            "stage237_report": str(args.stage237_report.resolve()),
            "stage237_report_sha256": sha256(args.stage237_report),
        },
        "outputs": {
            "ranking": str(args.output_ranking.resolve()),
            "ranking_sha256": sha256(args.output_ranking),
            "contact_sheets": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in sheets],
        },
        "policy": {
            "rankings_are_not_scene_truth": True,
            "no_color_or_scene_labels_modified": True,
            "agent_visual_review_required": True,
            "train_pixels_only": True,
            "validation_or_test_pixels_opened": 0,
            "frozen_video_used": False,
            "training_started": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
    }
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
