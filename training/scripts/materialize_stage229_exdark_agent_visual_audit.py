#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed-manifest", type=Path, required=True)
    parser.add_argument("--teacher-report", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    args.output_root.mkdir(parents=True)

    with args.reviewed_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    teacher = json.loads(args.teacher_report.read_text(encoding="utf-8"))
    decision_doc = json.loads(args.decisions.read_text(encoding="utf-8"))
    decisions = {item["source_image_name"]: item for item in decision_doc["decisions"]}
    teacher_accepted = [row for row in rows if row.get("color_teacher_consensus") == "accepted"]
    teacher_names = {row["source_image_name"] for row in teacher_accepted}
    if teacher_names != set(decisions) or len(teacher_accepted) != 9:
        raise ValueError("decision set must exactly cover all nine teacher-accepted rows")

    accepted = []
    rejected = []
    for row in teacher_accepted:
        item = decisions[row["source_image_name"]]
        if item["proposed_color"] != row["color"]:
            raise ValueError(f"decision color mismatch for {row['source_image_name']}")
        row["stage229_visual_decision"] = item["decision"]
        row["stage229_visual_reason"] = item["reason"]
        row["confirmed_real_night"] = "true"
        if item["decision"] == "accept":
            row["stage229_visual_accepted"] = "true"
            row["review_status"] = "approved_agent_visual_after_multiteacher"
            row["color_review_status"] = "agent_visual_visible_body_color_confirmed"
            row["formal_train_eligible"] = "true"
            accepted.append(row)
        else:
            row["stage229_visual_accepted"] = "false"
            row["color"] = "unknown"
            row["color_supervised"] = "false"
            row["review_status"] = "rejected_agent_visual"
            row["color_review_status"] = "agent_visual_ambiguous_or_illumination_contaminated"
            row["formal_train_eligible"] = "false"
            row["label_confidence"] = "unknown"
            rejected.append(row)
    for row in rows:
        row.setdefault("stage229_visual_decision", "not_reviewed")
        row.setdefault("stage229_visual_reason", "")
        row.setdefault("stage229_visual_accepted", "false")

    output_manifest = args.output_root / "stage229-exdark-agent-visual-reviewed.csv"
    additions = sorted({key for row in rows for key in row} - set(fields))
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + additions)
        writer.writeheader()
        writer.writerows(rows)

    ordered = teacher_accepted
    tile_w, tile_h, caption_h, columns = 260, 190, 54, 3
    sheet = Image.new("RGB", (tile_w * columns, (tile_h + caption_h) * math.ceil(len(ordered) / columns)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(ordered):
        with Image.open(row["image_path"]) as opened:
            image = ImageOps.contain(opened.convert("RGB"), (tile_w - 8, tile_h - 8))
        x = index % columns * tile_w
        y = index // columns * (tile_h + caption_h)
        sheet.paste(image, (x + (tile_w - image.width) // 2, y + (tile_h - image.height) // 2))
        decision = decisions[row["source_image_name"]]
        color = "green" if decision["decision"] == "accept" else "red"
        draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline=color, width=4)
        draw.text((x + 4, y + tile_h + 3), f"{decision['decision'].upper()} {row['color']} {row['source_image_name']}", fill=color)
        draw.text((x + 4, y + tile_h + 20), decision["reason"][:42], fill="black")
    sheet_path = args.output_root / "stage229-exdark-agent-visual-decisions.jpg"
    sheet.save(sheet_path, quality=94)

    accepted_counts = Counter(row["color"] for row in accepted)
    report = {
        "stage": "stage229-exdark-agent-visual-audit-r1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "complete_fail_closed_too_few_for_retrain",
        "teacher_status": teacher["status"],
        "teacher_audited_rows": teacher["audited_rows"],
        "teacher_accepted_rows": teacher["accepted_rows"],
        "agent_visual_reviewed_rows": len(teacher_accepted),
        "agent_visual_accepted_rows": len(accepted),
        "agent_visual_rejected_rows": len(rejected),
        "accepted_color_counts": dict(sorted(accepted_counts.items())),
        "accepted_confirmed_real_night_rows": len(accepted),
        "rejected_reasons": dict(Counter(decisions[row["source_image_name"]]["reason"] for row in rejected)),
        "training_authorized": False,
        "training_decision": "Do not retrain from three samples; retain them as a research-only supplement for a future larger batch.",
        "effective_license_policy": "research-only non-deployable pending resolution of CC0 repository metadata versus official README non-commercial restriction",
        "policy": {
            "visual_review_by_agent": True,
            "uncertain_colors_changed_to_unknown": True,
            "validation_pixels_opened": 0,
            "test_pixels_opened": 0,
            "frozen_video_used": False,
            "production_model_modified": False,
            "deployment_performed": False,
        },
        "inputs": {
            "reviewed_manifest": str(args.reviewed_manifest.resolve()),
            "reviewed_manifest_sha256": sha256(args.reviewed_manifest),
            "teacher_report": str(args.teacher_report.resolve()),
            "teacher_report_sha256": sha256(args.teacher_report),
            "decisions": str(args.decisions.resolve()),
            "decisions_sha256": sha256(args.decisions),
        },
        "outputs": {
            "manifest": str(output_manifest),
            "contact_sheet": str(sheet_path),
        },
    }
    report_path = args.output_root / "stage229-exdark-agent-visual-audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_root / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in (output_manifest, sheet_path, report_path, args.decisions):
            handle.write(f"{sha256(path)}  {path}\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
