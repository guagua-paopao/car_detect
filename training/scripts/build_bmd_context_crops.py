#!/usr/bin/env python3
"""Recover BMD boxes from official detection frames and build padded context crops."""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
from PIL import Image, ImageChops, ImageStat

def dhash64(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    bits = 0
    px = list(gray.getdata())
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(px[y * 9 + x] > px[y * 9 + x + 1])
    return bits

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def frame_candidates(source_frame: str, dataset_root: Path):
    token = source_frame.split(":", 1)[-1]
    seq, name = token.split("/", 1)
    stem = Path(name).stem
    for image_dir in (dataset_root / "detection" / "images" / "bmd_train", dataset_root / "detection" / "images" / "bmd_validation"):
        matches = list(image_dir.glob(f"bmd45_bmd-45-*_{seq}_{stem}.png"))
        if matches:
            frame = matches[0]
            label_dir = dataset_root / "detection" / "labels" / image_dir.name
            label = label_dir / (frame.stem + ".txt")
            if label.exists():
                yield frame, label

def boxes(label_path: Path, width: int, height: int):
    for line in label_path.read_text(encoding="utf-8").splitlines():
        vals = line.split()
        if len(vals) < 5:
            continue
        _, cx, cy, bw, bh = map(float, vals[:5])
        x1 = max(0, int(round((cx - bw / 2) * width))); y1 = max(0, int(round((cy - bh / 2) * height)))
        x2 = min(width, int(round((cx + bw / 2) * width))); y2 = min(height, int(round((cy + bh / 2) * height)))
        if x2 > x1 and y2 > y1:
            yield x1, y1, x2, y2

def context_box(box, width, height, margin: float):
    x1, y1, x2, y2 = box; bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw, nh = bw * (1 + 2 * margin), bh * (1 + 2 * margin)
    return (max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2)), min(width, int(cx + nw / 2)), min(height, int(cy + nh / 2)))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", type=Path, required=True); ap.add_argument("--dataset-root", type=Path, required=True); ap.add_argument("--output-manifest", type=Path, required=True); ap.add_argument("--crop-root", type=Path, required=True); ap.add_argument("--margin", type=float, default=0.35); ap.add_argument("--max-hamming", type=int, default=14); ap.add_argument("--max-mse", type=float, default=45.0); args = ap.parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as fh: rows = list(csv.DictReader(fh)); fields = list(rows[0])
    root = args.manifest.parent; args.crop_root.mkdir(parents=True, exist_ok=True)
    cache = {}; matched = 0; attempted = 0; distances = []; debug = []
    for idx, row in enumerate(rows):
        if row.get("source_dataset") != "BMD-45": continue
        is_small = str(row.get("small_target", "")).lower() in {"1", "true", "yes"} or row.get("vehicle_size") == "small"
        if not is_small: continue
        attempted += 1
        try: target_hash = int(row.get("dhash64", ""), 16)
        except ValueError: continue
        source = row.get("source_frame_id", "")
        if source not in cache:
            cache[source] = []
            for frame, label in frame_candidates(source, args.dataset_root):
                with Image.open(frame) as im:
                    cache[source] = [(frame, label, box) for box in boxes(label, *im.size)]
                break
        options = cache[source]
        if len(debug) < 8:
            debug.append({"source_frame_id": source, "options": len(options), "target_dhash": row.get("dhash64", "")})
        if not options: continue
        frame, label, _ = options[0]
        with Image.open(frame) as im:
            target_image = Image.open(root / row["image_path"]).convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
            candidates = []
            for box in [x[2] for x in options]:
                crop = im.crop(box)
                dist = hamming(dhash64(crop), target_hash)
                resized = crop.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
                mse = sum(ImageStat.Stat(ImageChops.difference(resized, target_image)).mean) / 3.0
                candidates.append((mse, dist, box))
            best_mse, best_dist, best_box = min(candidates, key=lambda x: x[0])
            if len(debug) <= 8:
                debug[-1]["best_hamming"] = best_dist
                debug[-1]["best_mse"] = best_mse
            if best_mse > args.max_mse and best_dist > args.max_hamming: continue
            out_box = context_box(best_box, im.width, im.height, args.margin)
            out_dir = args.crop_root / str(row.get("split", "unknown")); out_dir.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha1((row.get("image_path", "") + str(best_box)).encode()).hexdigest()[:20]
            out_path = out_dir / f"bmd_context_{digest}.jpg"
            im.crop(out_box).convert("RGB").save(out_path, quality=95)
        row["image_path"] = str(out_path.relative_to(root)).replace("\\", "/")
        row["context_crop"] = "true"; row["context_margin"] = str(args.margin); row["context_match_hamming"] = str(best_dist)
        matched += 1; distances.append(best_dist)
    for row in rows:
        row.setdefault("context_crop", "false"); row.setdefault("context_margin", ""); row.setdefault("context_match_hamming", "")
    out_fields = fields + [x for x in ("context_crop", "context_margin", "context_match_hamming") if x not in fields]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields); writer.writeheader(); writer.writerows(rows)
    report = {"schema_version": "bmd-context-crops-v1", "input_manifest": str(args.manifest), "output_manifest": str(args.output_manifest), "attempted_small_bmd": attempted, "matched_context_crops": matched, "match_rate": matched / attempted if attempted else 0.0, "max_hamming": args.max_hamming, "max_mse": args.max_mse, "margin": args.margin, "mean_match_hamming": sum(distances) / len(distances) if distances else None, "debug_first_rows": debug, "frozen_video_used": False}
    report_path = args.output_manifest.with_suffix(".report.json"); report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
