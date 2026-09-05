"""Select a leakage-controlled UVH-26 CCTV hard subset.

UVH-26 supplies real fixed-camera vehicle *type* labels (not colors).  This
script selects image frames with small boxes/high density and writes only a
download list plus a manifest plan.  It never reads the frozen evaluation
video and never creates pseudo color labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_TO_BODY = {
    "Hatchback": "other",
    "Sedan": "sedan",
    "SUV": "suv",
    "MUV": "mpv",
    "Bus": "bus",
    "Truck": "heavy_truck",
    "LCV": "light_truck",
    "Mini-bus": "bus",
    "Tempo-traveller": "bus",
    "Van": "van",
}


def stable_unit(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--file-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--min-dim", type=int, default=96)
    parser.add_argument("--max-area-ratio", type=float, default=0.025)
    args = parser.parse_args()
    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    index = json.loads(args.file_index.read_text(encoding="utf-8"))
    categories = {int(c["id"]): c["name"] for c in data["categories"]}
    file_by_name = {Path(x["path"]).name: x for x in index if str(x.get("path", "")).endswith(".png")}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        name = categories.get(int(ann["category_id"]))
        if name in SOURCE_TO_BODY:
            anns_by_image[int(ann["image_id"])].append(ann)
    candidates = []
    for image in data["images"]:
        image_id = int(image["id"])
        anns = anns_by_image.get(image_id, [])
        file_record = file_by_name.get(Path(image["file_name"]).name)
        if not anns or file_record is None:
            continue
        w, h = float(image["width"]), float(image["height"])
        small = 0
        for ann in anns:
            _, _, bw, bh = [float(v) for v in ann["bbox"]]
            if min(bw, bh) <= args.min_dim or (bw * bh) / (w * h) <= args.max_area_ratio:
                small += 1
        density = len(anns)
        if small == 0 and density < 6:
            continue
        score = 4.0 * small + 0.5 * density + 0.25 * len({SOURCE_TO_BODY[categories[int(a["category_id"])] ] for a in anns})
        candidates.append({"image": image, "annotations": anns, "file": file_record, "small": small, "density": density, "score": score})
    candidates.sort(key=lambda x: (-x["score"], stable_unit(str(x["image"]["file_name"]))))

    # Greedy class balancing: each selected frame contributes several crops;
    # cap class counts only after ensuring hard frames are represented.
    selected = []
    class_counts = Counter()
    for item in candidates:
        if len(selected) >= args.max_images:
            break
        labels = {SOURCE_TO_BODY[categories[int(a["category_id"])] ] for a in item["annotations"]}
        # Prefer frames that add an under-represented mapped body type.
        if len(selected) < min(args.max_images, 700) or any(class_counts[x] < 800 for x in labels):
            selected.append(item)
            class_counts.update(labels)
    if len(selected) < args.max_images:
        used = {x["image"]["file_name"] for x in selected}
        selected.extend(x for x in candidates if x["image"]["file_name"] not in used)
        selected = selected[: args.max_images]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "uvh26-subset-plan.json"
    download_path = args.output_dir / "download-list.tsv"
    manifest_path = args.output_dir / "uvh26-body-crops-plan.csv"
    selected_names = {x["image"]["file_name"] for x in selected}
    with download_path.open("w", encoding="utf-8", newline="") as handle:
        for item in selected:
            remote = item["file"]["path"]
            local = str(Path("images") / Path(remote).name)
            handle.write(f"{remote}\t{local}\t{item['file'].get('size', 0)}\n")
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_name", "source_image_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "source_class", "body_type", "small_target", "density", "split"])
        writer.writeheader()
        for item in selected:
            for ann in item["annotations"]:
                src = categories[int(ann["category_id"])]
                x, y, bw, bh = [float(v) for v in ann["bbox"]]
                writer.writerow({"file_name": str(Path("images") / Path(item["file"]["path"]).name), "source_image_id": item["image"]["id"], "bbox_x": x, "bbox_y": y, "bbox_w": bw, "bbox_h": bh, "source_class": src, "body_type": SOURCE_TO_BODY[src], "small_target": int(min(bw, bh) <= args.min_dim or (bw * bh) / (item["image"]["width"] * item["image"]["height"]) <= args.max_area_ratio), "density": item["density"], "split": "train"})
    report = {"schema_version": "uvh26-subset-v1", "source": "iisc-aim/UVH-26", "license": "CC-BY-4.0", "frozen_video_used": False, "candidate_images": len(candidates), "selected_images": len(selected), "selected_boxes": sum(len(x["annotations"]) for x in selected), "small_boxes": sum(x["small"] for x in selected), "class_counts": dict(class_counts), "download_list": str(download_path), "crop_plan": str(manifest_path), "image_index_sha256": hashlib.sha256(args.file_index.read_bytes()).hexdigest(), "annotation_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest()}
    plan_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
