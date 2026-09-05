#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
SOURCE_ROOT="$BASE/sources/vehicle-rear-stage149"
METADATA="$SOURCE_ROOT/extracted.partial/data/dataset_3.json"
OUTPUT="$SOURCE_ROOT/selected-r2"
PARTIAL="$SOURCE_ROOT/selected-r2.partial"
PLAN="$SOURCE_ROOT/stage149-selection-plan-r2.json"
MANIFEST="$SOURCE_ROOT/stage149-selection-plan-r2.csv"
WANTED="$SOURCE_ROOT/stage149-train-validation-members-r2.txt"
STATE="$BASE/runs/attributes/ATTR-STAGE149-VEHICLE-REAR-SELECTIVE-EXTRACT-R2.state.json"
LOG="$BASE/runs/attributes/ATTR-STAGE149-VEHICLE-REAR-SELECTIVE-EXTRACT-R2.log"
SESSION=VCAS-DL-VEHICLEREAR149-R2
DATA_URL=https://www.inf.ufpr.br/vri/databases/vehicle-reid/data.tgz

if [[ "${1:-}" != "--worker" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "$SESSION already running"
    exit 0
  fi
  tmux new-session -d -s "$SESSION" "bash '$0' --worker >>'$LOG' 2>&1"
  echo "started tmux:$SESSION"
  exit 0
fi

[[ -f "$METADATA" ]] || { echo "missing metadata: $METADATA" >&2; exit 2; }
[[ ! -e "$PARTIAL" ]] || { echo "refusing to reuse partial output: $PARTIAL" >&2; exit 2; }
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite output: $OUTPUT" >&2; exit 2; }
mkdir -p "$SOURCE_ROOT" "$(dirname "$STATE")"

/root/miniconda3/bin/python - "$METADATA" "$PLAN" "$MANIFEST" "$WANTED" "$OUTPUT" <<'PY'
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import sys

metadata_path = Path(sys.argv[1])
plan_path = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
wanted_path = Path(sys.argv[4])
output_root = Path(sys.argv[5])

def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def split_for(raw_id: str) -> str:
    value = int(digest("vehicle-rear-stage149-split:" + raw_id)[:16], 16) / float(16**16)
    if value < 0.75:
        return "train"
    if value < 0.90:
        return "validation"
    return "future_holdout"

def map_color(value: str):
    value = value.strip().lower()
    exact = {
        "black": "black", "white": "white", "silver": "silver",
        "grey": "gray", "gray": "gray", "red": "red", "blue": "blue",
        "green": "green", "brown": "brown", "yellow": "yellow",
        "purple": "other", "pink": "other",
    }
    if value in exact:
        return exact[value], "exact_official_registration_color"
    if value in {"orange", "golden"}:
        return "yellow_orange", "partial_official_registration_color"
    if value == "beige":
        return "brown_beige", "partial_official_registration_color"
    return "unknown", "unmapped_fail_closed"

def choose_three(paths):
    paths = sorted(set(paths), key=lambda p: (Path(p).parts, int(Path(p).stem)))
    if len(paths) <= 3:
        return paths
    return [paths[0], paths[len(paths) // 2], paths[-1]]

data = json.loads(metadata_path.read_text(encoding="utf-8"))
vehicles = {}
for set_name, pairs in data.items():
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) < 7:
            continue
        for paths, meta in ((pair[1], pair[5]), (pair[3], pair[6])):
            if not paths or not isinstance(meta, dict):
                continue
            raw_id = Path(paths[0]).parent.name.lower()
            group_id = digest("vehicle-rear-stage149-group:" + raw_id)
            rec = vehicles.setdefault(group_id, {
                "raw_id": raw_id, "colors": set(), "brands": set(),
                "models": set(), "years": set(), "paths": set(), "sets": set(),
            })
            rec["colors"].add(str(meta.get("color", "")).strip().lower())
            rec["brands"].add(str(meta.get("brand", "")).strip().lower())
            rec["models"].add(str(meta.get("model", "")).strip().lower())
            rec["years"].add(str(meta.get("year", "")).strip().lower())
            rec["paths"].update(paths)
            rec["sets"].add(set_name)

rows = []
wanted = set()
split_groups = Counter()
split_rows = Counter()
color_groups = Counter()
conflicts = []
for group_id, rec in sorted(vehicles.items()):
    if len(rec["colors"]) != 1:
        conflicts.append(group_id)
        continue
    split = split_for(rec["raw_id"])
    source_color = next(iter(rec["colors"]))
    color, supervision = map_color(source_color)
    split_groups[split] += 1
    color_groups[(split, color)] += 1
    by_camera = {}
    for member in rec["paths"]:
        parts = Path(member).parts
        camera = next((p for p in parts if p.startswith("Camera")), "unknown")
        by_camera.setdefault(camera, []).append(member)
    selected = []
    for camera, members in sorted(by_camera.items()):
        selected.extend(choose_three(members))
    for member in selected:
        parts = Path(member).parts
        camera = next((p for p in parts if p.startswith("Camera")), "unknown")
        source_set = next((p for p in parts if p.startswith("Set")), "unknown")
        row = {
            "tar_member": member,
            "image_path": str(output_root / member),
            "body_type": "unknown",
            "color": color,
            "color_supervision": supervision,
            "source_color": source_color,
            "camera_id": camera,
            "video_id": source_set,
            "track_group": group_id,
            "split": split,
            "source_dataset": "Vehicle-Rear",
            "source_license": "Apache-2.0 repository scope; dataset scope audit pending",
            "formal_train_eligible": "false",
            "research_only": "true",
            "future_holdout_sealed": "true" if split == "future_holdout" else "false",
        }
        rows.append(row)
        split_rows[split] += 1
        if split != "future_holdout":
            wanted.add(member)

fields = list(rows[0]) if rows else []
with manifest_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
wanted_path.write_text("\n".join(sorted(wanted)) + "\n", encoding="utf-8")
plan = {
    "schema_version": "stage149-vehicle-rear-selection-plan-v2",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": "selection_plan_complete_images_not_yet_audited",
    "metadata": str(metadata_path),
    "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    "unique_vehicle_groups": len(vehicles),
    "conflicting_color_groups_rejected": len(conflicts),
    "split_groups": dict(split_groups),
    "split_rows": dict(split_rows),
    "color_groups": {f"{k[0]}:{k[1]}": v for k, v in sorted(color_groups.items())},
    "selected_train_validation_members": len(wanted),
    "selection": "at most three temporally spread vehicle crops per camera and vehicle ID",
    "split_policy": "deterministic SHA256 of raw vehicle ID; 75% train, 15% validation, 10% sealed future holdout",
    "privacy": "raw vehicle IDs remain only inside official tar member paths; track_group is salted SHA256",
    "policy": {
        "formal_train_eligible": False,
        "future_holdout_images_extracted": False,
        "future_holdout_used_for_selection": False,
        "stage148_test_reused": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "deployment_performed": False,
    },
}
plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(plan, ensure_ascii=False))
PY

mkdir "$PARTIAL"
curl -LfsS "$DATA_URL" \
  | tee >(sha256sum > "$SOURCE_ROOT/data.tgz.r2.stream.sha256") \
  | tar -xzf - -C "$PARTIAL" --no-recursion -T "$WANTED"

/root/miniconda3/bin/python - "$PLAN" "$MANIFEST" "$PARTIAL" "$STATE" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
import csv
import hashlib
import json
import sys

plan_path, manifest_path, root, state_path = map(Path, sys.argv[1:])
plan = json.loads(plan_path.read_text(encoding="utf-8"))
rows = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
expected = {row["tar_member"] for row in rows if row["split"] != "future_holdout"}
actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
decode_failures = []
for member in sorted(actual):
    try:
        with Image.open(root / member) as image:
            image.verify()
    except Exception as exc:
        decode_failures.append({"member": member, "error": str(exc)})
payload = {
    "schema_version": "stage149-vehicle-rear-selective-extract-v2",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": "selective_extraction_complete_pending_license_dedup_and_label_audit"
        if expected == actual and not decode_failures else "fail_closed_extraction_audit",
    "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "archive_stream_sha256": (plan_path.parent / "data.tgz.r2.stream.sha256").read_text().split()[0],
    "expected_train_validation_members": len(expected),
    "actual_train_validation_members": len(actual),
    "missing_members": sorted(expected - actual)[:200],
    "unexpected_members": sorted(actual - expected)[:200],
    "decode_failures": decode_failures[:200],
    "sealed_future_holdout_groups": plan["split_groups"].get("future_holdout", 0),
    "policy": {
        "formal_train_eligible": False,
        "future_holdout_images_extracted": False,
        "future_holdout_used_for_selection": False,
        "stage148_test_reused": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
    },
}
state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
state_path.with_suffix(state_path.suffix + ".sha256").write_text(
    f"{hashlib.sha256(state_path.read_bytes()).hexdigest()}  {state_path.name}\n",
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False))
if payload["status"].startswith("fail_closed"):
    raise SystemExit(5)
PY

mv "$PARTIAL" "$OUTPUT"
echo "Stage149 R2 selective extraction complete; formal admission remains locked"
