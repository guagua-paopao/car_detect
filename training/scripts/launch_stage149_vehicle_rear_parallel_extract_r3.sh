#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
SOURCE_ROOT="$BASE/sources/vehicle-rear-stage149"
ARCHIVE="$SOURCE_ROOT/data.r3.tgz"
OUTPUT="$SOURCE_ROOT/selected-r3"
PARTIAL="$SOURCE_ROOT/selected-r3.partial"
PLAN="$SOURCE_ROOT/stage149-selection-plan-r2.json"
MANIFEST="$SOURCE_ROOT/stage149-selection-plan-r2.csv"
WANTED="$SOURCE_ROOT/stage149-train-validation-members-r2.txt"
STATE="$BASE/runs/attributes/ATTR-STAGE149-VEHICLE-REAR-PARALLEL-EXTRACT-R3.state.json"
LOG="$BASE/runs/attributes/ATTR-STAGE149-VEHICLE-REAR-PARALLEL-EXTRACT-R3.log"
SESSION=VCAS-DL-VEHICLEREAR149-R3
DATA_URL=https://www.inf.ufpr.br/vri/databases/vehicle-reid/data.tgz
EXPECTED_ARCHIVE_BYTES=6451005771

if [[ "${1:-}" != "--worker" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "$SESSION already running"
    exit 0
  fi
  tmux new-session -d -s "$SESSION" "bash '$0' --worker >>'$LOG' 2>&1"
  echo "started tmux:$SESSION"
  exit 0
fi

for required in "$PLAN" "$MANIFEST" "$WANTED"; do
  [[ -f "$required" ]] || { echo "missing required input: $required" >&2; exit 2; }
done
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite output: $OUTPUT" >&2; exit 2; }
[[ ! -e "$PARTIAL" ]] || { echo "refusing to reuse output: $PARTIAL" >&2; exit 2; }
mkdir -p "$SOURCE_ROOT" "$(dirname "$STATE")"

aria2c \
  --allow-overwrite=false \
  --auto-file-renaming=false \
  --continue=true \
  --file-allocation=none \
  --max-connection-per-server=8 \
  --min-split-size=16M \
  --split=8 \
  --summary-interval=30 \
  --dir="$SOURCE_ROOT" \
  --out="$(basename "$ARCHIVE")" \
  "$DATA_URL"

[[ -f "$ARCHIVE" && ! -L "$ARCHIVE" ]] || { echo "archive is not a regular non-link file" >&2; exit 3; }
archive_bytes=$(stat -c %s "$ARCHIVE")
[[ "$archive_bytes" == "$EXPECTED_ARCHIVE_BYTES" ]] || {
  echo "archive size mismatch: $archive_bytes" >&2
  exit 3
}
sha256sum "$ARCHIVE" > "$SOURCE_ROOT/data.r3.tgz.sha256"
tar -tzf "$ARCHIVE" >/dev/null

mkdir "$PARTIAL"
tar -xzf "$ARCHIVE" -C "$PARTIAL" --no-recursion -T "$WANTED"

/root/miniconda3/bin/python - "$PLAN" "$MANIFEST" "$PARTIAL" "$STATE" "$ARCHIVE" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
import csv
import hashlib
import json
import sys

plan_path, manifest_path, root, state_path, archive = map(Path, sys.argv[1:])
plan = json.loads(plan_path.read_text(encoding="utf-8"))
rows = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
expected = {row["tar_member"] for row in rows if row["split"] != "future_holdout"}
actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
decode_failures = []
total_image_bytes = 0
for member in sorted(actual):
    path = root / member
    total_image_bytes += path.stat().st_size
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        decode_failures.append({"member": member, "error": str(exc)})
archive_hash = hashlib.sha256()
with archive.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        archive_hash.update(chunk)
archive_sha = archive_hash.hexdigest()
payload = {
    "schema_version": "stage149-vehicle-rear-parallel-extract-v3",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": "selective_extraction_complete_pending_license_dedup_and_label_audit"
        if expected == actual and not decode_failures else "fail_closed_extraction_audit",
    "download": {
        "url": "https://www.inf.ufpr.br/vri/databases/vehicle-reid/data.tgz",
        "bytes": archive.stat().st_size,
        "sha256": archive_sha,
        "connections": 8,
        "archive_integrity": "pass",
    },
    "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "expected_train_validation_members": len(expected),
    "actual_train_validation_members": len(actual),
    "selected_image_bytes": total_image_bytes,
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

# The fully verified archive is a re-downloadable transfer cache. Keep its SHA256
# evidence, validate its exact fixed path again, and remove only that one file so
# training retains sufficient workspace. No source dataset or extracted image is removed.
/root/miniconda3/bin/python - "$ARCHIVE" "$SOURCE_ROOT" "$EXPECTED_ARCHIVE_BYTES" <<'PY'
from pathlib import Path
import sys
target = Path(sys.argv[1])
root = Path(sys.argv[2]).resolve()
expected = int(sys.argv[3])
resolved = target.resolve()
if resolved.parent != root or resolved.name != "data.r3.tgz":
    raise SystemExit(f"unsafe archive target: {resolved}")
if not target.is_file() or target.is_symlink() or target.stat().st_size != expected:
    raise SystemExit("archive target validation failed")
print(resolved)
PY
rm -- "$ARCHIVE"

echo "Stage149 R3 selective extraction complete; temporary verified archive removed; formal admission remains locked"
