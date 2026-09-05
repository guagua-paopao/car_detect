#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
SNAPSHOT="$BASE/code/training_stage84_v2"
TRAIN="$SNAPSHOT/scripts/train_attribute.py"
MIGRATOR="$SNAPSHOT/scripts/migrate_attribute_checkpoint_taxonomy_v2.py"
LABELS="$BASE/code/training/config/vehicle_labels.truck-subtype.v1.json"
MANIFEST="$BASE/datasets/attribute-domain-v2/stage93-truck-subtype-r2/attribute_manifest.stage93-truck-subtype.csv"
MANIFEST_REPORT="$BASE/datasets/attribute-domain-v2/stage93-truck-subtype-r2/stage93-truck-subtype-report.json"
SOURCE_CHECKPOINT="$BASE/runs/attributes/ATTR-STAGE84-BODY-V2-CHAIN-R2/ATTR-STAGE84-BODY-CONVNEXT-256-V2-MIO-CCTV-R2/best.pt"
ROOT="$BASE/runs/attributes/ATTR-STAGE94-TRUCK-SUBTYPE-R1"
INIT="$ROOT/stage94-truck-subtype-init.pt"
INIT_REPORT="$ROOT/stage94-truck-subtype-init-report.json"
RUN="$ROOT/ATTR-STAGE94-TRUCK-SUBTYPE-CONVNEXT-256-R1"
LOG="$ROOT/train.log"
SESSION=VCAS-STAGE94-TRUCK-SUBTYPE-R1

assert_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
    exit 64
  }
}

for required in "$PY" "$TRAIN" "$MIGRATOR" "$LABELS" "$MANIFEST" "$MANIFEST_REPORT" "$SOURCE_CHECKPOINT"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$TRAIN" b1a7f8b5b67b15ff4466ead9c3256285415230c13da3b7b4aaebe8bdf5341ae6
assert_sha256 "$MIGRATOR" 3aa5009bef44fd9b5ddfa097ae2538a16ee2b52cb56e38976c7a9fd3462b96f2
assert_sha256 "$LABELS" 2bbb5f10bd3f7377e9a43d827ba743dc4d0407c25e17254afa4cb8d64ad8cc46
assert_sha256 "$MANIFEST" d8582417f7826ac76c2e290595c754a54df0838dc6bef324531cd7167d49256a
assert_sha256 "$MANIFEST_REPORT" 27d13e2ce0e395bbc82fc8254c7d1dead5a4668c5310f90bd9d930a63771ea32
assert_sha256 "$SOURCE_CHECKPOINT" e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec

"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="pass"; assert d["policy"]["test_accessed"] is False; assert d["policy"]["frozen_video_used"] is False; assert d["selection"]["generic_truck_rows_allowed"]==0; assert d["integrity"]["post_split_leaks"]=={"exact":0,"near":0,"group":0}' "$MANIFEST_REPORT"
"$PY" -m py_compile "$TRAIN" "$MIGRATOR"

[[ ! -e "$ROOT" ]] || { echo "refusing to overwrite: $ROOT" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi

"$PY" "$MIGRATOR" \
  --input-checkpoint "$SOURCE_CHECKPOINT" \
  --labels "$LABELS" \
  --output-checkpoint "$INIT" \
  --output-report "$INIT_REPORT"

tmux new-session -d -s "$SESSION" \
  "'$PY' '$TRAIN' \
    --manifest '$MANIFEST' \
    --labels '$LABELS' \
    --input-size 256 \
    --architecture convnext_tiny \
    --resize-mode stretch \
    --epochs 10 \
    --batch-size 40 \
    --workers 8 \
    --learning-rate 0.00001 \
    --weight-decay 0.0001 \
    --body-loss-weight 1.0 \
    --color-loss-weight 0.0 \
    --focal-gamma 2.0 \
    --color-focal-gamma 0.0 \
    --class-weighting inverse_sqrt \
    --label-smoothing 0.02 \
    --gradient-clip-norm 5.0 \
    --freeze-backbone-epochs 1 \
    --patience 4 \
    --type-threshold 0.80 \
    --color-threshold 0.99 \
    --gate-type-precision 0.95 \
    --gate-type-coverage 0.65 \
    --gate-color-precision 0.93 \
    --gate-color-coverage 0.25 \
    --seed 20260830 \
    --augmentation-profile hard_scene \
    --selection-head body \
    --body-hierarchy none \
    --coarse-car-loss-weight 0.0 \
    --coarse-truck-loss-weight 0.0 \
    --night-sample-weight 2.0 \
    --occlusion-sample-weight 1.5 \
    --hard-sample-weight 0.2 \
    --small-sample-weight 1.5 \
    --color-sample-weight 0.0 \
    --pseudo-label-weight 1.0 \
    --init-checkpoint '$INIT' \
    --run-kind formal \
    --dataset-version attribute-domain-v2-stage94-truck-subtype-r1 \
    --code-revision stage94-truck-subtype-r1 \
    --skip-test \
    --output-dir '$RUN' >'$LOG' 2>&1"

echo "started tmux:$SESSION"
echo "run=$RUN"
echo "log=$LOG"
