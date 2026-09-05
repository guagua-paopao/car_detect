#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
SNAPSHOT="$BASE/code/training_stage84_v2"
TRAIN="$SNAPSHOT/scripts/train_attribute.py"
LABELS="$BASE/code/training/config/vehicle_labels.truck-subtype.v1.json"
DATA="$BASE/datasets/attribute-domain-v2/stage120-axle-semantic-clean-r1"
MANIFEST="$DATA/attribute_manifest.stage120-axle-semantic-clean.csv"
REPORT="$DATA/stage120-axle-semantic-clean-report.json"
ROOT="$BASE/runs/attributes/ATTR-STAGE126-FRESH-AXLE-CLEAN-R1"
RUN="$ROOT/ATTR-STAGE126-TRUCK-SUBTYPE-CONVNEXT-256-FRESH-R1"
LOG="$ROOT/train.log"
SESSION=VCAS-STAGE126-FRESH-AXLE-CLEAN-R1

assert_sha256() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || { echo "SHA256 mismatch: $path" >&2; exit 64; }
}

for required in "$PY" "$TRAIN" "$LABELS" "$MANIFEST" "$REPORT"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$TRAIN" b1a7f8b5b67b15ff4466ead9c3256285415230c13da3b7b4aaebe8bdf5341ae6
assert_sha256 "$LABELS" 2bbb5f10bd3f7377e9a43d827ba743dc4d0407c25e17254afa4cb8d64ad8cc46
assert_sha256 "$MANIFEST" ed2614c9a0acf615452d90730a94cb84f094c72387244f63be938ef653c33c67
assert_sha256 "$REPORT" 30a4a8e3a2b3209da6ff5c336490a6c74021812d949d8f7eb77ee1f77ed7228d
"$PY" - "$REPORT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
assert d['status']=='pass'
assert d['output']['train_counts']=={'heavy_truck':2647,'light_truck':21852}
assert d['output']['validation_rows_preserved']==4713
assert d['integrity']['post_split_leaks']=={'exact':0,'near':0,'group':0}
assert d['policy']['generic_truck_never_supervises_heavy_subtype'] is True
assert d['policy']['test_accessed'] is False and d['policy']['frozen_video_used'] is False
PY
"$PY" -m py_compile "$TRAIN"

[[ ! -e "$ROOT" ]] || { echo "refusing to overwrite: $ROOT" >&2; exit 66; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists" >&2; exit 67; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
  echo "GPU is occupied" >&2; exit 68
fi
mkdir -p "$ROOT"
tmux new-session -d -s "$SESSION" \
  "'$PY' '$TRAIN' \
    --manifest '$MANIFEST' \
    --labels '$LABELS' \
    --input-size 256 \
    --architecture convnext_tiny \
    --resize-mode stretch \
    --epochs 12 \
    --batch-size 40 \
    --workers 8 \
    --learning-rate 0.00005 \
    --weight-decay 0.0001 \
    --body-loss-weight 1.0 \
    --color-loss-weight 0.0 \
    --focal-gamma 2.0 \
    --color-focal-gamma 0.0 \
    --class-weighting none \
    --label-smoothing 0.02 \
    --gradient-clip-norm 5.0 \
    --freeze-backbone-epochs 2 \
    --patience 5 \
    --type-threshold 0.80 \
    --color-threshold 0.99 \
    --gate-type-precision 0.95 \
    --gate-type-coverage 0.65 \
    --seed 20260905 \
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
    --run-kind formal \
    --dataset-version attribute-domain-v2-stage120-fresh-axle-clean-r1 \
    --code-revision stage126-fresh-axle-clean-r1 \
    --skip-test \
    --output-dir '$RUN' >'$LOG' 2>&1"
echo "started tmux:$SESSION"
echo "run=$RUN"
echo "log=$LOG"
