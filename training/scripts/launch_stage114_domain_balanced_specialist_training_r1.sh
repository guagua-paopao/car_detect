#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
SNAPSHOT="$BASE/code/training_stage84_v2"
TRAIN="$SNAPSHOT/scripts/train_attribute.py"
LABELS="$BASE/code/training/config/vehicle_labels.truck-subtype.v1.json"
DATA="$BASE/datasets/attribute-domain-v2/stage113-domain-balanced-specialist-r1"
MANIFEST="$DATA/attribute_manifest.stage113-domain-balanced-specialist.csv"
REPORT="$DATA/stage113-domain-balanced-specialist-report.json"
INIT="$BASE/runs/attributes/ATTR-STAGE99-TWO-AXLE-SPECIALIST-R1/ATTR-STAGE99-TRUCK-SUBTYPE-CONVNEXT-256-TWO-AXLE-R1/best.pt"
ROOT="$BASE/runs/attributes/ATTR-STAGE114-DOMAIN-BALANCED-SPECIALIST-R1"
RUN="$ROOT/ATTR-STAGE114-TRUCK-SUBTYPE-CONVNEXT-256-DOMAIN-BALANCED-R1"
LOG="$ROOT/train.log"
SESSION=VCAS-STAGE114-DOMAIN-BALANCED-SPECIALIST-R1

assert_sha256() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
    exit 64
  }
}

for required in "$PY" "$TRAIN" "$LABELS" "$MANIFEST" "$REPORT" "$INIT"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$TRAIN" b1a7f8b5b67b15ff4466ead9c3256285415230c13da3b7b4aaebe8bdf5341ae6
assert_sha256 "$LABELS" 2bbb5f10bd3f7377e9a43d827ba743dc4d0407c25e17254afa4cb8d64ad8cc46
assert_sha256 "$MANIFEST" 70100165c058f72cbb7c9fe35a81fd4e6336c3dcc32de0fdd02dc93ce7969cb0
assert_sha256 "$REPORT" fc77afeb786a63ca1f9b8a2cb444ac73191b0ae108becaf3a42d2499ab637833
assert_sha256 "$INIT" 0da63e45ffe4c5af4594288b10ed0ea209bee51993c676083d01847fc2ca4fae

"$PY" - "$REPORT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
assert d['status']=='pass'
assert d['output']['train_counts']=={'heavy_truck':14913,'light_truck':25125}
assert d['output']['validation_rows_preserved']==4713
assert d['output']['excluded']=={'noncommercial_or_research_only':8659}
assert d['integrity']['post_split_leaks']=={'exact':0,'near':0,'group':0}
assert d['policy']['noncommercial_and_research_only_train_rows_excluded'] is True
assert d['policy']['test_accessed'] is False and d['policy']['frozen_video_used'] is False
PY
"$PY" -m py_compile "$TRAIN"

[[ ! -e "$ROOT" ]] || { echo "refusing to overwrite: $ROOT" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "session already exists" >&2; exit 67; fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
  echo "GPU is occupied" >&2
  exit 68
fi

mkdir -p "$ROOT"
tmux new-session -d -s "$SESSION" \
  "'$PY' '$TRAIN' \
    --manifest '$MANIFEST' \
    --labels '$LABELS' \
    --input-size 256 \
    --architecture convnext_tiny \
    --resize-mode stretch \
    --epochs 8 \
    --batch-size 40 \
    --workers 8 \
    --learning-rate 0.000005 \
    --weight-decay 0.0001 \
    --body-loss-weight 1.0 \
    --color-loss-weight 0.0 \
    --focal-gamma 1.0 \
    --color-focal-gamma 0.0 \
    --class-weighting inverse_sqrt \
    --label-smoothing 0.01 \
    --gradient-clip-norm 5.0 \
    --freeze-backbone-epochs 0 \
    --patience 3 \
    --type-threshold 0.80 \
    --color-threshold 0.99 \
    --gate-type-precision 0.95 \
    --gate-type-coverage 0.65 \
    --seed 20260902 \
    --augmentation-profile hard_scene \
    --selection-head body \
    --body-hierarchy none \
    --coarse-car-loss-weight 0.0 \
    --coarse-truck-loss-weight 0.0 \
    --night-sample-weight 1.5 \
    --occlusion-sample-weight 1.25 \
    --hard-sample-weight 0.05 \
    --small-sample-weight 0.75 \
    --color-sample-weight 0.0 \
    --pseudo-label-weight 1.0 \
    --init-checkpoint '$INIT' \
    --run-kind formal \
    --dataset-version attribute-domain-v2-stage113-domain-balanced-specialist-r1 \
    --code-revision stage114-domain-balanced-specialist-r1 \
    --skip-test \
    --output-dir '$RUN' >'$LOG' 2>&1"

echo "started tmux:$SESSION"
echo "run=$RUN"
echo "log=$LOG"
