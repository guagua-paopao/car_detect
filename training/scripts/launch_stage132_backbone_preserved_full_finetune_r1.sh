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
INIT="$BASE/runs/attributes/ATTR-STAGE129-BACKBONE-PRESERVED-HEAD-RESET-R1/ATTR-STAGE129-TRUCK-SUBTYPE-CONVNEXT-256-HEAD-ONLY-R1/best.pt"
BODY_STATE="$BASE/runs/attributes/ATTR-STAGE130-HEAD-RESET-VALIDATION-R1.state.json"
INTEGRATED_STATE="$BASE/runs/attributes/ATTR-STAGE131-INTEGRATED-COMPONENT-GATE-R1.state.json"
ROOT="$BASE/runs/attributes/ATTR-STAGE132-BACKBONE-PRESERVED-FULL-FINETUNE-R1"
RUN="$ROOT/ATTR-STAGE132-TRUCK-SUBTYPE-CONVNEXT-256-FULL-FINETUNE-R1"
LOG="$ROOT/train.log"
SESSION=VCAS-STAGE132-BACKBONE-PRESERVED-FULL-FINETUNE-R1

assert_sha256() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
    exit 64
  }
}

for required in "$PY" "$TRAIN" "$LABELS" "$MANIFEST" "$REPORT" "$INIT" "$BODY_STATE" "$INTEGRATED_STATE"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$TRAIN" b1a7f8b5b67b15ff4466ead9c3256285415230c13da3b7b4aaebe8bdf5341ae6
assert_sha256 "$LABELS" 2bbb5f10bd3f7377e9a43d827ba743dc4d0407c25e17254afa4cb8d64ad8cc46
assert_sha256 "$MANIFEST" ed2614c9a0acf615452d90730a94cb84f094c72387244f63be938ef653c33c67
assert_sha256 "$REPORT" 30a4a8e3a2b3209da6ff5c336490a6c74021812d949d8f7eb77ee1f77ed7228d
assert_sha256 "$INIT" 1cb1bb40c5e8cde2244b8dcce992239e758af6ca5cf074f4d53f5761b2ada2a7
assert_sha256 "$BODY_STATE" 1e797ece7a9d105a1a5c7fa889fcd5b1d934bf3427329a9d6993ce4ec9facc61
assert_sha256 "$INTEGRATED_STATE" 6faa7fb2b76fd0a7cabf0b63e478aaadbe6e21c70fbd17bef55b8e58f90ca0c8

"$PY" - "$REPORT" "$BODY_STATE" "$INTEGRATED_STATE" <<'PY'
import json,sys
report=json.load(open(sys.argv[1],encoding='utf-8'))
body=json.load(open(sys.argv[2],encoding='utf-8'))
integrated=json.load(open(sys.argv[3],encoding='utf-8'))
assert report['status']=='pass'
assert report['output']['train_counts']=={'heavy_truck':2647,'light_truck':21852}
assert report['integrity']['post_split_leaks']=={'exact':0,'near':0,'group':0}
assert report['policy']['generic_truck_never_supervises_heavy_subtype'] is True
assert report['policy']['test_accessed'] is False and report['policy']['frozen_video_used'] is False
assert body['decision']=='body_component_rejected_fail_closed'
assert body['qualified_variants']==[]
assert integrated['status']=='component_repair_required_fail_closed'
assert integrated['independent_test_authorized'] is False
assert integrated['test_accessed'] is False and integrated['frozen_video_used'] is False
PY
"$PY" -m py_compile "$TRAIN"

[[ ! -e "$ROOT" ]] || { echo "refusing to overwrite: $ROOT" >&2; exit 66; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists" >&2; exit 67; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
  echo "GPU is occupied" >&2
  exit 68
fi

mkdir -p "$ROOT"
tmux new-session -d -s "$SESSION" \
  "'$PY' '$TRAIN' \
    --manifest '$MANIFEST' --labels '$LABELS' \
    --input-size 256 --architecture convnext_tiny --resize-mode stretch \
    --epochs 6 --batch-size 40 --workers 8 --learning-rate 0.000003 --weight-decay 0.0001 \
    --body-loss-weight 1.0 --color-loss-weight 0.0 --focal-gamma 2.0 --color-focal-gamma 0.0 \
    --class-weighting none --label-smoothing 0.01 --gradient-clip-norm 5.0 \
    --freeze-backbone-epochs 0 --patience 4 \
    --type-threshold 0.80 --color-threshold 0.99 \
    --gate-type-precision 0.95 --gate-type-coverage 0.65 \
    --seed 20260907 --augmentation-profile hard_scene --selection-head body \
    --body-hierarchy none --coarse-car-loss-weight 0.0 --coarse-truck-loss-weight 0.0 \
    --night-sample-weight 2.0 --occlusion-sample-weight 1.5 --hard-sample-weight 0.2 \
    --small-sample-weight 1.5 --color-sample-weight 0.0 --pseudo-label-weight 1.0 \
    --init-checkpoint '$INIT' --run-kind formal \
    --dataset-version attribute-domain-v2-stage120-backbone-preserved-full-finetune-r1 \
    --code-revision stage132-backbone-preserved-full-finetune-r1 --skip-test \
    --output-dir '$RUN' >'$LOG' 2>&1"
echo "started tmux:$SESSION"
echo "run=$RUN"
echo "log=$LOG"
