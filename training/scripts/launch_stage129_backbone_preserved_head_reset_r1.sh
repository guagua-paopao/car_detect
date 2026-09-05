#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage97_eval_r1"
TRAIN_SNAPSHOT="$BASE/code/training_stage84_v2"
RESET="$CODE/scripts/reset_stage129_specialist_head.py"
RESET_TEST="$CODE/tests/test_reset_stage129_specialist_head.py"
TRAIN="$TRAIN_SNAPSHOT/scripts/train_attribute.py"
LABELS="$BASE/code/training/config/vehicle_labels.truck-subtype.v1.json"
DATA="$BASE/datasets/attribute-domain-v2/stage120-axle-semantic-clean-r1"
MANIFEST="$DATA/attribute_manifest.stage120-axle-semantic-clean.csv"
REPORT="$DATA/stage120-axle-semantic-clean-report.json"
SOURCE="$BASE/runs/attributes/ATTR-STAGE99-TWO-AXLE-SPECIALIST-R1/ATTR-STAGE99-TRUCK-SUBTYPE-CONVNEXT-256-TWO-AXLE-R1/best.pt"
ROOT="$BASE/runs/attributes/ATTR-STAGE129-BACKBONE-PRESERVED-HEAD-RESET-R1"
INIT="$ROOT/stage129-backbone-preserved-head-reset.pt"
INIT_REPORT="$ROOT/stage129-backbone-preserved-head-reset-report.json"
RUN="$ROOT/ATTR-STAGE129-TRUCK-SUBTYPE-CONVNEXT-256-HEAD-ONLY-R1"
LOG="$ROOT/train.log"
SESSION=VCAS-STAGE129-BACKBONE-PRESERVED-HEAD-RESET-R1

assert_sha256() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || { echo "SHA256 mismatch: $path" >&2; exit 64; }
}

for required in "$PY" "$RESET" "$RESET_TEST" "$TRAIN" "$LABELS" "$MANIFEST" "$REPORT" "$SOURCE"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$RESET" 0ef8184b4b0ca85eabed39eea43e141684e4adf889a0c32a5b0476745c330f83
assert_sha256 "$RESET_TEST" 218726134cad08655ad0dbcbd6e79ab111971bbb7ee2fa66e96b04c6f9f2221e
assert_sha256 "$TRAIN" b1a7f8b5b67b15ff4466ead9c3256285415230c13da3b7b4aaebe8bdf5341ae6
assert_sha256 "$LABELS" 2bbb5f10bd3f7377e9a43d827ba743dc4d0407c25e17254afa4cb8d64ad8cc46
assert_sha256 "$MANIFEST" ed2614c9a0acf615452d90730a94cb84f094c72387244f63be938ef653c33c67
assert_sha256 "$REPORT" 30a4a8e3a2b3209da6ff5c336490a6c74021812d949d8f7eb77ee1f77ed7228d
assert_sha256 "$SOURCE" 0da63e45ffe4c5af4594288b10ed0ea209bee51993c676083d01847fc2ca4fae
"$PY" -m py_compile "$RESET" "$RESET_TEST" "$TRAIN"
cd "$CODE"
"$PY" -m unittest discover -s tests -p 'test_reset_stage129_specialist_head.py'
[[ ! -e "$ROOT" ]] || { echo "refusing to overwrite Stage129" >&2; exit 66; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists" >&2; exit 67; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
  echo "GPU is occupied" >&2; exit 68
fi
"$PY" "$RESET" \
  --input-checkpoint "$SOURCE" \
  --expected-input-sha256 0da63e45ffe4c5af4594288b10ed0ea209bee51993c676083d01847fc2ca4fae \
  --output-checkpoint "$INIT" --output-report "$INIT_REPORT" --seed 20260906
"$PY" - "$INIT_REPORT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
assert d['status']=='pass'
assert d['unchanged_non_head'] is True
assert d['backbone_digest_before']==d['backbone_digest_after']
assert d['body_head_digest_before']!=d['body_head_digest_after']
assert d['policy']['test_accessed'] is False and d['policy']['frozen_video_used'] is False
PY
tmux new-session -d -s "$SESSION" \
  "'$PY' '$TRAIN' \
    --manifest '$MANIFEST' --labels '$LABELS' \
    --input-size 256 --architecture convnext_tiny --resize-mode stretch \
    --epochs 4 --batch-size 40 --workers 8 --learning-rate 0.0005 --weight-decay 0.0001 \
    --body-loss-weight 1.0 --color-loss-weight 0.0 --focal-gamma 2.0 --color-focal-gamma 0.0 \
    --class-weighting none --label-smoothing 0.02 --gradient-clip-norm 5.0 \
    --freeze-backbone-epochs 4 --patience 4 \
    --type-threshold 0.80 --color-threshold 0.99 \
    --gate-type-precision 0.95 --gate-type-coverage 0.65 \
    --seed 20260906 --augmentation-profile hard_scene --selection-head body \
    --body-hierarchy none --coarse-car-loss-weight 0.0 --coarse-truck-loss-weight 0.0 \
    --night-sample-weight 2.0 --occlusion-sample-weight 1.5 --hard-sample-weight 0.2 \
    --small-sample-weight 1.5 --color-sample-weight 0.0 --pseudo-label-weight 1.0 \
    --init-checkpoint '$INIT' --run-kind formal \
    --dataset-version attribute-domain-v2-stage120-backbone-preserved-head-reset-r1 \
    --code-revision stage129-backbone-preserved-head-reset-r1 --skip-test \
    --output-dir '$RUN' >'$LOG' 2>&1"
echo "started tmux:$SESSION"
echo "init_report=$INIT_REPORT"
echo "run=$RUN"
