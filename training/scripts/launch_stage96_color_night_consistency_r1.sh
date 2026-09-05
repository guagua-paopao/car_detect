#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
SNAPSHOT="$BASE/code/training_stage84_v2"
TRAIN="$SNAPSHOT/scripts/train_attribute.py"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
MANIFEST="$BASE/datasets/attribute-domain-v2/stage78-dvm-fine-color-runtime-v2/attribute_manifest.stage78-dvm-fine-color-runtime.csv"
MANIFEST_REPORT="$BASE/datasets/attribute-domain-v2/stage78-dvm-fine-color-runtime-v2/stage78-dvm-fine-color-runtime-report.json"
UNLABELED="$BASE/datasets/attribute-domain-v2/stage95-color-consistency-r1/attribute_manifest.stage95-color-consistency.csv"
UNLABELED_REPORT="$BASE/datasets/attribute-domain-v2/stage95-color-consistency-r1/stage95-color-consistency-report.json"
INIT="$BASE/runs/attributes/ATTR-STAGE80-COLOR-CONVNEXT-256-V2-DVM-R3/best.pt"
ROOT="$BASE/runs/attributes/ATTR-STAGE96-COLOR-NIGHT-CONSISTENCY-R1"
RUN="$ROOT/ATTR-STAGE96-COLOR-CONVNEXT-256-NIGHT-CONSISTENCY-R1"
LOG="$ROOT/train.log"
SESSION=VCAS-STAGE96-COLOR-NIGHT-R1

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

for required in "$PY" "$TRAIN" "$LABELS" "$MANIFEST" "$MANIFEST_REPORT" "$UNLABELED" "$UNLABELED_REPORT" "$INIT"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$TRAIN" b1a7f8b5b67b15ff4466ead9c3256285415230c13da3b7b4aaebe8bdf5341ae6
assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
assert_sha256 "$MANIFEST" dc06fe38daa5aaea1daf0c38133265e9d7f0650b5aed32f613462a1d94f1edb5
assert_sha256 "$MANIFEST_REPORT" 905228437b5ee81e103344e97edb83426fc7a853367bb40ea788fa58a883a970
assert_sha256 "$UNLABELED" 793dba62275f352150555d75441b145812cd98a34b843321adc2ca4df43105d3
assert_sha256 "$UNLABELED_REPORT" a2269b04f6b94393e4441390ebd2047272ee8af66e5524b8a6a8e2248b3ce68e
assert_sha256 "$INIT" a2894e21060a09c690756d83bb776ffe71acda56da9f2760df577ce080c00c47

"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="pass"; assert d["output"]["verified_night_rows"]>=1500; assert d["policy"]["all_attributes_unknown_unsupervised"] is True; assert d["policy"]["test_accessed"] is False; assert d["policy"]["frozen_video_used"] is False' "$UNLABELED_REPORT"
"$PY" -m py_compile "$TRAIN"

[[ ! -e "$ROOT" ]] || { echo "refusing to overwrite: $ROOT" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
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
    --body-loss-weight 0.0 \
    --color-loss-weight 1.3 \
    --focal-gamma 0.0 \
    --color-focal-gamma 1.5 \
    --class-weighting inverse_sqrt \
    --label-smoothing 0.03 \
    --gradient-clip-norm 5.0 \
    --freeze-backbone-epochs 0 \
    --patience 3 \
    --type-threshold 0.99 \
    --color-threshold 0.70 \
    --gate-type-precision 0.93 \
    --gate-type-coverage 0.45 \
    --gate-color-precision 0.93 \
    --gate-color-coverage 0.25 \
    --seed 20260831 \
    --augmentation-profile color_scene \
    --selection-head color \
    --body-hierarchy none \
    --coarse-car-loss-weight 0.0 \
    --coarse-truck-loss-weight 0.0 \
    --night-sample-weight 1.0 \
    --occlusion-sample-weight 1.0 \
    --hard-sample-weight 0.0 \
    --small-sample-weight 0.0 \
    --color-sample-weight 0.0 \
    --pseudo-label-weight 1.0 \
    --unlabeled-manifest '$UNLABELED' \
    --unlabeled-root '$BASE/datasets/attribute-domain-v2' \
    --unlabeled-batch-size 40 \
    --unlabeled-consistency-weight 0.02 \
    --unlabeled-consistency-mode feature \
    --unlabeled-body-weight 0.0 \
    --unlabeled-color-weight 1.0 \
    --unlabeled-consistency-temperature 1.0 \
    --unlabeled-ema-decay 0.999 \
    --unlabeled-night-sample-weight 8.0 \
    --init-checkpoint '$INIT' \
    --run-kind formal \
    --dataset-version attribute-domain-v2-stage96-color-night-consistency-r1 \
    --code-revision stage96-color-night-consistency-r1 \
    --skip-test \
    --output-dir '$RUN' >'$LOG' 2>&1"

echo "started tmux:$SESSION"
echo "run=$RUN"
echo "log=$LOG"
