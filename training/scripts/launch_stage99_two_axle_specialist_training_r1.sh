#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
SNAPSHOT="$BASE/code/training_stage84_v2"
TRAIN="$SNAPSHOT/scripts/train_attribute.py"
LABELS="$BASE/code/training/config/vehicle_labels.truck-subtype.v1.json"
MANIFEST="$BASE/datasets/attribute-domain-v2/stage98-two-axle-cctv-r1/attribute_manifest.stage98-two-axle-cctv.csv"
MANIFEST_REPORT="$BASE/datasets/attribute-domain-v2/stage98-two-axle-cctv-r1/stage98-two-axle-cctv-report.json"
INIT="$BASE/runs/attributes/ATTR-STAGE94-TRUCK-SUBTYPE-R1/ATTR-STAGE94-TRUCK-SUBTYPE-CONVNEXT-256-R1/best.pt"
ROOT="$BASE/runs/attributes/ATTR-STAGE99-TWO-AXLE-SPECIALIST-R1"
RUN="$ROOT/ATTR-STAGE99-TRUCK-SUBTYPE-CONVNEXT-256-TWO-AXLE-R1"
LOG="$ROOT/train.log"
SESSION=VCAS-STAGE99-TWO-AXLE-R1

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

for required in "$PY" "$TRAIN" "$LABELS" "$MANIFEST" "$MANIFEST_REPORT" "$INIT"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$TRAIN" b1a7f8b5b67b15ff4466ead9c3256285415230c13da3b7b4aaebe8bdf5341ae6
assert_sha256 "$LABELS" 2bbb5f10bd3f7377e9a43d827ba743dc4d0407c25e17254afa4cb8d64ad8cc46
assert_sha256 "$MANIFEST" fd1da923aa4104c95b17d7b9c30bc0cde5cd170aa4f9b0bf74484cd65c4a2af3
assert_sha256 "$MANIFEST_REPORT" 234b2d55cf9689320311aad34ac24bb5a99aa168a7faa593388b09cc67f0f4c7
assert_sha256 "$INIT" 0f716377586fe340876042f3e7213383dd6cc2a6ec087a5fde0f737186ddf232

"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="pass"; assert d["mapping"]["contract"]=="two_axle_to_light_truck;three_plus_axle_to_heavy_truck"; assert d["output"]["eligible_two_axle_rows"]>=4000; assert d["integrity"]["post_split_leaks"]=={"exact":0,"near":0,"group":0}; assert d["policy"]["publisher_validation_payloads_read"]==0; assert d["policy"]["publisher_test_payloads_read"]==0; assert d["policy"]["vfg_validation_used_for_training"] is False; assert d["policy"]["frozen_video_used"] is False' "$MANIFEST_REPORT"
"$PY" -m py_compile "$TRAIN"

[[ ! -e "$ROOT" ]] || { echo "refusing to overwrite: $ROOT" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
  echo "GPU is occupied; Stage99 must wait for Stage96 and its independent validation" >&2
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
    --learning-rate 0.000007 \
    --weight-decay 0.0001 \
    --body-loss-weight 1.0 \
    --color-loss-weight 0.0 \
    --focal-gamma 2.0 \
    --color-focal-gamma 0.0 \
    --class-weighting inverse_sqrt \
    --label-smoothing 0.02 \
    --gradient-clip-norm 5.0 \
    --freeze-backbone-epochs 0 \
    --patience 3 \
    --type-threshold 0.80 \
    --color-threshold 0.99 \
    --gate-type-precision 0.95 \
    --gate-type-coverage 0.65 \
    --seed 20260831 \
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
    --dataset-version attribute-domain-v2-stage99-two-axle-specialist-r1 \
    --code-revision stage99-two-axle-specialist-r1 \
    --skip-test \
    --output-dir '$RUN' >'$LOG' 2>&1"

echo "started tmux:$SESSION"
echo "run=$RUN"
echo "log=$LOG"
