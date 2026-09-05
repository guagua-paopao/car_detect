#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
SNAPSHOT="$BASE/code/training_stage84_v2"
TRAIN="$SNAPSHOT/scripts/train_attribute.py"
LABELS="$BASE/code/training/config/vehicle_labels.truck-subtype.v1.json"
DATA="$BASE/datasets/attribute-domain-v2/stage136-mio-road-domain-axle-r1"
MANIFEST="$DATA/attribute_manifest.stage136-mio-road-domain-axle.csv"
REPORT="$DATA/stage136-mio-road-domain-axle-report.json"
INIT="$BASE/runs/attributes/ATTR-STAGE121-AXLE-SEMANTIC-CLEAN-R1/ATTR-STAGE121-TRUCK-SUBTYPE-CONVNEXT-256-AXLE-CLEAN-R1/best.pt"
ROOT="$BASE/runs/attributes/ATTR-STAGE137-MIO-ROAD-DOMAIN-HEAD-TUNE-R1"
RUN="$ROOT/ATTR-STAGE137-TRUCK-SUBTYPE-CONVNEXT-256-MIO-HEAD-TUNE-R1"
LOG="$ROOT/train.log"
SESSION=VCAS-STAGE137-MIO-ROAD-DOMAIN-HEAD-TUNE-R1

assert_sha256() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || { echo "SHA256 mismatch: $path" >&2; exit 64; }
}

for required in "$PY" "$TRAIN" "$LABELS" "$MANIFEST" "$REPORT" "$INIT"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$TRAIN" b1a7f8b5b67b15ff4466ead9c3256285415230c13da3b7b4aaebe8bdf5341ae6
assert_sha256 "$LABELS" 2bbb5f10bd3f7377e9a43d827ba743dc4d0407c25e17254afa4cb8d64ad8cc46
assert_sha256 "$MANIFEST" a12be8b1cc38fa777bfd219b2a39f74b59092347c7e4b63b0d6e97509007645d
assert_sha256 "$REPORT" ed1e8c01e75fb13b7d5294d9fe8a3aa363d9f96aafbea833b989fda661823c7d
assert_sha256 "$INIT" fa80144ffc84886188d8268935f5cdadb5d74cda3af170b44cf80a4d3f060545
"$PY" - "$REPORT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
assert d['status']=='pass'
assert d['output']['accepted_class_counts']=={'heavy_truck':2259,'light_truck':2300}
assert d['integrity']['post_split_leaks']=={'exact':0,'near':0,'group':0}
assert d['policy']['research_only'] is True and d['policy']['deployment_eligible'] is False
assert d['policy']['official_test_image_payloads_read']==0
assert d['policy']['test_accessed'] is False and d['policy']['frozen_video_used'] is False
PY
"$PY" -m py_compile "$TRAIN"
[[ ! -e "$ROOT" ]] || { echo "refusing to overwrite Stage137" >&2; exit 66; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists" >&2; exit 67; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
  echo "GPU is occupied" >&2; exit 68
fi
mkdir -p "$ROOT"
tmux new-session -d -s "$SESSION" \
  "'$PY' '$TRAIN' \
    --manifest '$MANIFEST' --labels '$LABELS' \
    --input-size 256 --architecture convnext_tiny --resize-mode stretch \
    --epochs 4 --batch-size 40 --workers 8 --learning-rate 0.0001 --weight-decay 0.0001 \
    --body-loss-weight 1.0 --color-loss-weight 0.0 --focal-gamma 2.0 --color-focal-gamma 0.0 \
    --class-weighting none --label-smoothing 0.01 --gradient-clip-norm 5.0 \
    --freeze-backbone-epochs 4 --patience 4 \
    --type-threshold 0.80 --color-threshold 0.99 \
    --gate-type-precision 0.95 --gate-type-coverage 0.65 \
    --seed 20260908 --augmentation-profile hard_scene --selection-head body \
    --body-hierarchy none --coarse-car-loss-weight 0.0 --coarse-truck-loss-weight 0.0 \
    --night-sample-weight 2.0 --occlusion-sample-weight 1.5 --hard-sample-weight 0.2 \
    --small-sample-weight 1.5 --color-sample-weight 0.0 --pseudo-label-weight 1.0 \
    --init-checkpoint '$INIT' --run-kind formal \
    --dataset-version attribute-domain-v2-stage136-mio-road-domain-axle-r1 \
    --code-revision stage137-mio-road-domain-head-tune-r1 --skip-test \
    --output-dir '$RUN' >'$LOG' 2>&1"
echo "started tmux:$SESSION"
echo "run=$RUN"
echo "log=$LOG"
