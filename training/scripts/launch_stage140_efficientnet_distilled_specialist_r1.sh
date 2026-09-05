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
TEACHER="$BASE/runs/attributes/ATTR-STAGE121-AXLE-SEMANTIC-CLEAN-R1/ATTR-STAGE121-TRUCK-SUBTYPE-CONVNEXT-256-AXLE-CLEAN-R1/best.pt"
REJECTED_STATE="$BASE/runs/attributes/ATTR-STAGE139-INTEGRATED-COMPONENT-GATE-R1.state.json"
ROOT="$BASE/runs/attributes/ATTR-STAGE140-EFFICIENTNET-DISTILLED-SPECIALIST-R1"
RUN="$ROOT/ATTR-STAGE140-TRUCK-SUBTYPE-EFFICIENTNETV2S-320-DISTILLED-R1"
LOG="$ROOT/train.log"
SESSION=VCAS-STAGE140-EFFICIENTNET-DISTILLED-SPECIALIST-R1

assert_sha256() { local path="$1" expected="$2" actual; actual="$(sha256sum "$path" | awk '{print tolower($1)}')"; [[ "$actual" == "$expected" ]] || { echo "SHA256 mismatch: $path" >&2; exit 64; }; }

for required in "$PY" "$TRAIN" "$LABELS" "$MANIFEST" "$REPORT" "$TEACHER" "$REJECTED_STATE"; do [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }; done
assert_sha256 "$TRAIN" b1a7f8b5b67b15ff4466ead9c3256285415230c13da3b7b4aaebe8bdf5341ae6
assert_sha256 "$LABELS" 2bbb5f10bd3f7377e9a43d827ba743dc4d0407c25e17254afa4cb8d64ad8cc46
assert_sha256 "$MANIFEST" a12be8b1cc38fa777bfd219b2a39f74b59092347c7e4b63b0d6e97509007645d
assert_sha256 "$REPORT" ed1e8c01e75fb13b7d5294d9fe8a3aa363d9f96aafbea833b989fda661823c7d
assert_sha256 "$TEACHER" fa80144ffc84886188d8268935f5cdadb5d74cda3af170b44cf80a4d3f060545
assert_sha256 "$REJECTED_STATE" 2a028156ab0e9a93879d1a9410987589a6c93f2fbba1fe9adef9e068ae4285ef
"$PY" - "$REPORT" "$REJECTED_STATE" <<'PY'
import json,sys
report=json.load(open(sys.argv[1],encoding='utf-8')); rejected=json.load(open(sys.argv[2],encoding='utf-8'))
assert report['status']=='pass' and report['policy']['research_only'] is True
assert report['integrity']['post_split_leaks']=={'exact':0,'near':0,'group':0}
assert report['policy']['test_accessed'] is False and report['policy']['frozen_video_used'] is False
assert rejected['status']=='component_repair_required_fail_closed'
assert rejected['independent_test_authorized'] is False
assert rejected['test_accessed'] is False and rejected['frozen_video_used'] is False
PY
"$PY" -m py_compile "$TRAIN"
[[ ! -e "$ROOT" ]] || { echo "refusing to overwrite Stage140" >&2; exit 66; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists" >&2; exit 67; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then echo "GPU is occupied" >&2; exit 68; fi
mkdir -p "$ROOT"
tmux new-session -d -s "$SESSION" \
  "'$PY' '$TRAIN' \
    --manifest '$MANIFEST' --labels '$LABELS' \
    --input-size 320 --architecture efficientnet_v2_s --resize-mode stretch \
    --epochs 8 --batch-size 16 --workers 8 --learning-rate 0.00005 --weight-decay 0.0001 \
    --body-loss-weight 1.0 --color-loss-weight 0.0 --focal-gamma 2.0 --color-focal-gamma 0.0 \
    --class-weighting none --label-smoothing 0.01 --gradient-clip-norm 5.0 \
    --freeze-backbone-epochs 2 --patience 4 \
    --type-threshold 0.80 --color-threshold 0.99 --gate-type-precision 0.95 --gate-type-coverage 0.65 \
    --seed 20260909 --augmentation-profile hard_scene --selection-head body \
    --body-hierarchy none --coarse-car-loss-weight 0.0 --coarse-truck-loss-weight 0.0 \
    --night-sample-weight 2.0 --occlusion-sample-weight 1.5 --hard-sample-weight 0.2 \
    --small-sample-weight 1.5 --color-sample-weight 0.0 --pseudo-label-weight 1.0 \
    --body-teacher-checkpoint '$TEACHER' --distill-weight 0.20 --distill-temperature 2.0 \
    --distill-body-weight 1.0 --distill-color-weight 0.0 \
    --run-kind formal --dataset-version attribute-domain-v2-stage136-mio-road-domain-axle-r1 \
    --code-revision stage140-efficientnet-distilled-specialist-r1 --skip-test \
    --output-dir '$RUN' >'$LOG' 2>&1"
echo "started tmux:$SESSION"; echo "run=$RUN"; echo "log=$LOG"
