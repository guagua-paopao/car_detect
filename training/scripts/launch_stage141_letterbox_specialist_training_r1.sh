#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage141_letterbox_r1"
TRAIN="$CODE/scripts/train_attribute.py"
HIERARCHY="$CODE/src/attribute_hierarchy.py"
MODEL="$CODE/src/multitask_mobilenet_v3.py"
COMMON="$CODE/src/common.py"
LETTERBOX_TEST="$CODE/tests/test_attribute_letterbox_geometry.py"
SHARED_TEST="$CODE/tests/test_evaluate_v2_decoupled_shared_validation.py"
LABELS="$BASE/code/training/config/vehicle_labels.truck-subtype.v1.json"
DATA="$BASE/datasets/attribute-domain-v2/stage136-mio-road-domain-axle-r1"
MANIFEST="$DATA/attribute_manifest.stage136-mio-road-domain-axle.csv"
REPORT="$DATA/stage136-mio-road-domain-axle-report.json"
INIT="$BASE/runs/attributes/ATTR-STAGE121-AXLE-SEMANTIC-CLEAN-R1/ATTR-STAGE121-TRUCK-SUBTYPE-CONVNEXT-256-AXLE-CLEAN-R1/best.pt"
ROOT="$BASE/runs/attributes/ATTR-STAGE141-LETTERBOX-SPECIALIST-R1"
RUN="$ROOT/ATTR-STAGE141-TRUCK-SUBTYPE-CONVNEXT-256-LETTERBOX-R1"
LOG="$ROOT/train.log"
SESSION=VCAS-STAGE141-LETTERBOX-SPECIALIST-R1

assert_sha256() { local path="$1" expected="$2" actual; actual="$(sha256sum "$path" | awk '{print tolower($1)}')"; [[ "$actual" == "$expected" ]] || { echo "SHA256 mismatch: $path" >&2; exit 64; }; }
for required in "$PY" "$TRAIN" "$HIERARCHY" "$MODEL" "$COMMON" "$LETTERBOX_TEST" "$SHARED_TEST" "$LABELS" "$MANIFEST" "$REPORT" "$INIT"; do [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }; done
assert_sha256 "$TRAIN" ca31774366304d04e43425e38ce5d209ac85b8732d6aaa706ba66abe1bc6c6a1
assert_sha256 "$HIERARCHY" 1ffe558a91091ce54ccf885870a5aafb19bdbd912bec8a4935b7a49564f5428f
assert_sha256 "$MODEL" a66e1fc94a538cd29cebcf565d0e8a0e8a694be47e62572e5f25c038e6319bb8
assert_sha256 "$COMMON" a7be6dad0629cd0e4cf50955731558339351b8653bf681c4b7874592e2713a1d
assert_sha256 "$LETTERBOX_TEST" 7ddd3751803b5c1ac4c334384d0473aad18731bf103a4a422b6f87b958063561
assert_sha256 "$SHARED_TEST" f20bb6846bf7aa507d9761d948d665ca2cee0da81b773f4db4e4253e4f49f3ea
assert_sha256 "$LABELS" 2bbb5f10bd3f7377e9a43d827ba743dc4d0407c25e17254afa4cb8d64ad8cc46
assert_sha256 "$MANIFEST" a12be8b1cc38fa777bfd219b2a39f74b59092347c7e4b63b0d6e97509007645d
assert_sha256 "$REPORT" ed1e8c01e75fb13b7d5294d9fe8a3aa363d9f96aafbea833b989fda661823c7d
assert_sha256 "$INIT" fa80144ffc84886188d8268935f5cdadb5d74cda3af170b44cf80a4d3f060545
"$PY" - "$REPORT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
assert d['status']=='pass' and d['policy']['research_only'] is True
assert d['integrity']['post_split_leaks']=={'exact':0,'near':0,'group':0}
assert d['policy']['test_accessed'] is False and d['policy']['frozen_video_used'] is False
PY
"$PY" -m py_compile "$TRAIN" "$HIERARCHY" "$MODEL" "$LETTERBOX_TEST"
cd "$CODE"
PYTHONPATH="$CODE/scripts" "$PY" -m unittest discover -s tests -p 'test_attribute_letterbox_geometry.py' -q
PYTHONPATH="$CODE/scripts" "$PY" -m unittest discover -s tests -p 'test_evaluate_v2_decoupled_shared_validation.py' -q
[[ ! -e "$ROOT" ]] || { echo "refusing to overwrite Stage141" >&2; exit 66; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists" >&2; exit 67; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then echo "GPU is occupied" >&2; exit 68; fi
mkdir -p "$ROOT"
tmux new-session -d -s "$SESSION" \
  "'$PY' '$TRAIN' \
    --manifest '$MANIFEST' --labels '$LABELS' \
    --input-size 256 --architecture convnext_tiny --resize-mode letterbox \
    --epochs 8 --batch-size 40 --workers 8 --learning-rate 0.000003 --weight-decay 0.0001 \
    --body-loss-weight 1.0 --color-loss-weight 0.0 --focal-gamma 2.0 --color-focal-gamma 0.0 \
    --class-weighting none --label-smoothing 0.01 --gradient-clip-norm 5.0 \
    --freeze-backbone-epochs 1 --patience 4 \
    --type-threshold 0.80 --color-threshold 0.99 --gate-type-precision 0.95 --gate-type-coverage 0.65 \
    --seed 20260910 --augmentation-profile hard_scene --selection-head body \
    --body-hierarchy none --coarse-car-loss-weight 0.0 --coarse-truck-loss-weight 0.0 \
    --night-sample-weight 2.0 --occlusion-sample-weight 1.5 --hard-sample-weight 0.2 \
    --small-sample-weight 1.5 --color-sample-weight 0.0 --pseudo-label-weight 1.0 \
    --init-checkpoint '$INIT' --run-kind formal \
    --dataset-version attribute-domain-v2-stage136-mio-road-domain-letterbox-r1 \
    --code-revision stage141-letterbox-specialist-r1 --skip-test \
    --output-dir '$RUN' >'$LOG' 2>&1"
echo "started tmux:$SESSION"; echo "run=$RUN"; echo "log=$LOG"
