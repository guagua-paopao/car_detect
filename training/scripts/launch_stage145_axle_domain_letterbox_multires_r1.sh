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
DATA="$BASE/datasets/attribute-domain-v2/stage120-axle-semantic-clean-r1"
MANIFEST="$DATA/attribute_manifest.stage120-axle-semantic-clean.csv"
REPORT="$DATA/stage120-axle-semantic-clean-report.json"
INIT="$BASE/runs/attributes/ATTR-STAGE121-AXLE-SEMANTIC-CLEAN-R1/ATTR-STAGE121-TRUCK-SUBTYPE-CONVNEXT-256-AXLE-CLEAN-R1/best.pt"
STAGE144_STATE="$BASE/runs/attributes/ATTR-STAGE144-LETTERBOX-ROUTER-DIAGNOSIS-R1.state.json"
ROOT="$BASE/runs/attributes/ATTR-STAGE145-AXLE-DOMAIN-LETTERBOX-MULTIRES-R1"
LOG="$ROOT.log"
SESSION=VCAS-STAGE145-AXLE-DOMAIN-LETTERBOX-MULTIRES-R1

assert_sha256() {
  local path="$1" expected="$2" actual
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 65; }
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "${expected,,}" ]] || { echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2; exit 65; }
}

train_variant() {
  local size="$1" batch="$2"
  local run="$ROOT/ATTR-STAGE145-TRUCK-SUBTYPE-CONVNEXT-${size}-AXLE-LETTERBOX-R1"
  "$PY" "$TRAIN" \
    --manifest "$MANIFEST" --labels "$LABELS" \
    --input-size "$size" --architecture convnext_tiny --resize-mode letterbox \
    --epochs 8 --batch-size "$batch" --workers 8 --learning-rate 0.000001 --weight-decay 0.0001 \
    --body-loss-weight 1.0 --color-loss-weight 0.0 --focal-gamma 2.0 --color-focal-gamma 0.0 \
    --class-weighting none --label-smoothing 0.01 --gradient-clip-norm 5.0 \
    --freeze-backbone-epochs 1 --patience 4 \
    --type-threshold 0.80 --color-threshold 0.99 --gate-type-precision 0.95 --gate-type-coverage 0.65 \
    --seed 20260911 --augmentation-profile hard_scene --selection-head body \
    --body-hierarchy none --coarse-car-loss-weight 0.0 --coarse-truck-loss-weight 0.0 \
    --night-sample-weight 2.0 --occlusion-sample-weight 1.5 --hard-sample-weight 0.2 \
    --small-sample-weight 1.5 --color-sample-weight 0.0 --pseudo-label-weight 1.0 \
    --init-checkpoint "$INIT" --run-kind formal \
    --dataset-version "attribute-domain-v2-stage120-axle-semantic-clean-letterbox-${size}-r1" \
    --code-revision stage145-axle-domain-letterbox-multires-r1 --skip-test \
    --output-dir "$run" >"$ROOT/train-${size}.log" 2>&1
}

worker() {
  assert_sha256 "$TRAIN" ca31774366304d04e43425e38ce5d209ac85b8732d6aaa706ba66abe1bc6c6a1
  assert_sha256 "$HIERARCHY" 1ffe558a91091ce54ccf885870a5aafb19bdbd912bec8a4935b7a49564f5428f
  assert_sha256 "$MODEL" a66e1fc94a538cd29cebcf565d0e8a0e8a694be47e62572e5f25c038e6319bb8
  assert_sha256 "$COMMON" a7be6dad0629cd0e4cf50955731558339351b8653bf681c4b7874592e2713a1d
  assert_sha256 "$LETTERBOX_TEST" 402327c27103b213e223018255c9b677e10e1aee1f6b6d4bc6d0c3dc28e884ac
  assert_sha256 "$SHARED_TEST" f20bb6846bf7aa507d9761d948d665ca2cee0da81b773f4db4e4253e4f49f3ea
  assert_sha256 "$LABELS" 2bbb5f10bd3f7377e9a43d827ba743dc4d0407c25e17254afa4cb8d64ad8cc46
  assert_sha256 "$MANIFEST" ed2614c9a0acf615452d90730a94cb84f094c72387244f63be938ef653c33c67
  assert_sha256 "$REPORT" 30a4a8e3a2b3209da6ff5c336490a6c74021812d949d8f7eb77ee1f77ed7228d
  assert_sha256 "$INIT" fa80144ffc84886188d8268935f5cdadb5d74cda3af170b44cf80a4d3f060545
  assert_sha256 "$STAGE144_STATE" f6cbddbc31a520966a5e33f9ecb699b67f8e76665dd6a41d6d5abe6a71a68397
  "$PY" - "$REPORT" "$STAGE144_STATE" <<'PY'
import json,sys
report=json.load(open(sys.argv[1],encoding='utf-8'))
diagnosis=json.load(open(sys.argv[2],encoding='utf-8'))
assert report['status']=='pass'
assert report['output']['train_counts']=={'heavy_truck':2647,'light_truck':21852}
assert report['output']['semantic_role_counts']=={'bmd_lcv_light':16280,'inatrc_heavy':2647,'inatrc_light':5572}
assert report['integrity']['post_split_leaks']=={'exact':0,'near':0,'group':0}
assert report['policy']['generic_truck_never_supervises_heavy_subtype'] is True
assert report['policy']['uncertain_subtype_rows_excluded_not_coerced'] is True
assert report['policy']['test_accessed'] is False and report['policy']['frozen_video_used'] is False
assert diagnosis['status']=='complete_validation_only_error_attribution'
assert diagnosis['decision']=='plan_domain_targeted_repair_without_lowering_gates'
assert diagnosis['test_accessed'] is False and diagnosis['frozen_video_used'] is False
PY
  cd "$CODE"
  PYTHONPATH="$CODE/scripts" "$PY" -m unittest discover -s tests -p 'test_attribute_letterbox_geometry.py' -q
  PYTHONPATH="$CODE/scripts" "$PY" -m unittest discover -s tests -p 'test_evaluate_v2_decoupled_shared_validation.py' -q
  mkdir -p "$ROOT"
  train_variant 256 40
  train_variant 288 32
}

if [[ "${1:-}" == "--worker" ]]; then worker; exit; fi
[[ ! -e "$ROOT" && ! -e "$LOG" ]] || { echo "refusing to overwrite Stage145" >&2; exit 66; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists" >&2; exit 67; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
  echo "GPU is occupied" >&2
  exit 68
fi
tmux new-session -d -s "$SESSION" "bash '$0' --worker >'$LOG' 2>&1"
echo "started tmux:$SESSION"
echo "root=$ROOT"
echo "log=$LOG"
