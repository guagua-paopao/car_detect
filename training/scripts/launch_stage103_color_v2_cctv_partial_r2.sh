#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage102_partial_color_r1"
TRAIN="$CODE/scripts/train_attribute.py"
DATASET_CODE="$CODE/src/attribute_dataset.py"
HIERARCHY_CODE="$CODE/src/attribute_hierarchy.py"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
MANIFEST="$BASE/datasets/attribute-domain-v2/stage102-v2-partial-color-r1/attribute_manifest.stage102-v2-partial-color.csv"
MANIFEST_REPORT="$BASE/datasets/attribute-domain-v2/stage102-v2-partial-color-r1/stage102-v2-partial-color-report.json"
SOURCE_AUDIT="$BASE/datasets/attribute-domain-v2/stage102-v2-partial-color-r1/stage74-source-provenance-audit.json"
INIT="$BASE/runs/attributes/ATTR-STAGE80-COLOR-CONVNEXT-256-V2-DVM-R3/best.pt"
ROOT="$BASE/runs/attributes/ATTR-STAGE103-COLOR-V2-CCTV-PARTIAL-R2"
RUN="$ROOT/ATTR-STAGE103-COLOR-CONVNEXT-256-V2-CCTV-PARTIAL-R2"
LOG="$ROOT/train.log"
SESSION=VCAS-STAGE103-COLOR-PARTIAL-R2

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

for required in "$PY" "$TRAIN" "$DATASET_CODE" "$HIERARCHY_CODE" "$LABELS" "$MANIFEST" "$MANIFEST_REPORT" "$SOURCE_AUDIT" "$INIT"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$TRAIN" 65cf1c9bd3ef9c2c0648ff5f8b63bbb47959bb885322af728be9567a1eed6f16
assert_sha256 "$DATASET_CODE" a7cebd29288f7beb627a6ec12777df88628935bd496e7611a732e6b1d953609b
assert_sha256 "$HIERARCHY_CODE" 1ffe558a91091ce54ccf885870a5aafb19bdbd912bec8a4935b7a49564f5428f
assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
assert_sha256 "$MANIFEST" aac74a71cdbfb9ace020654973ecc443537b67ef30d6135f2fe321caf3d1d21c
assert_sha256 "$MANIFEST_REPORT" 07f629eb73c04757ef64580493c7b7e883676f7488eeabce84ebe00a43da283e
assert_sha256 "$SOURCE_AUDIT" 9e1510aa4bc1c5ab19d10a892f7c74a172a22006a5fd5c7d7df26fd8d1347e57
assert_sha256 "$INIT" a2894e21060a09c690756d83bb776ffe71acda56da9f2760df577ce080c00c47

"$PY" -c 'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); assert r["status"]=="pass"; assert r["rows"]["base_train"]==118338; assert r["rows"]["base_validation"]==6598; assert r["rows"]["added_cctv_train"]==1654; assert r["added_source_counts"]=={"BMD-45-RAW":931,"Open-Images-V7":723}; assert r["leakage"]["exact_train_validation"]==0; assert r["leakage"]["group_train_validation"]==0; assert r["leakage"]["near_train_validation"]==0; assert r["policy"]["test_accessed"] is False; assert r["policy"]["frozen_video_used"] is False; assert r["policy"]["merged_silver_gray_exact_class_fabricated"] is False' "$MANIFEST_REPORT"
"$PY" -c 'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); assert r["status"]=="complete_training_manifest_source_audit"; assert not r["source_combinations"]["eligible_ua_sources"]; assert r["policy"]["test_accessed"] is False; assert r["policy"]["frozen_video_used"] is False' "$SOURCE_AUDIT"
"$PY" -m py_compile "$TRAIN" "$DATASET_CODE" "$HIERARCHY_CODE"

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
    --label-smoothing 0.02 \
    --gradient-clip-norm 5.0 \
    --freeze-backbone-epochs 0 \
    --patience 3 \
    --type-threshold 0.99 \
    --color-threshold 0.70 \
    --gate-type-precision 0.93 \
    --gate-type-coverage 0.45 \
    --gate-color-precision 0.93 \
    --gate-color-coverage 0.25 \
    --seed 20260901 \
    --augmentation-profile color_scene \
    --selection-head color \
    --body-hierarchy none \
    --coarse-car-loss-weight 0.0 \
    --coarse-truck-loss-weight 0.0 \
    --coarse-color-loss-weight 0.5 \
    --night-sample-weight 1.5 \
    --occlusion-sample-weight 1.25 \
    --hard-sample-weight 0.1 \
    --small-sample-weight 1.5 \
    --color-sample-weight 0.5 \
    --pseudo-label-weight 0.5 \
    --init-checkpoint '$INIT' \
    --run-kind formal \
    --dataset-version attribute-domain-v2-stage103-color-v2-cctv-partial-r2 \
    --code-revision stage103-color-v2-cctv-partial-r2-amp-fix \
    --skip-test \
    --output-dir '$RUN' >'$LOG' 2>&1"

echo "started tmux:$SESSION"
echo "run=$RUN"
echo "log=$LOG"
