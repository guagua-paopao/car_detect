#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code"
RUNNER="$CODE/training/scripts/run_stage77_body_validation_after_stage76.py"
EVALUATOR="$CODE/training/scripts/evaluate_attribute_baseline.py"
TRACK_SWEEPER="$CODE/training/scripts/sweep_attribute_track_fusion_fail_closed_v2.py"
LABELS="$CODE/config/vehicle_labels.v1.json"
VFG_MANIFEST="$BASE/datasets/attribute-domain-v2/vfg7-eval-v1/attribute_manifest.csv"
HARD_MANIFEST="$BASE/datasets/attribute-domain-v2/attribute_manifest.bmd-raw-v34-hard-v2.csv"
PRODUCTION="$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
STAGE71="$BASE/runs/attributes/ATTR-STAGE71-CONVNEXT-TEACHERS-V1/ATTR-STAGE71-BODY-CONVNEXT-256-REAL-CCTV/best.pt"
TRAINING_STATE="$BASE/runs/attributes/ATTR-STAGE76-BODY-PRETRAIN-FINETUNE-V1.state.json"
CANDIDATE="$BASE/runs/attributes/ATTR-STAGE76-BODY-PRETRAIN-FINETUNE-V1/ATTR-STAGE76-BODY-CONVNEXT-256-CCTV-FINETUNE/best.pt"
OUTPUT_ROOT="$BASE/runs/attributes/ATTR-STAGE77-BODY-VALIDATION-V1"
STATE="$OUTPUT_ROOT.state.json"
LOG="$OUTPUT_ROOT.log"
SESSION=VCAS-STAGE77-BODY-VALIDATION

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

for required in \
  "$PY" "$RUNNER" "$EVALUATOR" "$TRACK_SWEEPER" "$LABELS" \
  "$VFG_MANIFEST" "$HARD_MANIFEST" "$PRODUCTION" "$STAGE71" "$TRAINING_STATE"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done

assert_sha256 "$RUNNER" 8987fb933c787f7245e2e8bcf04642cf7f518c200bdbc9748c3859f4a30bd9f5
assert_sha256 "$EVALUATOR" 974a684d34dc45e4572d8efa5e703bee2de486f3b8ba44649a1909bef37ef2bb
assert_sha256 "$TRACK_SWEEPER" e3dcfa41d3d2e0a787eae77824309c6cca5641c8dd413ac55136dd585da28e5d
assert_sha256 "$LABELS" c71675a0e2de950880f76f088c108c6fb286561a2399cb5a182055289fb5eb4f
assert_sha256 "$VFG_MANIFEST" 5582a65fd02874b6e84776d9e00a883d4ad0cb4fa579ac10c829583ec751bce3
assert_sha256 "$HARD_MANIFEST" 51cbef8aa078579b67435908bfaf42a31efce47e79045c2552a4a4ec3d920c40
assert_sha256 "$PRODUCTION" 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383
assert_sha256 "$STAGE71" ec51e835eeb406aa7e0af27c4a6b055c2036d2f4ffd01c912218b6c4373e13af

[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing to overwrite: $OUTPUT_ROOT" >&2; exit 66; }
[[ ! -e "$STATE" ]] || { echo "refusing to overwrite: $STATE" >&2; exit 66; }
[[ ! -e "$LOG" ]] || { echo "refusing to overwrite: $LOG" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi

tmux new-session -d -s "$SESSION" \
  "'$PY' '$RUNNER' \
    --training-state '$TRAINING_STATE' \
    --candidate-checkpoint '$CANDIDATE' \
    --output-root '$OUTPUT_ROOT' \
    --state '$STATE' \
    --evaluator '$EVALUATOR' \
    --track-sweeper '$TRACK_SWEEPER' \
    --labels '$LABELS' \
    --vfg-manifest '$VFG_MANIFEST' \
    --hard-manifest '$HARD_MANIFEST' \
    --production-checkpoint '$PRODUCTION' \
    --stage71-checkpoint '$STAGE71' \
    --device cuda --poll-seconds 30 --timeout-seconds 21600 >'$LOG' 2>&1"

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "log=$LOG"
