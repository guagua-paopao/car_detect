#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code"
SCRIPT="$CODE/training/scripts/evaluate_stage75_body_pseudolabel_rule.py"
TEST="$CODE/training/tests/test_evaluate_stage75_body_pseudolabel_rule.py"
DATASET_ROOT="$BASE/datasets/attribute-domain-v2"
MANIFEST="$DATASET_ROOT/vfg7-eval-v1/attribute_manifest.csv"
LABELS="$CODE/config/vehicle_labels.v1.json"
PRODUCTION="$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
EFFICIENT="$BASE/runs/stage7/ATTR-V5-EFFV2S-256-UVH26-BODY/best.pt"
RESNET="$BASE/runs/stage14/ATTR-V12-RESNET50-256-HARD/best.pt"
STAGE71="$BASE/runs/attributes/ATTR-STAGE71-CONVNEXT-TEACHERS-V1/ATTR-STAGE71-BODY-CONVNEXT-256-REAL-CCTV/best.pt"
OUTPUT_ROOT="$BASE/runs/attributes/ATTR-STAGE75-BODY-PSEUDOLABEL-RULE-VALIDATION-V2"
REPORT="$OUTPUT_ROOT/report.json"
STATE="$OUTPUT_ROOT.state.json"
LOG="$OUTPUT_ROOT.log"
SESSION=VCAS-STAGE75-BODY-RULE-VALIDATION-V2

assert_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || { echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2; exit 64; }
}

for required in "$PY" "$SCRIPT" "$TEST" "$MANIFEST" "$LABELS" "$PRODUCTION" "$EFFICIENT" "$RESNET" "$STAGE71"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$SCRIPT" 2f5ae111bf50baf634569468a46b3ae00f5c99a812f6fff67c1b1ce6395063d4
assert_sha256 "$TEST" 687527d47e3f7edfe252fa23c27e95ab39e6d5bc5a194d4630f9d033db3283c7
assert_sha256 "$MANIFEST" 5582a65fd02874b6e84776d9e00a883d4ad0cb4fa579ac10c829583ec751bce3
assert_sha256 "$LABELS" c71675a0e2de950880f76f088c108c6fb286561a2399cb5a182055289fb5eb4f
assert_sha256 "$PRODUCTION" 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383
assert_sha256 "$EFFICIENT" f48b27a83b722b087329feca17c5ffb17958b2fed063e69248616d8d44ddceaf
assert_sha256 "$RESNET" b15dc188e3928a33a85988d896828f4f5c9fd5e9f9cafef5b806785f2cb6edc0
assert_sha256 "$STAGE71" ec51e835eeb406aa7e0af27c4a6b055c2036d2f4ffd01c912218b6c4373e13af

"$PY" "$TEST"
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing to overwrite: $OUTPUT_ROOT" >&2; exit 66; }
[[ ! -e "$STATE" ]] || { echo "refusing to overwrite: $STATE" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi
mkdir -p "$OUTPUT_ROOT"

tmux new-session -d -s "$SESSION" \
  "set +e; '$PY' '$SCRIPT' \
    --manifest '$MANIFEST' --dataset-root '$DATASET_ROOT' \
    --labels '$LABELS' --expected-labels-sha256 c71675a0e2de950880f76f088c108c6fb286561a2399cb5a182055289fb5eb4f \
    --checkpoint 'production=$PRODUCTION' \
    --checkpoint 'efficient=$EFFICIENT' \
    --checkpoint 'resnet=$RESNET' \
    --checkpoint 'stage71=$STAGE71' \
    --output '$REPORT' --batch-size 128 --workers 6 --device cuda >'$LOG' 2>&1; \
    rc=\$?; report_sha=''; \
    if [[ -f '$REPORT' ]]; then report_sha=\$(sha256sum '$REPORT' | awk '{print tolower(\$1)}'); printf '%s  %s\n' \"\$report_sha\" report.json >'$REPORT.sha256'; fi; \
    '$PY' -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({\"status\":\"complete_rule_available\" if int(sys.argv[2]) == 0 else \"fail_closed_no_rule_or_runtime\",\"exit_code\":int(sys.argv[2]),\"report\":sys.argv[3],\"report_sha256\":sys.argv[4],\"split\":\"validation\",\"test_accessed\":False,\"frozen_video_used\":False,\"production_model_modified\":False,\"deployment_performed\":False},indent=2)+\"\\n\",encoding=\"utf-8\")' '$STATE' \"\$rc\" '$REPORT' \"\$report_sha\"; \
    exit \"\$rc\""

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "log=$LOG"
