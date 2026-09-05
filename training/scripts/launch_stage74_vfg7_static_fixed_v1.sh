#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code"
EVALUATOR="$CODE/training/scripts/evaluate_attribute_baseline.py"
LABELS="$CODE/config/vehicle_labels.v1.json"
MANIFEST="$BASE/datasets/attribute-domain-v2/vfg7-eval-v1/attribute_manifest.csv"
CHECKPOINT="$BASE/runs/attributes/ATTR-STAGE74-COLOR-CCTV-JOINT-V2/ATTR-STAGE74-COLOR-CONVNEXT-256-CCTV-JOINT-V2/best.pt"
OUTPUT_ROOT="$BASE/runs/attributes/ATTR-STAGE74-VFG7-STATIC-FIXED-V1"
REPORT="$OUTPUT_ROOT/report.json"
STATE="$OUTPUT_ROOT.state.json"
LOG="$OUTPUT_ROOT.log"
SESSION=VCAS-STAGE74-VFG7-STATIC-FIXED

assert_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || { echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2; exit 64; }
}

for required in "$PY" "$EVALUATOR" "$LABELS" "$MANIFEST" "$CHECKPOINT"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$EVALUATOR" 974a684d34dc45e4572d8efa5e703bee2de486f3b8ba44649a1909bef37ef2bb
assert_sha256 "$LABELS" c71675a0e2de950880f76f088c108c6fb286561a2399cb5a182055289fb5eb4f
assert_sha256 "$MANIFEST" 5582a65fd02874b6e84776d9e00a883d4ad0cb4fa579ac10c829583ec751bce3
assert_sha256 "$CHECKPOINT" 0c17e979c79a9c16978ef46d628650a3d883b311599af0c24f8b7b3a2264c581

[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing to overwrite: $OUTPUT_ROOT" >&2; exit 66; }
[[ ! -e "$STATE" ]] || { echo "refusing to overwrite: $STATE" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi
mkdir -p "$OUTPUT_ROOT"

tmux new-session -d -s "$SESSION" \
  "set +e; '$PY' '$EVALUATOR' \
    --manifest '$MANIFEST' --checkpoint '$CHECKPOINT' --labels '$LABELS' \
    --output '$REPORT' --split validation --batch-size 128 --workers 4 --device cuda \
    --type-threshold 0.96 --color-threshold 0.891 >'$LOG' 2>&1; \
    rc=\$?; report_sha=''; \
    if [[ -f '$REPORT' ]]; then report_sha=\$(sha256sum '$REPORT' | awk '{print tolower(\$1)}'); printf '%s  %s\n' \"\$report_sha\" report.json >'$REPORT.sha256'; fi; \
    '$PY' -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({\"status\":\"complete_fixed_threshold_validation\" if int(sys.argv[2]) == 0 else \"failed_closed_runtime\",\"exit_code\":int(sys.argv[2]),\"selected_on\":\"VFG-7 validation fine sweep\",\"fixed_color_threshold\":0.891,\"split\":\"validation\",\"threshold_sweep\":False,\"test_accessed\":False,\"frozen_video_used\":False,\"production_model_modified\":False,\"backend_gates_run\":False,\"deployment_performed\":False,\"report\":sys.argv[3],\"report_sha256\":sys.argv[4]},indent=2)+\"\\n\",encoding=\"utf-8\")' '$STATE' \"\$rc\" '$REPORT' \"\$report_sha\"; \
    exit \"\$rc\""

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "log=$LOG"
