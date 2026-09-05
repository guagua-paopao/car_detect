#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code"
SCRIPT="$CODE/training/scripts/run_vfg7_validation_comparison.py"
STATIC="$CODE/training/scripts/evaluate_attribute_baseline.py"
TRACK="$CODE/training/scripts/evaluate_attribute_track_fusion.py"
LABELS="$CODE/config/vehicle_labels.v1.json"
MANIFEST="$BASE/datasets/attribute-domain-v2/vfg7-eval-v1/attribute_manifest.csv"
PRODUCTION="$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
STAGE71="$BASE/runs/attributes/ATTR-STAGE71-CONVNEXT-TEACHERS-V1/ATTR-STAGE71-COLOR-CONVNEXT-256-DVM-REPLAY-CCTV/best.pt"
STAGE74="$BASE/runs/attributes/ATTR-STAGE74-COLOR-CCTV-JOINT-V2/ATTR-STAGE74-COLOR-CONVNEXT-256-CCTV-JOINT-V2/best.pt"
OUTPUT_ROOT="$BASE/runs/attributes/ATTR-STAGE74-COLOR-VALIDATION-ONLY-V1"
OUT="$OUTPUT_ROOT/vfg7-comparison"
REPORT="$OUT/comparison-report.json"
LOG="$OUTPUT_ROOT/vfg7-comparison.log"
STATE="$OUTPUT_ROOT/state.json"
SESSION=VCAS-STAGE74-VFG7-COLOR-VALIDATION

assert_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print toupper($1)}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
    exit 64
  fi
}

for required in "$PY" "$SCRIPT" "$STATIC" "$TRACK" "$LABELS" "$MANIFEST" "$PRODUCTION" "$STAGE71" "$STAGE74"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$SCRIPT" 5A924C27BC927A82C79C158341E4C3188333EDD76D07F4B4CF84A050A19A6899
assert_sha256 "$STATIC" 974A684D34DC45E4572D8EFA5E703BEE2DE486F3B8BA44649A1909BEF37EF2BB
assert_sha256 "$TRACK" 20881E01AEC678E5F59F2A4EEBB851CF9C015CCCA5D4009780024A988459DA90
assert_sha256 "$LABELS" C71675A0E2DE950880F76F088C108C6FB286561A2399CB5A182055289FB5EB4F
assert_sha256 "$MANIFEST" 5582A65FD02874B6E84776D9E00A883D4AD0CB4FA579AC10C829583EC751BCE3
assert_sha256 "$PRODUCTION" 6F651BBA1C62082F13740728C96C82A074FBB3A8ECAF27FAC1F90504C6BB7383
assert_sha256 "$STAGE71" 14BF423D277085C7DECEC236F1CC3931274AD778D0101E5D839BE43648254121
assert_sha256 "$STAGE74" 0C17E979C79A9C16978EF46D628650A3D883B311599AF0C24F8B7B3A2264C581

[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing to overwrite Stage74 validation evidence: $OUTPUT_ROOT" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi
mkdir -p "$OUTPUT_ROOT"

tmux new-session -d -s "$SESSION" \
  "set +e; '$PY' '$SCRIPT' \
    --manifest '$MANIFEST' \
    --labels '$LABELS' \
    --scripts-root '$CODE/training/scripts' \
    --output-dir '$OUT' \
    --python '$PY' \
    --batch-size 128 \
    --workers 4 \
    --model 'production=$PRODUCTION' \
    --model 'stage71_color=$STAGE71' \
    --model 'stage74_color=$STAGE74' >'$LOG' 2>&1; \
    rc=\$?; report_sha=''; \
    if [[ -f '$REPORT' ]]; then report_sha=\$(sha256sum '$REPORT' | awk '{print toupper(\$1)}'); printf '%s  %s\n' \"\$report_sha\" comparison-report.json >'$REPORT.sha256'; fi; \
    '$PY' -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({\"status\":\"complete_validation_only\" if int(sys.argv[2]) == 0 else \"failed_closed_runtime\",\"exit_code\":int(sys.argv[2]),\"report\":sys.argv[3],\"report_sha256\":sys.argv[4],\"split\":\"validation\",\"test_accessed\":False,\"frozen_video_used\":False,\"production_model_modified\":False,\"backend_gates_run\":False,\"deployment_performed\":False},indent=2)+\"\\n\",encoding=\"utf-8\")' '$STATE' \"\$rc\" '$REPORT' \"\$report_sha\"; \
    exit \"\$rc\""

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "log=$LOG"
