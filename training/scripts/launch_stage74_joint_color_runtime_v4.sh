#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/bin/python
CODE_ROOT=/root/autodl-tmp/vcas/code
DATA_ROOT=/root/autodl-tmp/vcas/datasets/attribute-domain-v2
SCRIPT="$CODE_ROOT/training/scripts/normalize_stage74_joint_color_runtime_paths.py"
INPUT_MANIFEST="$DATA_ROOT/stage74-joint-color-v3/attribute_manifest.stage74-joint-color.csv"
INPUT_REPORT="$DATA_ROOT/stage74-joint-color-v3/report.json"
BASE_MANIFEST="$DATA_ROOT/stage71-teacher-manifests-v5/attribute_manifest.stage71-color-teacher.csv"
UA_ROOT="$DATA_ROOT/stage70-uadetrac-training-pool-v2"
BMD_ROOT="$DATA_ROOT"
SAFETY_ROOT=/root/autodl-tmp/vcas/datasets
OUTPUT_ROOT="$DATA_ROOT/stage74-joint-color-runtime-v4"
OUTPUT_MANIFEST="$DATA_ROOT/stage71-teacher-manifests-v5/attribute_manifest.stage74-joint-color-runtime-v4.csv"
OUTPUT_REPORT="$OUTPUT_ROOT/report.json"
STATE="$OUTPUT_ROOT/state.json"
LOG="$OUTPUT_ROOT/run.log"
SESSION=VCAS-STAGE74-RUNTIME-NORM-V4

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

for required in "$PYTHON" "$SCRIPT" "$INPUT_MANIFEST" "$INPUT_REPORT" "$BASE_MANIFEST"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$SCRIPT" AB6905A2BDB542DE33D55E94067AF859F787FF2B8879328D7BF9709DD94AB1B3
assert_sha256 "$INPUT_MANIFEST" 735985DB2F46F60C42CBADFEBF3C2644DF8B8A244EE7D0B6BFA7F690F1DBCF09
assert_sha256 "$INPUT_REPORT" D85D0069E8E1A67143427C66EC0C3CFDD0D528AC8A5A5DA98A1AE329DF4F8219
assert_sha256 "$BASE_MANIFEST" F4576A85019C99A3E287692AEEBC1F470E21113E26A58DB264A2703F502D8A63

for output in "$OUTPUT_ROOT" "$OUTPUT_MANIFEST"; do
  [[ ! -e "$output" ]] || { echo "refusing to overwrite Stage74 V4 runtime evidence: $output" >&2; exit 66; }
done
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi

mkdir -p "$OUTPUT_ROOT"
tmux new-session -d -s "$SESSION" \
  "set +e; '$PYTHON' '$SCRIPT' \
    --input-manifest '$INPUT_MANIFEST' \
    --expected-input-manifest-sha256 735985DB2F46F60C42CBADFEBF3C2644DF8B8A244EE7D0B6BFA7F690F1DBCF09 \
    --input-report '$INPUT_REPORT' \
    --expected-input-report-sha256 D85D0069E8E1A67143427C66EC0C3CFDD0D528AC8A5A5DA98A1AE329DF4F8219 \
    --base-manifest '$BASE_MANIFEST' \
    --expected-base-manifest-sha256 F4576A85019C99A3E287692AEEBC1F470E21113E26A58DB264A2703F502D8A63 \
    --ua-root '$UA_ROOT' \
    --bmd-root '$BMD_ROOT' \
    --datasets-safety-root '$SAFETY_ROOT' \
    --output-manifest '$OUTPUT_MANIFEST' \
    --output-report '$OUTPUT_REPORT' >'$LOG' 2>&1; \
    rc=\$?; \
    report_sha=''; manifest_sha=''; \
    if [[ -f '$OUTPUT_REPORT' ]]; then report_sha=\$(sha256sum '$OUTPUT_REPORT' | awk '{print toupper(\$1)}'); printf '%s  %s\n' \"\$report_sha\" report.json >'$OUTPUT_REPORT.sha256'; fi; \
    if [[ -f '$OUTPUT_MANIFEST' ]]; then manifest_sha=\$(sha256sum '$OUTPUT_MANIFEST' | awk '{print toupper(\$1)}'); printf '%s  %s\n' \"\$manifest_sha\" \"\$(basename '$OUTPUT_MANIFEST')\" >'$OUTPUT_MANIFEST.sha256'; fi; \
    '$PYTHON' -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({\"status\":\"pass_runtime_paths_normalized\" if int(sys.argv[2]) == 0 else \"failed_closed\",\"exit_code\":int(sys.argv[2]),\"report_sha256\":sys.argv[3],\"manifest_sha256\":sys.argv[4],\"test_accessed\":False,\"frozen_video_used\":False,\"production_model_modified\":False,\"deployment_performed\":False},indent=2)+\"\\n\",encoding=\"utf-8\")' '$STATE' \"\$rc\" \"\$report_sha\" \"\$manifest_sha\"; \
    exit \"\$rc\""

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "log=$LOG"
