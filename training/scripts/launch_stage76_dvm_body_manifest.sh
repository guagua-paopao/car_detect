#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code"
SCRIPT="$CODE/training/scripts/build_stage76_dvm_body_pretrain_manifest.py"
TEST="$CODE/training/tests/test_build_stage76_dvm_body_pretrain_manifest.py"
DATASET_ROOT="$BASE/datasets/attribute-domain-v2"
INPUT="$DATASET_ROOT/stage68-dvm-color-pool-v1/attribute_manifest.audit-clean-v2.csv"
ARCHIVE="$BASE/sources/color-scale-stage68/dvm-car-cc-by-nc/archive.zip"
LABELS="$CODE/config/vehicle_labels.v1.json"
OUTPUT_ROOT="$DATASET_ROOT/stage76-dvm-body-pretrain-v1"
MANIFEST="$OUTPUT_ROOT/attribute_manifest.stage76-dvm-body-pretrain.csv"
REPORT="$OUTPUT_ROOT/stage76-dvm-body-pretrain-report.json"
STATE="$OUTPUT_ROOT.state.json"
LOG="$OUTPUT_ROOT.log"
SESSION=VCAS-STAGE76-DVM-BODY-MANIFEST

assert_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || { echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2; exit 64; }
}

for required in "$PY" "$SCRIPT" "$TEST" "$INPUT" "$ARCHIVE" "$LABELS"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$SCRIPT" 259190e70d1870092a1f911b1ed714b6a66b39b12048bbcb39679cd8f07d9399
assert_sha256 "$TEST" c0adcabb099eb2a509b2c8f19e9d9dceb3ca1e10592b1c4eb0a535eaa8db572b
assert_sha256 "$INPUT" 9c7bc1296e7c5d2caa7cabfc8d34b031e1dbdbccb9be355ce36b41d7d9f0910b
assert_sha256 "$ARCHIVE" 6a39bcb0e4f95fe7319a9eb798c61429f383b32a0d5b5adba35ac9edde8d79db
assert_sha256 "$LABELS" c71675a0e2de950880f76f088c108c6fb286561a2399cb5a182055289fb5eb4f

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
    --input-manifest '$INPUT' --expected-input-sha256 9c7bc1296e7c5d2caa7cabfc8d34b031e1dbdbccb9be355ce36b41d7d9f0910b \
    --archive '$ARCHIVE' --expected-archive-sha256 6a39bcb0e4f95fe7319a9eb798c61429f383b32a0d5b5adba35ac9edde8d79db \
    --labels '$LABELS' --expected-labels-sha256 c71675a0e2de950880f76f088c108c6fb286561a2399cb5a182055289fb5eb4f \
    --dataset-root '$DATASET_ROOT' --output-manifest '$MANIFEST' --output-report '$REPORT' >'$LOG' 2>&1; \
    rc=\$?; manifest_sha=''; report_sha=''; \
    if [[ -f '$MANIFEST' ]]; then manifest_sha=\$(sha256sum '$MANIFEST' | awk '{print tolower(\$1)}'); printf '%s  %s\n' \"\$manifest_sha\" \"\$(basename '$MANIFEST')\" >'$MANIFEST.sha256'; fi; \
    if [[ -f '$REPORT' ]]; then report_sha=\$(sha256sum '$REPORT' | awk '{print tolower(\$1)}'); printf '%s  %s\n' \"\$report_sha\" \"\$(basename '$REPORT')\" >'$REPORT.sha256'; fi; \
    '$PY' -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({\"status\":\"complete_manifest_pass\" if int(sys.argv[2]) == 0 else \"fail_closed\",\"exit_code\":int(sys.argv[2]),\"manifest\":sys.argv[3],\"manifest_sha256\":sys.argv[4],\"report\":sys.argv[5],\"report_sha256\":sys.argv[6],\"test_accessed\":False,\"frozen_video_used\":False,\"production_model_modified\":False,\"deployment_performed\":False},indent=2)+\"\\n\",encoding=\"utf-8\")' '$STATE' \"\$rc\" '$MANIFEST' \"\$manifest_sha\" '$REPORT' \"\$report_sha\"; \
    exit \"\$rc\""

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "log=$LOG"
