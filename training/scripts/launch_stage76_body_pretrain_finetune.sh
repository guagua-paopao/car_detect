#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code"
RUNNER="$CODE/training/scripts/run_stage76_body_pretrain_finetune.py"
TEST="$CODE/training/tests/test_run_stage76_body_pretrain_finetune.py"
MATRIX="$CODE/training/stage76-body-pretrain-finetune-matrix-v1.json"
OUTPUT_ROOT="$BASE/runs/attributes/ATTR-STAGE76-BODY-PRETRAIN-FINETUNE-V1"
STATE="$OUTPUT_ROOT.state.json"
PREFLIGHT_LOG="$OUTPUT_ROOT.preflight.log"
RUNNER_LOG="$OUTPUT_ROOT.runner.log"
SESSION=VCAS-STAGE76-BODY-PRETRAIN-FINETUNE

assert_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || { echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2; exit 64; }
}

for required in "$PY" "$RUNNER" "$TEST" "$MATRIX"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$RUNNER" 50f2b1bed03c5db54ffdbe0be90f0ea8a660b80df95bd6cf5d57a6f6cb83ef2c
assert_sha256 "$TEST" 30d11d2be7000558e7553a874e473490d0c6550c825c23c757e5a7906a12e380
assert_sha256 "$MATRIX" 871d284c3aea1513f0032555036032adf38d94434140cec7df9a92982d10fd9a

"$PY" "$TEST"
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing to overwrite: $OUTPUT_ROOT" >&2; exit 66; }
[[ ! -e "$STATE" ]] || { echo "refusing to overwrite: $STATE" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi

tmux new-session -d -s "$SESSION" \
  "set +e; '$PY' '$RUNNER' --matrix '$MATRIX' --output-root '$OUTPUT_ROOT' --state '$STATE' \
    --device cuda --code-revision stage76-body-pretrain-finetune-v1 --preflight-only >'$PREFLIGHT_LOG' 2>&1; \
    rc=\$?; \
    if [[ \$rc -eq 0 ]]; then \
      '$PY' '$RUNNER' --matrix '$MATRIX' --output-root '$OUTPUT_ROOT' --state '$STATE' \
        --device cuda --code-revision stage76-body-pretrain-finetune-v1 >'$RUNNER_LOG' 2>&1; rc=\$?; \
    else \
      '$PY' -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({\"status\":\"failed_closed_preflight\",\"exit_code\":int(sys.argv[2]),\"preflight_log\":sys.argv[3],\"test_accessed\":False,\"frozen_video_used\":False,\"production_model_modified\":False,\"deployment_performed\":False},indent=2)+\"\\n\",encoding=\"utf-8\")' '$STATE' \"\$rc\" '$PREFLIGHT_LOG'; \
    fi; \
    exit \"\$rc\""

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "preflight_log=$PREFLIGHT_LOG"
echo "runner_log=$RUNNER_LOG"
