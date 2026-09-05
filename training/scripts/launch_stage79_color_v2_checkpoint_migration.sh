#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code"
MIGRATOR="$CODE/training/scripts/migrate_attribute_checkpoint_taxonomy_v2.py"
LABELS="$CODE/config/vehicle_labels.v2.json"
SOURCE="$BASE/runs/attributes/ATTR-STAGE74-COLOR-CCTV-JOINT-V2/ATTR-STAGE74-COLOR-CONVNEXT-256-CCTV-JOINT-V2/best.pt"
OUTPUT_ROOT="$BASE/runs/attributes/ATTR-STAGE79-COLOR-V2-INIT-FROM-STAGE74"
CHECKPOINT="$OUTPUT_ROOT/convnext-256-v2-init.pt"
REPORT="$OUTPUT_ROOT/migration-report.json"
STATE="$OUTPUT_ROOT.state.json"
LOG="$OUTPUT_ROOT.log"
SESSION=VCAS-STAGE79-COLOR-V2-MIGRATE

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

for required in "$PY" "$MIGRATOR" "$LABELS" "$SOURCE"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$MIGRATOR" 3aa5009bef44fd9b5ddfa097ae2538a16ee2b52cb56e38976c7a9fd3462b96f2
assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
assert_sha256 "$SOURCE" 0c17e979c79a9c16978ef46d628650a3d883b311599af0c24f8b7b3a2264c581

[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing to overwrite: $OUTPUT_ROOT" >&2; exit 66; }
[[ ! -e "$STATE" ]] || { echo "refusing to overwrite: $STATE" >&2; exit 66; }
[[ ! -e "$LOG" ]] || { echo "refusing to overwrite: $LOG" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi

mkdir -p "$OUTPUT_ROOT"
tmux new-session -d -s "$SESSION" \
  "set +e; '$PY' '$MIGRATOR' \
    --input-checkpoint '$SOURCE' --labels '$LABELS' \
    --output-checkpoint '$CHECKPOINT' --output-report '$REPORT' >'$LOG' 2>&1; \
  rc=\$?; checkpoint_sha=''; report_sha=''; \
  if [[ -f '$CHECKPOINT' ]]; then checkpoint_sha=\$(sha256sum '$CHECKPOINT' | awk '{print tolower(\$1)}'); printf '%s  %s\n' \"\$checkpoint_sha\" convnext-256-v2-init.pt >'$CHECKPOINT.sha256'; fi; \
  if [[ -f '$REPORT' ]]; then report_sha=\$(sha256sum '$REPORT' | awk '{print tolower(\$1)}'); printf '%s  %s\n' \"\$report_sha\" migration-report.json >'$REPORT.sha256'; fi; \
  '$PY' -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({\"status\":\"complete_migration_only\" if int(sys.argv[2]) == 0 else \"failed_closed\",\"exit_code\":int(sys.argv[2]),\"checkpoint\":sys.argv[3],\"checkpoint_sha256\":sys.argv[4],\"report\":sys.argv[5],\"report_sha256\":sys.argv[6],\"test_accessed\":False,\"frozen_video_used\":False,\"production_model_modified\":False,\"deployment_performed\":False},indent=2)+\"\\n\",encoding=\"utf-8\")' '$STATE' \"\$rc\" '$CHECKPOINT' \"\$checkpoint_sha\" '$REPORT' \"\$report_sha\"; \
  exit \"\$rc\""

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "log=$LOG"
