#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code"
EVALUATOR="$CODE/training/scripts/evaluate_attribute_baseline.py"
LABELS="$CODE/config/vehicle_labels.v1.json"
MANIFEST="$BASE/datasets/attribute-domain-v2/attribute_manifest.bmd-raw-v34-hard-v2.csv"
PRODUCTION="$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
STAGE74="$BASE/runs/attributes/ATTR-STAGE74-COLOR-CCTV-JOINT-V2/ATTR-STAGE74-COLOR-CONVNEXT-256-CCTV-JOINT-V2/best.pt"
OUTPUT_ROOT="$BASE/runs/attributes/ATTR-STAGE74-HARD-COLOR-VALIDATION-V1"
STATE="$OUTPUT_ROOT.state.json"
LOG="$OUTPUT_ROOT.log"
SESSION=VCAS-STAGE74-HARD-COLOR-VALIDATION

assert_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || { echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2; exit 64; }
}

for required in "$PY" "$EVALUATOR" "$LABELS" "$MANIFEST" "$PRODUCTION" "$STAGE74"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$EVALUATOR" 974a684d34dc45e4572d8efa5e703bee2de486f3b8ba44649a1909bef37ef2bb
assert_sha256 "$LABELS" c71675a0e2de950880f76f088c108c6fb286561a2399cb5a182055289fb5eb4f
assert_sha256 "$MANIFEST" 51cbef8aa078579b67435908bfaf42a31efce47e79045c2552a4a4ec3d920c40
assert_sha256 "$PRODUCTION" 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383
assert_sha256 "$STAGE74" 0c17e979c79a9c16978ef46d628650a3d883b311599af0c24f8b7b3a2264c581

[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing to overwrite: $OUTPUT_ROOT" >&2; exit 66; }
[[ ! -e "$STATE" ]] || { echo "refusing to overwrite: $STATE" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi
mkdir -p "$OUTPUT_ROOT"

tmux new-session -d -s "$SESSION" \
  "set +e; rc=0; \
  '$PY' '$EVALUATOR' --manifest '$MANIFEST' --checkpoint '$PRODUCTION' --labels '$LABELS' \
    --output '$OUTPUT_ROOT/production.json' --split validation --batch-size 128 --workers 4 --device cuda \
    --type-threshold 0.75 --color-threshold 0.70 || rc=\$?; \
  if [[ \$rc -eq 0 ]]; then \
    '$PY' '$EVALUATOR' --manifest '$MANIFEST' --checkpoint '$STAGE74' --labels '$LABELS' \
      --output '$OUTPUT_ROOT/stage74_color.json' --split validation --batch-size 128 --workers 4 --device cuda \
      --type-threshold 0.96 --color-threshold 0.891 || rc=\$?; \
  fi >'$LOG' 2>&1; \
  for name in production stage74_color; do [[ -f '$OUTPUT_ROOT'/\"\$name\".json ]] && sha256sum '$OUTPUT_ROOT'/\"\$name\".json > '$OUTPUT_ROOT'/\"\$name\".json.sha256; done; \
  '$PY' -c 'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); rc=int(sys.argv[2]); files={n:{\"path\":str(root/f\"{n}.json\"),\"exists\":(root/f\"{n}.json\").is_file()} for n in (\"production\",\"stage74_color\")}; Path(sys.argv[3]).write_text(json.dumps({\"status\":\"complete_fixed_threshold_validation\" if rc == 0 and all(v[\"exists\"] for v in files.values()) else \"failed_closed_runtime\",\"exit_code\":rc,\"selection_source\":\"VFG-7 validation only\",\"fixed_thresholds\":{\"production_color\":0.70,\"stage74_color\":0.891},\"split\":\"validation\",\"threshold_sweep\":False,\"test_accessed\":False,\"frozen_video_used\":False,\"production_model_modified\":False,\"backend_gates_run\":False,\"deployment_performed\":False,\"outputs\":files},indent=2)+\"\\n\",encoding=\"utf-8\")' '$OUTPUT_ROOT' \"\$rc\" '$STATE'; \
  exit \"\$rc\""

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "log=$LOG"
