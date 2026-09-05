#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code"
SWEEP="$CODE/training/scripts/sweep_attribute_track_fusion_fail_closed_v2.py"
V1="$CODE/training/scripts/sweep_attribute_track_fusion.py"
FUSION="$CODE/training/scripts/track_fusion_fail_closed_v2.py"
MANIFEST="$BASE/datasets/attribute-domain-v2/vfg7-eval-v1/attribute_manifest.csv"
PRODUCTION="$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
STAGE71="$BASE/runs/attributes/ATTR-STAGE71-CONVNEXT-TEACHERS-V1/ATTR-STAGE71-COLOR-CONVNEXT-256-DVM-REPLAY-CCTV/best.pt"
STAGE74="$BASE/runs/attributes/ATTR-STAGE74-COLOR-CCTV-JOINT-V2/ATTR-STAGE74-COLOR-CONVNEXT-256-CCTV-JOINT-V2/best.pt"
OUTPUT_ROOT="$BASE/runs/attributes/ATTR-STAGE74-VFG7-TRACK-FUSION-V2"
STATE="$OUTPUT_ROOT.state.json"
LOG="$OUTPUT_ROOT.log"
SESSION=VCAS-STAGE74-VFG7-FUSION-V2

assert_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
    exit 64
  fi
}

for required in "$PY" "$SWEEP" "$V1" "$FUSION" "$MANIFEST" "$PRODUCTION" "$STAGE71" "$STAGE74"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
done
assert_sha256 "$SWEEP" e3dcfa41d3d2e0a787eae77824309c6cca5641c8dd413ac55136dd585da28e5d
assert_sha256 "$V1" e3310eda88ae9e8afec5337408fd5fe9f977a0a176ac1b8f55375722c195e502
assert_sha256 "$FUSION" faca95f8b17661a4992b566e471def9b7f09ac3914741a88612e8fecfb396264
assert_sha256 "$MANIFEST" 5582a65fd02874b6e84776d9e00a883d4ad0cb4fa579ac10c829583ec751bce3
assert_sha256 "$PRODUCTION" 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383
assert_sha256 "$STAGE71" 14bf423d277085c7decec236f1cc3931274ad778d0101e5d839be43648254121
assert_sha256 "$STAGE74" 0c17e979c79a9c16978ef46d628650a3d883b311599af0c24f8b7b3a2264c581

[[ ! -e "$OUTPUT_ROOT" ]] || { echo "refusing to overwrite: $OUTPUT_ROOT" >&2; exit 66; }
[[ ! -e "$STATE" ]] || { echo "refusing to overwrite: $STATE" >&2; exit 66; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi

THRESHOLDS=()
for value in $(seq 50 99); do
  THRESHOLDS+=("0.$value")
done

mkdir -p "$OUTPUT_ROOT"

tmux new-session -d -s "$SESSION" \
  "set +e; \
  models=('production=$PRODUCTION' 'stage71_color=$STAGE71' 'stage74_color=$STAGE74'); \
  rc=0; \
  for item in \"\${models[@]}\"; do \
    name=\"\${item%%=*}\"; checkpoint=\"\${item#*=}\"; \
    '$PY' '$SWEEP' \
      --manifest '$MANIFEST' \
      --checkpoint \"\$checkpoint\" \
      --output '$OUTPUT_ROOT'/\"\$name\".json \
      --split validation \
      --minimum-window-frames 3 \
      --thresholds ${THRESHOLDS[*]} \
      --windows 3 5 \
      --minimum-shares 0.50 0.60 0.70 0.80 0.90 0.95 \
      --minimum-margins 0.00 0.10 0.20 0.30 0.40 0.50 \
      --batch-size 128 --workers 4 --device cuda || { rc=\$?; break; }; \
    sha256sum '$OUTPUT_ROOT'/\"\$name\".json > '$OUTPUT_ROOT'/\"\$name\".json.sha256; \
  done >'$LOG' 2>&1; \
  '$PY' -c 'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); rc=int(sys.argv[2]); names=(\"production\",\"stage71_color\",\"stage74_color\"); outputs={name:{\"path\":str(root/f\"{name}.json\"),\"exists\":(root/f\"{name}.json\").is_file()} for name in names}; Path(sys.argv[3]).write_text(json.dumps({\"status\":\"complete_validation_only\" if rc == 0 and all(v[\"exists\"] for v in outputs.values()) else \"failed_closed_runtime\",\"exit_code\":rc,\"split\":\"validation\",\"test_accessed\":False,\"frozen_video_used\":False,\"production_model_modified\":False,\"backend_gates_run\":False,\"deployment_performed\":False,\"outputs\":outputs},indent=2)+\"\\n\",encoding=\"utf-8\")' '$OUTPUT_ROOT' \"\$rc\" '$STATE'; \
  exit \"\$rc\""

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "log=$LOG"
