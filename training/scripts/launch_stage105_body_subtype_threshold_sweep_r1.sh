#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
EVAL_ROOT="$BASE/code/training_stage97_eval_r1"
EVALUATOR="$EVAL_ROOT/scripts/evaluate_v2_decoupled_shared_validation.py"
CODE="$BASE/code/training_stage102_partial_color_r1"
SUMMARIZER="$CODE/scripts/summarize_stage105_body_subtype_sweep.py"
MANIFEST="$BASE/datasets/attribute-domain-v2/stage64-validation-views-v1/vfg.validation-taxonomy-v2.csv"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
BODY="$BASE/runs/attributes/ATTR-STAGE84-BODY-V2-CHAIN-R2/ATTR-STAGE84-BODY-CONVNEXT-256-V2-MIO-CCTV-R2/best.pt"
SPECIALIST="$BASE/runs/attributes/ATTR-STAGE99-TWO-AXLE-SPECIALIST-R1/ATTR-STAGE99-TRUCK-SUBTYPE-CONVNEXT-256-TWO-AXLE-R1/best.pt"
COLOR="$BASE/runs/attributes/ATTR-STAGE80-COLOR-CONVNEXT-256-V2-DVM-R3/best.pt"
BASELINE="$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
OUTPUT="$BASE/runs/attributes/ATTR-STAGE105-BODY-SUBTYPE-THRESHOLD-SWEEP-R1"
STATE="$BASE/runs/attributes/ATTR-STAGE105-BODY-SUBTYPE-THRESHOLD-SWEEP-R1.state.json"
LOG="$BASE/runs/attributes/ATTR-STAGE105-BODY-SUBTYPE-THRESHOLD-SWEEP-R1.log"
SESSION=VCAS-STAGE105-BODY-SUBTYPE-SWEEP-R1
UPSTREAM_SESSION=VCAS-STAGE104-COLOR-PARTIAL-CHAIN-R2
SELF="$CODE/scripts/launch_stage105_body_subtype_threshold_sweep_r1.sh"

run_worker() {
  exec >"$LOG" 2>&1
  while tmux has-session -t "$UPSTREAM_SESSION" 2>/dev/null; do
    sleep 30
  done
  mkdir -p "$OUTPUT"
  "$PY" -c 'import json,sys; from datetime import datetime,timezone; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"status":"running_validation_only_subtype_threshold_sweep","started_at":datetime.now(timezone.utc).isoformat(),"thresholds":[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95],"test_accessed":False,"frozen_video_used":False,"production_model_modified":False},indent=2)+"\n",encoding="utf-8")' "$STATE"
  local spec code threshold report
  cd "$EVAL_ROOT"
  for spec in 050:0.50 055:0.55 060:0.60 065:0.65 070:0.70 075:0.75 080:0.80 085:0.85 090:0.90 095:0.95; do
    code="${spec%%:*}"
    threshold="${spec#*:}"
    report="$OUTPUT/subtype-$code.json"
    "$PY" "$EVALUATOR" \
      --manifest "$MANIFEST" --expected-manifest-sha256 70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6 \
      --labels "$LABELS" --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
      --body-checkpoint "$BODY" --expected-body-checkpoint-sha256 e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec \
      --body-specialist-checkpoint "$SPECIALIST" --expected-body-specialist-checkpoint-sha256 0da63e45ffe4c5af4594288b10ed0ea209bee51993c676083d01847fc2ca4fae \
      --body-specialist-subtype-threshold "$threshold" \
      --color-checkpoint "$COLOR" --expected-color-checkpoint-sha256 a2894e21060a09c690756d83bb776ffe71acda56da9f2760df577ce080c00c47 \
      --baseline-checkpoint "$BASELINE" --expected-baseline-checkpoint-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
      --output "$report" --datasets-safety-root "$BASE/datasets" \
      --device cuda --batch-size 64 --workers 8 --precision-gate 0.93
    sha256sum "$report" > "$report.sha256"
  done
  sha256sum "$OUTPUT"/*.json > "$OUTPUT/reports.sha256"
  "$PY" "$SUMMARIZER" --reports-dir "$OUTPUT" --output "$STATE"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi

for absent in "$OUTPUT" "$STATE" "$LOG"; do
  [[ ! -e "$absent" ]] || { echo "refusing to overwrite Stage105 evidence: $absent" >&2; exit 66; }
done
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi
echo "447dd2d2e5f9e0902212154f4c207abd550df75b12ab42dbe420fa67a937ca38  $EVALUATOR" | sha256sum -c -
echo "d946021a1fc6ec8ff157fac15e6038db11b47649b3eeb5b355297d259c62b92b  $SUMMARIZER" | sha256sum -c -
echo "70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6  $MANIFEST" | sha256sum -c -
echo "22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f  $LABELS" | sha256sum -c -
echo "e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec  $BODY" | sha256sum -c -
echo "0da63e45ffe4c5af4594288b10ed0ea209bee51993c676083d01847fc2ca4fae  $SPECIALIST" | sha256sum -c -
echo "a2894e21060a09c690756d83bb776ffe71acda56da9f2760df577ce080c00c47  $COLOR" | sha256sum -c -
echo "6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383  $BASELINE" | sha256sum -c -
tmux new-session -d -s "$SESSION" "bash '$SELF' --worker"

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "output=$OUTPUT"
echo "log=$LOG"
