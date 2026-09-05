#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
EVAL_ROOT="$BASE/code/training_stage97_eval_r1"
EVALUATOR="$EVAL_ROOT/scripts/evaluate_v2_decoupled_shared_validation.py"
CODE="$BASE/code/training_stage102_partial_color_r1"
SUMMARIZER="$CODE/scripts/summarize_stage104_color_validation.py"
MANIFEST="$BASE/datasets/attribute-domain-v2/stage64-validation-views-v1/vfg.validation-taxonomy-v2.csv"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
BODY="$BASE/runs/attributes/ATTR-STAGE84-BODY-V2-CHAIN-R2/ATTR-STAGE84-BODY-CONVNEXT-256-V2-MIO-CCTV-R2/best.pt"
BASELINE="$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
STAGE103_ROOT="$BASE/runs/attributes/ATTR-STAGE103-COLOR-V2-CCTV-PARTIAL-R2"
STAGE103_RUN="$STAGE103_ROOT/ATTR-STAGE103-COLOR-CONVNEXT-256-V2-CCTV-PARTIAL-R2"
OUTPUT="$BASE/runs/attributes/ATTR-STAGE104-COLOR-PARTIAL-INDEPENDENT-VALIDATION-R2"
STATE="$BASE/runs/attributes/ATTR-STAGE104-COLOR-PARTIAL-INDEPENDENT-VALIDATION-R2.state.json"
LOG="$BASE/runs/attributes/ATTR-STAGE104-COLOR-PARTIAL-INDEPENDENT-VALIDATION-R2.log"
SESSION=VCAS-STAGE104-COLOR-PARTIAL-CHAIN-R2
STAGE103_SESSION=VCAS-STAGE103-COLOR-PARTIAL-R2
SELF="$CODE/scripts/launch_stage104_color_partial_validation_chain_r2.sh"

run_worker() {
  exec >"$LOG" 2>&1
  while tmux has-session -t "$STAGE103_SESSION" 2>/dev/null; do
    sleep 30
  done
  for required in \
    "$STAGE103_RUN/best.pt" "$STAGE103_RUN/gate-best.pt" \
    "$STAGE103_RUN/metrics.json" "$STAGE103_RUN/model_card.json"; do
    [[ -f "$required" ]] || { echo "Stage103 R2 incomplete: missing $required" >&2; exit 70; }
  done
  "$PY" -c 'import json,sys; c=json.load(open(sys.argv[1],encoding="utf-8")); p=c.get("coarse_color_partial_supervision",{}); assert p.get("enabled") is True; assert abs(float(p.get("loss_weight"))-0.5)<1e-12; assert c.get("metrics",{}).get("test",{}).get("status")=="not_run"' "$STAGE103_RUN/model_card.json"
  local best_sha gate_sha metrics_sha card_sha color_sha variant
  best_sha="$(sha256sum "$STAGE103_RUN/best.pt" | awk '{print tolower($1)}')"
  gate_sha="$(sha256sum "$STAGE103_RUN/gate-best.pt" | awk '{print tolower($1)}')"
  metrics_sha="$(sha256sum "$STAGE103_RUN/metrics.json" | awk '{print tolower($1)}')"
  card_sha="$(sha256sum "$STAGE103_RUN/model_card.json" | awk '{print tolower($1)}')"
  mkdir -p "$OUTPUT"
  "$PY" -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"status":"stage103_r2_complete_inputs_pinned","stage103_best_sha256":sys.argv[2],"stage103_gate_best_sha256":sys.argv[3],"stage103_metrics_sha256":sys.argv[4],"stage103_model_card_sha256":sys.argv[5],"split":"validation","test_accessed":False,"frozen_video_used":False,"production_model_modified":False},indent=2)+"\n",encoding="utf-8")' \
    "$STATE" "$best_sha" "$gate_sha" "$metrics_sha" "$card_sha"
  cd "$EVAL_ROOT"
  for variant in best gate-best; do
    if [[ "$variant" == best ]]; then color_sha="$best_sha"; else color_sha="$gate_sha"; fi
    "$PY" "$EVALUATOR" \
      --manifest "$MANIFEST" --expected-manifest-sha256 70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6 \
      --labels "$LABELS" --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
      --body-checkpoint "$BODY" --expected-body-checkpoint-sha256 e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec \
      --color-checkpoint "$STAGE103_RUN/$variant.pt" --expected-color-checkpoint-sha256 "$color_sha" \
      --baseline-checkpoint "$BASELINE" --expected-baseline-checkpoint-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
      --output "$OUTPUT/$variant.json" --datasets-safety-root "$BASE/datasets" \
      --device cuda --batch-size 64 --workers 8 --precision-gate 0.93
    sha256sum "$OUTPUT/$variant.json" > "$OUTPUT/$variant.json.sha256"
  done
  sha256sum "$OUTPUT"/*.json > "$OUTPUT/reports.sha256"
  "$PY" "$SUMMARIZER" --reports-dir "$OUTPUT" --output-state "$STATE"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi

for absent in "$OUTPUT" "$STATE" "$LOG"; do
  [[ ! -e "$absent" ]] || { echo "refusing to overwrite Stage104 R2 evidence: $absent" >&2; exit 66; }
done
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi
echo "447dd2d2e5f9e0902212154f4c207abd550df75b12ab42dbe420fa67a937ca38  $EVALUATOR" | sha256sum -c -
echo "0abe756458548c979bcd06f5693f1ed302f9c9a60cb1e5bf9416153dd1dce55b  $SUMMARIZER" | sha256sum -c -
echo "70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6  $MANIFEST" | sha256sum -c -
echo "22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f  $LABELS" | sha256sum -c -
echo "e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec  $BODY" | sha256sum -c -
echo "6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383  $BASELINE" | sha256sum -c -
tmux new-session -d -s "$SESSION" "bash '$SELF' --worker"

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "output=$OUTPUT"
echo "log=$LOG"
