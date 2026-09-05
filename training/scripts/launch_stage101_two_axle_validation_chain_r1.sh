#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
EVAL_ROOT="$BASE/code/training_stage97_eval_r1"
EVALUATOR="$EVAL_ROOT/scripts/evaluate_v2_decoupled_shared_validation.py"
MANIFEST="$BASE/datasets/attribute-domain-v2/stage64-validation-views-v1/vfg.validation-taxonomy-v2.csv"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
BODY="$BASE/runs/attributes/ATTR-STAGE84-BODY-V2-CHAIN-R2/ATTR-STAGE84-BODY-CONVNEXT-256-V2-MIO-CCTV-R2/best.pt"
COLOR="$BASE/runs/attributes/ATTR-STAGE80-COLOR-CONVNEXT-256-V2-DVM-R3/best.pt"
BASELINE="$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
STAGE99_ROOT="$BASE/runs/attributes/ATTR-STAGE99-TWO-AXLE-SPECIALIST-R1"
STAGE99_RUN="$STAGE99_ROOT/ATTR-STAGE99-TRUCK-SUBTYPE-CONVNEXT-256-TWO-AXLE-R1"
OUTPUT="$BASE/runs/attributes/ATTR-STAGE101-TWO-AXLE-SPECIALIST-VALIDATION-R1"
STATE="$BASE/runs/attributes/ATTR-STAGE101-TWO-AXLE-SPECIALIST-VALIDATION-R1.state.json"
LOG="$BASE/runs/attributes/ATTR-STAGE101-TWO-AXLE-SPECIALIST-VALIDATION-R1.log"
SESSION=VCAS-STAGE101-TWO-AXLE-CHAIN-R1
STAGE99_SESSION=VCAS-STAGE99-TWO-AXLE-R1
SELF="$BASE/code/training/scripts/launch_stage101_two_axle_validation_chain_r1.sh"

run_worker() {
  exec >"$LOG" 2>&1
  local wait_seconds=0
  while [[ ! -d "$STAGE99_ROOT" ]] && ! tmux has-session -t "$STAGE99_SESSION" 2>/dev/null; do
    (( wait_seconds < 43200 )) || { echo "Stage99 did not start within 12 hours" >&2; exit 71; }
    sleep 30
    wait_seconds=$((wait_seconds + 30))
  done
  while tmux has-session -t "$STAGE99_SESSION" 2>/dev/null; do
    sleep 30
  done
  for required in \
    "$STAGE99_RUN/best.pt" "$STAGE99_RUN/gate-best.pt" \
    "$STAGE99_RUN/metrics.json" "$STAGE99_RUN/model_card.json"; do
    [[ -f "$required" ]] || { echo "Stage99 incomplete: missing $required" >&2; exit 72; }
  done
  local best_sha gate_sha metrics_sha card_sha specialist_sha variant
  best_sha="$(sha256sum "$STAGE99_RUN/best.pt" | awk '{print tolower($1)}')"
  gate_sha="$(sha256sum "$STAGE99_RUN/gate-best.pt" | awk '{print tolower($1)}')"
  metrics_sha="$(sha256sum "$STAGE99_RUN/metrics.json" | awk '{print tolower($1)}')"
  card_sha="$(sha256sum "$STAGE99_RUN/model_card.json" | awk '{print tolower($1)}')"
  mkdir -p "$OUTPUT"
  "$PY" -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"status":"stage99_complete_inputs_pinned","stage99_best_sha256":sys.argv[2],"stage99_gate_best_sha256":sys.argv[3],"stage99_metrics_sha256":sys.argv[4],"stage99_model_card_sha256":sys.argv[5],"split":"validation","test_accessed":False,"frozen_video_used":False,"production_model_modified":False},indent=2)+"\n",encoding="utf-8")' \
    "$STATE" "$best_sha" "$gate_sha" "$metrics_sha" "$card_sha"
  cd "$EVAL_ROOT"
  for variant in best gate-best; do
    if [[ "$variant" == best ]]; then specialist_sha="$best_sha"; else specialist_sha="$gate_sha"; fi
    "$PY" "$EVALUATOR" \
      --manifest "$MANIFEST" --expected-manifest-sha256 70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6 \
      --labels "$LABELS" --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
      --body-checkpoint "$BODY" --expected-body-checkpoint-sha256 e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec \
      --body-specialist-checkpoint "$STAGE99_RUN/$variant.pt" --expected-body-specialist-checkpoint-sha256 "$specialist_sha" \
      --body-specialist-subtype-threshold 0.80 \
      --color-checkpoint "$COLOR" --expected-color-checkpoint-sha256 a2894e21060a09c690756d83bb776ffe71acda56da9f2760df577ce080c00c47 \
      --baseline-checkpoint "$BASELINE" --expected-baseline-checkpoint-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
      --output "$OUTPUT/$variant.json" --datasets-safety-root "$BASE/datasets" \
      --device cuda --batch-size 64 --workers 8 --precision-gate 0.93
    sha256sum "$OUTPUT/$variant.json" > "$OUTPUT/$variant.json.sha256"
  done
  "$PY" -c 'import json,sys; [(_ for _ in ()).throw(RuntimeError(f"invalid Stage101 report: {p}")) if (json.load(open(p)).get("status")!="complete_validation_only" or json.load(open(p)).get("policy",{}).get("test_accessed") is not False or json.load(open(p)).get("policy",{}).get("frozen_video_used") is not False) else None for p in sys.argv[1:]]' \
    "$OUTPUT/best.json" "$OUTPUT/gate-best.json"
  sha256sum "$OUTPUT"/*.json > "$OUTPUT/reports.sha256"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi

for absent in "$OUTPUT" "$STATE" "$LOG"; do
  [[ ! -e "$absent" ]] || { echo "refusing to overwrite Stage101 evidence: $absent" >&2; exit 66; }
done
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi
echo "447dd2d2e5f9e0902212154f4c207abd550df75b12ab42dbe420fa67a937ca38  $EVALUATOR" | sha256sum -c -
echo "70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6  $MANIFEST" | sha256sum -c -
echo "22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f  $LABELS" | sha256sum -c -
echo "e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec  $BODY" | sha256sum -c -
echo "a2894e21060a09c690756d83bb776ffe71acda56da9f2760df577ce080c00c47  $COLOR" | sha256sum -c -
echo "6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383  $BASELINE" | sha256sum -c -
tmux new-session -d -s "$SESSION" "bash '$SELF' --worker"

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "output=$OUTPUT"
echo "log=$LOG"
