#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
EVAL_ROOT="$BASE/code/training_stage97_eval_r1"
EVALUATOR="$EVAL_ROOT/scripts/evaluate_v2_decoupled_shared_validation.py"
MANIFEST="$BASE/datasets/attribute-domain-v2/stage64-validation-views-v1/vfg.validation-taxonomy-v2.csv"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
BODY="$BASE/runs/attributes/ATTR-STAGE84-BODY-V2-CHAIN-R2/ATTR-STAGE84-BODY-CONVNEXT-256-V2-MIO-CCTV-R2/best.pt"
BASELINE="$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
STAGE96_ROOT="$BASE/runs/attributes/ATTR-STAGE96-COLOR-NIGHT-CONSISTENCY-R1"
STAGE96_RUN="$STAGE96_ROOT/ATTR-STAGE96-COLOR-CONVNEXT-256-NIGHT-CONSISTENCY-R1"
OUTPUT="$BASE/runs/attributes/ATTR-STAGE100-COLOR-INDEPENDENT-VALIDATION-R1"
STATE="$BASE/runs/attributes/ATTR-STAGE100-COLOR-INDEPENDENT-VALIDATION-R1.state.json"
LOG="$BASE/runs/attributes/ATTR-STAGE100-COLOR-INDEPENDENT-VALIDATION-R1.log"
STAGE99_LAUNCHER="$BASE/code/training/scripts/launch_stage99_two_axle_specialist_training_r1.sh"
SESSION=VCAS-STAGE100-COLOR-CHAIN-R1
SELF="$BASE/code/training/scripts/launch_stage100_color_validation_chain_r1.sh"

run_worker() {
  exec >"$LOG" 2>&1
  while tmux has-session -t VCAS-STAGE96-COLOR-NIGHT-R1 2>/dev/null; do
    sleep 30
  done
  for required in \
    "$STAGE96_RUN/best.pt" "$STAGE96_RUN/gate-best.pt" \
    "$STAGE96_RUN/metrics.json" "$STAGE96_RUN/model_card.json"; do
    [[ -f "$required" ]] || { echo "Stage96 incomplete: missing $required" >&2; exit 70; }
  done
  local best_sha gate_sha metrics_sha card_sha color_sha variant
  best_sha="$(sha256sum "$STAGE96_RUN/best.pt" | awk '{print tolower($1)}')"
  gate_sha="$(sha256sum "$STAGE96_RUN/gate-best.pt" | awk '{print tolower($1)}')"
  metrics_sha="$(sha256sum "$STAGE96_RUN/metrics.json" | awk '{print tolower($1)}')"
  card_sha="$(sha256sum "$STAGE96_RUN/model_card.json" | awk '{print tolower($1)}')"
  mkdir -p "$OUTPUT"
  "$PY" -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"status":"stage96_complete_inputs_pinned","stage96_best_sha256":sys.argv[2],"stage96_gate_best_sha256":sys.argv[3],"stage96_metrics_sha256":sys.argv[4],"stage96_model_card_sha256":sys.argv[5],"split":"validation","test_accessed":False,"frozen_video_used":False,"production_model_modified":False},indent=2)+"\n",encoding="utf-8")' \
    "$STATE" "$best_sha" "$gate_sha" "$metrics_sha" "$card_sha"
  cd "$EVAL_ROOT"
  for variant in best gate-best; do
    if [[ "$variant" == best ]]; then color_sha="$best_sha"; else color_sha="$gate_sha"; fi
    "$PY" "$EVALUATOR" \
      --manifest "$MANIFEST" --expected-manifest-sha256 70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6 \
      --labels "$LABELS" --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
      --body-checkpoint "$BODY" --expected-body-checkpoint-sha256 e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec \
      --color-checkpoint "$STAGE96_RUN/$variant.pt" --expected-color-checkpoint-sha256 "$color_sha" \
      --baseline-checkpoint "$BASELINE" --expected-baseline-checkpoint-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
      --output "$OUTPUT/$variant.json" --datasets-safety-root "$BASE/datasets" \
      --device cuda --batch-size 64 --workers 8 --precision-gate 0.93
    sha256sum "$OUTPUT/$variant.json" > "$OUTPUT/$variant.json.sha256"
  done
  "$PY" -c 'import json,sys; [(_ for _ in ()).throw(RuntimeError(f"invalid Stage100 report: {p}")) if (json.load(open(p)).get("status")!="complete_validation_only" or json.load(open(p)).get("policy",{}).get("test_accessed") is not False or json.load(open(p)).get("policy",{}).get("frozen_video_used") is not False) else None for p in sys.argv[1:]]' \
    "$OUTPUT/best.json" "$OUTPUT/gate-best.json"
  sha256sum "$OUTPUT"/*.json > "$OUTPUT/reports.sha256"
  bash "$STAGE99_LAUNCHER"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi

for absent in "$OUTPUT" "$STATE" "$LOG"; do
  [[ ! -e "$absent" ]] || { echo "refusing to overwrite Stage100 evidence: $absent" >&2; exit 66; }
done
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi

echo "447dd2d2e5f9e0902212154f4c207abd550df75b12ab42dbe420fa67a937ca38  $EVALUATOR" | sha256sum -c -
echo "70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6  $MANIFEST" | sha256sum -c -
echo "22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f  $LABELS" | sha256sum -c -
echo "e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec  $BODY" | sha256sum -c -
echo "6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383  $BASELINE" | sha256sum -c -
echo "0142c3586f8e55238a9cdde44d752f9bc536cdeefb06a4c18ecd9c24518680a2  $STAGE99_LAUNCHER" | sha256sum -c -
tmux new-session -d -s "$SESSION" "bash '$SELF' --worker"

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "output=$OUTPUT"
echo "log=$LOG"
