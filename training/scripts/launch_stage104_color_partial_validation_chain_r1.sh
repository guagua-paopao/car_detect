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
STAGE103_ROOT="$BASE/runs/attributes/ATTR-STAGE103-COLOR-V2-CCTV-PARTIAL-R1"
STAGE103_RUN="$STAGE103_ROOT/ATTR-STAGE103-COLOR-CONVNEXT-256-V2-CCTV-PARTIAL-R1"
OUTPUT="$BASE/runs/attributes/ATTR-STAGE104-COLOR-PARTIAL-INDEPENDENT-VALIDATION-R1"
STATE="$BASE/runs/attributes/ATTR-STAGE104-COLOR-PARTIAL-INDEPENDENT-VALIDATION-R1.state.json"
LOG="$BASE/runs/attributes/ATTR-STAGE104-COLOR-PARTIAL-INDEPENDENT-VALIDATION-R1.log"
SESSION=VCAS-STAGE104-COLOR-PARTIAL-CHAIN-R1
STAGE103_SESSION=VCAS-STAGE103-COLOR-PARTIAL-R1
SELF="$BASE/code/training_stage102_partial_color_r1/scripts/launch_stage104_color_partial_validation_chain_r1.sh"

run_worker() {
  exec >"$LOG" 2>&1
  while tmux has-session -t "$STAGE103_SESSION" 2>/dev/null; do
    sleep 30
  done
  for required in \
    "$STAGE103_RUN/best.pt" "$STAGE103_RUN/gate-best.pt" \
    "$STAGE103_RUN/metrics.json" "$STAGE103_RUN/model_card.json"; do
    [[ -f "$required" ]] || { echo "Stage103 incomplete: missing $required" >&2; exit 70; }
  done
  "$PY" -c 'import json,sys; c=json.load(open(sys.argv[1],encoding="utf-8")); p=c.get("coarse_color_partial_supervision",{}); assert p.get("enabled") is True; assert abs(float(p.get("loss_weight"))-0.5)<1e-12; assert c.get("metrics",{}).get("test",{}).get("status")=="not_run"' "$STAGE103_RUN/model_card.json"
  local best_sha gate_sha metrics_sha card_sha color_sha variant
  best_sha="$(sha256sum "$STAGE103_RUN/best.pt" | awk '{print tolower($1)}')"
  gate_sha="$(sha256sum "$STAGE103_RUN/gate-best.pt" | awk '{print tolower($1)}')"
  metrics_sha="$(sha256sum "$STAGE103_RUN/metrics.json" | awk '{print tolower($1)}')"
  card_sha="$(sha256sum "$STAGE103_RUN/model_card.json" | awk '{print tolower($1)}')"
  mkdir -p "$OUTPUT"
  "$PY" -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"status":"stage103_complete_inputs_pinned","stage103_best_sha256":sys.argv[2],"stage103_gate_best_sha256":sys.argv[3],"stage103_metrics_sha256":sys.argv[4],"stage103_model_card_sha256":sys.argv[5],"split":"validation","test_accessed":False,"frozen_video_used":False,"production_model_modified":False},indent=2)+"\n",encoding="utf-8")' \
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
  "$PY" -c 'import json,sys; [(_ for _ in ()).throw(RuntimeError(f"invalid Stage104 report: {p}")) if (json.load(open(p)).get("status")!="complete_validation_only" or json.load(open(p)).get("policy",{}).get("test_accessed") is not False or json.load(open(p)).get("policy",{}).get("frozen_video_used") is not False) else None for p in sys.argv[1:]]' \
    "$OUTPUT/best.json" "$OUTPUT/gate-best.json"
  sha256sum "$OUTPUT"/*.json > "$OUTPUT/reports.sha256"
  "$PY" -c 'import hashlib,json,sys; from datetime import datetime,timezone; from pathlib import Path; names=("best","gate-best"); reports={n:json.load(open(str(Path(sys.argv[2])/f"{n}.json"),encoding="utf-8")) for n in names}; gates=("color_static_precision","color_static_coverage","color_complex_unknown_reduction","color_track_precision","color_track_coverage","color_track_stability"); summary={}; qualified=[];
for n,r in reports.items():
 g=r["shared_validation_gates"]["gates"]; ok=all(g[k] for k in gates); qualified.extend([n] if ok else []); summary[n]={"static":r["threshold_selection"]["color_shared"],"complex_unknown_relative_reduction":r["comparison"]["color_complex_static_unknown_relative_reduction"],"track_final":r["track_fusion"]["candidate"]["color_shared"]["track_final"],"stability":r["track_fusion"]["candidate"]["color_shared"]["stability"],"color_gates":{k:g[k] for k in gates},"all_color_gates_pass":ok,"report_sha256":hashlib.sha256(Path(sys.argv[2],f"{n}.json").read_bytes()).hexdigest()};
state={"status":"complete_validation_only","completed_at":datetime.now(timezone.utc).isoformat(),"decision":"color_component_qualified" if qualified else "color_candidate_rejected_fail_closed","qualified_variants":qualified,"variants":summary,"test_accessed":False,"frozen_video_used":False,"production_model_modified":False,"backend_gates_run":False,"deployment_performed":False}; Path(sys.argv[1]).write_text(json.dumps(state,indent=2)+"\n",encoding="utf-8")' "$STATE" "$OUTPUT"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi

for absent in "$OUTPUT" "$STATE" "$LOG"; do
  [[ ! -e "$absent" ]] || { echo "refusing to overwrite Stage104 evidence: $absent" >&2; exit 66; }
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
tmux new-session -d -s "$SESSION" "bash '$SELF' --worker"

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "output=$OUTPUT"
echo "log=$LOG"
