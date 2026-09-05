#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage141_letterbox_r1"
ANALYZER="$CODE/scripts/analyze_stage111_body_router_errors.py"
ANALYZER_BASE="$CODE/scripts/evaluate_v2_decoupled_shared_validation.py"
MANIFEST="$BASE/datasets/attribute-domain-v2/stage64-validation-views-v1/vfg.validation-taxonomy-v2.csv"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
BODY="$BASE/runs/attributes/ATTR-STAGE84-BODY-V2-CHAIN-R2/ATTR-STAGE84-BODY-CONVNEXT-256-V2-MIO-CCTV-R2/best.pt"
TRAIN_RUN="$BASE/runs/attributes/ATTR-STAGE141-LETTERBOX-SPECIALIST-R1/ATTR-STAGE141-TRUCK-SUBTYPE-CONVNEXT-256-LETTERBOX-R1"
STAGE142_STATE="$BASE/runs/attributes/ATTR-STAGE142-LETTERBOX-VALIDATION-R1.state.json"
STAGE143_STATE="$BASE/runs/attributes/ATTR-STAGE143-INTEGRATED-COMPONENT-GATE-R1.state.json"
ROOT="$BASE/runs/attributes/ATTR-STAGE144-LETTERBOX-ROUTER-DIAGNOSIS-R1"
STATE="$ROOT.state.json"
LOG="$ROOT.log"
SESSION=VCAS-STAGE144-LETTERBOX-ROUTER-DIAGNOSIS-R1

assert_sha256() {
  local path="$1" expected="$2" actual
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 65; }
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "${expected,,}" ]] || { echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2; exit 65; }
}

worker() {
  assert_sha256 "$ANALYZER" 942eb58bdc4a466739e7a23d7e05eda6fbcef46cd6725217dcfc147d9c4ed82f
  assert_sha256 "$ANALYZER_BASE" 35d19635280299e1a894647a7b98803aca13be308a59117f15aa3c61032b0e12
  assert_sha256 "$MANIFEST" 70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6
  assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
  assert_sha256 "$BODY" e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec
  assert_sha256 "$STAGE142_STATE" 6869d040b9f8ef573b57cea759a8eab19df427f3bf87c7462219df5d8c2e3254
  for required in "$PY" "$STAGE143_STATE" "$TRAIN_RUN/best.pt" "$TRAIN_RUN/last.pt" "$TRAIN_RUN/gate-best.pt"; do
    [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
  done
  "$PY" - "$STAGE142_STATE" "$STAGE143_STATE" <<'PY'
import json,sys
s142=json.load(open(sys.argv[1],encoding='utf-8'))
s143=json.load(open(sys.argv[2],encoding='utf-8'))
assert s142['decision']=='body_component_rejected_fail_closed'
assert s142['test_accessed'] is False and s142['frozen_video_used'] is False
assert s143['status']=='component_repair_required_fail_closed'
assert s143['independent_test_authorized'] is False
assert s143['test_accessed'] is False and s143['frozen_video_used'] is False
PY
  [[ ! -e "$ROOT" && ! -e "$STATE" ]] || { echo "refusing to overwrite Stage144" >&2; exit 66; }
  variants=(best last gate-best)
  hashes=(
    32b0678a32ba0bae51c2203d379c45d4174108676014516e07e3029b5469dade
    50e599f65aa7d0567511faf982a500fc289e7b84c833c1b32f6a8481928e77a9
    047636ff0d637d927d4d4377b84271e25f2d9ea9d09aaccab586463e673e8400
  )
  for i in "${!variants[@]}"; do
    variant="${variants[$i]}"
    checkpoint="$TRAIN_RUN/$variant.pt"
    "$PY" "$ANALYZER" \
      --manifest "$MANIFEST" \
      --expected-manifest-sha256 70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6 \
      --labels "$LABELS" \
      --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
      --body-checkpoint "$BODY" \
      --expected-body-checkpoint-sha256 e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec \
      --specialist-checkpoint "$checkpoint" \
      --expected-specialist-checkpoint-sha256 "${hashes[$i]}" \
      --datasets-safety-root "$BASE/datasets" \
      --device cuda --batch-size 64 --workers 8 \
      --output "$ROOT/$variant/report.json"
  done
  "$PY" - "$ROOT" "$STATE" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
root,state=Path(sys.argv[1]),Path(sys.argv[2])
items=[]
for variant in ('best','last','gate-best'):
    path=root/variant/'report.json'
    report=json.loads(path.read_text(encoding='utf-8'))
    route=report['aggregate']['truck_family_route']
    items.append({
        'variant':variant,
        'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'geometry_contract':report['inputs']['geometry_contract'],
        **route,
        'diagnosis':report['aggregate']['diagnosis'],
        'light_truck_stratified':report['light_truck_stratified'],
    })
selected=max(items,key=lambda x:(x['light_specialist_precision_given_family_route'],x['light_specialist_recall_given_family_route'],x['specialist_argmax_accuracy_on_subtype_truth']))
out={
  'status':'complete_validation_only_error_attribution',
  'created_at':datetime.now(timezone.utc).isoformat(),
  'source_training_batch':'ATTR-STAGE141-LETTERBOX-SPECIALIST-R1',
  'selected_diagnostic':selected['variant'],
  'variants':items,
  'comparison_reference':{
    'stage135_best_light_precision':0.3698630136986301,
    'stage135_best_light_recall':0.4747191011235955,
  },
  'decision':'plan_domain_targeted_repair_without_lowering_gates',
  'test_accessed':False,
  'frozen_video_used':False,
  'production_model_modified':False,
  'backend_gates_run':False,
  'deployment_performed':False,
}
state.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
Path(str(state)+'.sha256').write_text(hashlib.sha256(state.read_bytes()).hexdigest()+'  '+state.name+'\n',encoding='utf-8')
print(json.dumps({'status':out['status'],'selected_diagnostic':out['selected_diagnostic'],'selected':selected},ensure_ascii=False))
PY
}

if [[ "${1:-}" == "--worker" ]]; then worker; exit; fi
[[ ! -e "$ROOT" && ! -e "$STATE" && ! -e "$LOG" ]] || { echo "refusing to overwrite Stage144" >&2; exit 66; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists" >&2; exit 67; }
tmux new-session -d -s "$SESSION" "bash '$0' --worker >'$LOG' 2>&1"
echo "started tmux:$SESSION"
echo "state=$STATE"
