#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage97_eval_r1"
EVAL="$CODE/scripts/evaluate_stage110_body_class_thresholds.py"
TRAIN_RUN="$BASE/runs/attributes/ATTR-STAGE114-DOMAIN-BALANCED-SPECIALIST-R1/ATTR-STAGE114-TRUCK-SUBTYPE-CONVNEXT-256-DOMAIN-BALANCED-R1"
SPECIALIST="$TRAIN_RUN/best.pt"
MANIFEST="$BASE/datasets/attribute-domain-v2/stage64-validation-views-v1/vfg.validation-taxonomy-v2.csv"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
BODY="$BASE/runs/attributes/ATTR-STAGE84-BODY-V2-CHAIN-R2/ATTR-STAGE84-BODY-CONVNEXT-256-V2-MIO-CCTV-R2/best.pt"
COLOR="$BASE/runs/attributes/ATTR-STAGE103-COLOR-V2-CCTV-PARTIAL-R2/ATTR-STAGE103-COLOR-CONVNEXT-256-V2-CCTV-PARTIAL-R2/gate-best.pt"
BASELINE="$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
ROOT="$BASE/runs/attributes/ATTR-STAGE115-DOMAIN-SPECIALIST-VALIDATION-R3"
STATE="$ROOT.state.json"
LOG="$ROOT.log"
SESSION=VCAS-STAGE115-DOMAIN-SPECIALIST-VALIDATION-R3

assert_sha256() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || { echo "SHA256 mismatch: $path" >&2; exit 64; }
}

worker() {
  for required in "$PY" "$EVAL" "$MANIFEST" "$LABELS" "$BODY" "$COLOR" "$BASELINE" \
                  "$SPECIALIST" "$TRAIN_RUN/metrics.json" "$TRAIN_RUN/model_card.json" "$TRAIN_RUN/test_metrics.json"; do
    [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
  done
  assert_sha256 "$EVAL" 0c7d105a54101ed2c8e8b9f803339f10ae7c46644479a36f40107b0a0a269f17
  assert_sha256 "$MANIFEST" 70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6
  assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
  assert_sha256 "$BODY" e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec
  assert_sha256 "$COLOR" 811881e434d69a2956de80b88546644722bd1a2e0fc1490e03545fda3a5b7ef2
  assert_sha256 "$BASELINE" 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383
  assert_sha256 "$SPECIALIST" 236c4f561a9ab33cec95956de8aca3adf2dec92593c75541b8c20aca62a935df
  assert_sha256 "$TRAIN_RUN/metrics.json" f462c79347fe1daad0fe6544c4d33d23578a27105133ca378ec8c40f8db4cd6a
  assert_sha256 "$TRAIN_RUN/model_card.json" 0588be20a816f47a509ccc45bd833af2926c497e7eab2ae56ff5e85b46f96a05
  "$PY" - "$TRAIN_RUN/model_card.json" "$TRAIN_RUN/test_metrics.json" <<'PY'
import json,sys
card=json.load(open(sys.argv[1],encoding='utf-8'))
test=json.load(open(sys.argv[2],encoding='utf-8'))
assert test.get('status')=='not_run'
assert card.get('dataset_version')=='attribute-domain-v2-stage113-domain-balanced-specialist-r1'
assert card.get('code_revision')=='stage114-domain-balanced-specialist-r1'
PY
  [[ ! -e "$ROOT" && ! -e "$STATE" ]] || { echo "refusing to overwrite Stage115 R3" >&2; exit 66; }
  set +e
  "$PY" "$EVAL" \
    --manifest "$MANIFEST" --expected-manifest-sha256 70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6 \
    --labels "$LABELS" --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
    --body-checkpoint "$BODY" --expected-body-checkpoint-sha256 e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec \
    --body-specialist-checkpoint "$SPECIALIST" --expected-body-specialist-checkpoint-sha256 236c4f561a9ab33cec95956de8aca3adf2dec92593c75541b8c20aca62a935df \
    --body-specialist-subtype-threshold 0.50 \
    --color-checkpoint "$COLOR" --expected-color-checkpoint-sha256 811881e434d69a2956de80b88546644722bd1a2e0fc1490e03545fda3a5b7ef2 \
    --baseline-checkpoint "$BASELINE" --expected-baseline-checkpoint-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
    --datasets-safety-root "$BASE/datasets" --precision-target 0.930 \
    --device cuda --batch-size 64 --workers 8 --output "$ROOT/best/report.json"
  code=$?
  set -e
  [[ "$code" == 0 || "$code" == 2 ]] || exit "$code"
  "$PY" - "$TRAIN_RUN" "$ROOT" "$STATE" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
train,root,state=map(Path,sys.argv[1:])
report_path=root/'best'/'report.json'
report=json.load(open(report_path,encoding='utf-8'))
assert report['policy']['split']=='validation'
assert report['policy']['test_accessed'] is False and report['policy']['frozen_video_used'] is False
selected=report['selection']
item={
  'variant':'best','qualified':bool(report['gates']['all_pass']),
  'specialist_checkpoint':str(train/'best.pt'),
  'specialist_checkpoint_sha256':hashlib.sha256((train/'best.pt').read_bytes()).hexdigest(),
  'thresholds':selected['thresholds'],'static':report['static']['candidate'],
  'complex_coverage_gain':report['comparison']['body_complex_static_coverage_gain'],
  'track_final':report['track_fusion']['candidate']['track_final'],
  'stability':report['track_fusion']['candidate']['stability'],
  'gates':report['gates']['gates'],'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
}
qualified=['best'] if item['qualified'] else []
out={'status':'complete_validation_only','created_at':datetime.now(timezone.utc).isoformat(),
     'decision':'body_component_qualified_pending_integrated_gate' if qualified else 'body_component_rejected_fail_closed',
     'qualified_variants':qualified,'selected_diagnostic':'best','variants':[item],
     'source_training_batch':'ATTR-STAGE114-DOMAIN-BALANCED-SPECIALIST-R1',
     'prior_contract_failures':['R1 required optional gate-best.pt','R2 pre-created evaluator-owned output directory'],
     'test_accessed':False,'frozen_video_used':False,'production_model_modified':False,
     'backend_gates_run':False,'deployment_performed':False}
state.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
Path(str(state)+'.sha256').write_text(hashlib.sha256(state.read_bytes()).hexdigest()+'  '+state.name+'\n',encoding='utf-8')
print(json.dumps({k:out[k] for k in ('decision','qualified_variants','selected_diagnostic')},ensure_ascii=False))
PY
}

if [[ "${1:-}" == "--worker" ]]; then worker; exit; fi
[[ ! -e "$ROOT" && ! -e "$STATE" && ! -e "$LOG" ]] || { echo "refusing to overwrite Stage115 R3" >&2; exit 66; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists" >&2; exit 67; }
tmux new-session -d -s "$SESSION" "bash '$0' --worker >'$LOG' 2>&1"
echo "started tmux:$SESSION"
echo "state=$STATE"
