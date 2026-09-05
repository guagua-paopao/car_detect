#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage97_eval_r1"
EVAL="$CODE/scripts/evaluate_stage110_body_class_thresholds.py"
TRAIN_ROOT="$BASE/runs/attributes/ATTR-STAGE114-DOMAIN-BALANCED-SPECIALIST-R1"
TRAIN_RUN="$TRAIN_ROOT/ATTR-STAGE114-TRUCK-SUBTYPE-CONVNEXT-256-DOMAIN-BALANCED-R1"
TRAIN_SESSION=VCAS-STAGE114-DOMAIN-BALANCED-SPECIALIST-R1
MANIFEST="$BASE/datasets/attribute-domain-v2/stage64-validation-views-v1/vfg.validation-taxonomy-v2.csv"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
BODY="$BASE/runs/attributes/ATTR-STAGE84-BODY-V2-CHAIN-R2/ATTR-STAGE84-BODY-CONVNEXT-256-V2-MIO-CCTV-R2/best.pt"
COLOR="$BASE/runs/attributes/ATTR-STAGE103-COLOR-V2-CCTV-PARTIAL-R2/ATTR-STAGE103-COLOR-CONVNEXT-256-V2-CCTV-PARTIAL-R2/gate-best.pt"
BASELINE="$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
ROOT="$BASE/runs/attributes/ATTR-STAGE115-DOMAIN-SPECIALIST-VALIDATION-R1"
STATE="$ROOT.state.json"
LOG="$ROOT.log"
SESSION=VCAS-STAGE115-DOMAIN-SPECIALIST-VALIDATION-R1

assert_sha256() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || { echo "SHA256 mismatch: $path" >&2; exit 64; }
}

worker() {
  while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do sleep 30; done
  for required in "$PY" "$EVAL" "$MANIFEST" "$LABELS" "$BODY" "$COLOR" "$BASELINE" \
                  "$TRAIN_RUN/best.pt" "$TRAIN_RUN/gate-best.pt" "$TRAIN_RUN/metrics.json" "$TRAIN_RUN/model_card.json"; do
    [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
  done
  assert_sha256 "$EVAL" 0c7d105a54101ed2c8e8b9f803339f10ae7c46644479a36f40107b0a0a269f17
  assert_sha256 "$MANIFEST" 70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6
  assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
  assert_sha256 "$BODY" e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec
  assert_sha256 "$COLOR" 811881e434d69a2956de80b88546644722bd1a2e0fc1490e03545fda3a5b7ef2
  assert_sha256 "$BASELINE" 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383
  "$PY" - "$TRAIN_RUN/model_card.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
assert d.get('metrics',{}).get('test',{}).get('status')=='not_run'
assert d.get('dataset_version')=='attribute-domain-v2-stage113-domain-balanced-specialist-r1'
assert d.get('code_revision')=='stage114-domain-balanced-specialist-r1'
PY
  [[ ! -e "$ROOT" && ! -e "$STATE" ]] || { echo "refusing to overwrite Stage115" >&2; exit 66; }
  mkdir -p "$ROOT"
  for variant in best gate-best; do
    specialist="$TRAIN_RUN/$variant.pt"
    specialist_sha="$(sha256sum "$specialist" | awk '{print tolower($1)}')"
    set +e
    "$PY" "$EVAL" \
      --manifest "$MANIFEST" --expected-manifest-sha256 70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6 \
      --labels "$LABELS" --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
      --body-checkpoint "$BODY" --expected-body-checkpoint-sha256 e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec \
      --body-specialist-checkpoint "$specialist" --expected-body-specialist-checkpoint-sha256 "$specialist_sha" \
      --body-specialist-subtype-threshold 0.50 \
      --color-checkpoint "$COLOR" --expected-color-checkpoint-sha256 811881e434d69a2956de80b88546644722bd1a2e0fc1490e03545fda3a5b7ef2 \
      --baseline-checkpoint "$BASELINE" --expected-baseline-checkpoint-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
      --datasets-safety-root "$BASE/datasets" --precision-target 0.930 \
      --device cuda --batch-size 64 --workers 8 --output "$ROOT/$variant/report.json"
    code=$?
    set -e
    [[ "$code" == 0 || "$code" == 2 ]] || exit "$code"
  done
  "$PY" - "$TRAIN_RUN" "$ROOT" "$STATE" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
train,root,state=map(Path,sys.argv[1:])
variants=[]
for name in ('best','gate-best'):
    report_path=root/name/'report.json'; report=json.load(open(report_path,encoding='utf-8'))
    assert report['policy']['split']=='validation'
    assert report['policy']['test_accessed'] is False and report['policy']['frozen_video_used'] is False
    selected=report['selection']
    variants.append({
      'variant':name,'qualified':bool(report['gates']['all_pass']),
      'specialist_checkpoint':str(train/f'{name}.pt'),
      'specialist_checkpoint_sha256':hashlib.sha256((train/f'{name}.pt').read_bytes()).hexdigest(),
      'thresholds':selected['thresholds'],'static':report['static']['candidate'],
      'complex_coverage_gain':report['comparison']['body_complex_static_coverage_gain'],
      'track_final':report['track_fusion']['candidate']['track_final'],
      'stability':report['track_fusion']['candidate']['stability'],
      'gates':report['gates']['gates'],'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
    })
qualified=[item['variant'] for item in variants if item['qualified']]
diagnostic=max(variants,key=lambda item:(sum(item['gates'].values()),item['static']['coverage'],item['track_final']['coverage'],item['static']['precision']))
out={'status':'complete_validation_only','created_at':datetime.now(timezone.utc).isoformat(),
     'decision':'body_component_qualified_pending_integrated_gate' if qualified else 'body_component_rejected_fail_closed',
     'qualified_variants':qualified,'selected_diagnostic':diagnostic['variant'],'variants':variants,
     'test_accessed':False,'frozen_video_used':False,'production_model_modified':False,
     'backend_gates_run':False,'deployment_performed':False}
state.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
Path(str(state)+'.sha256').write_text(hashlib.sha256(state.read_bytes()).hexdigest()+'  '+state.name+'\n',encoding='utf-8')
print(json.dumps({k:out[k] for k in ('decision','qualified_variants','selected_diagnostic')},ensure_ascii=False))
PY
}

if [[ "${1:-}" == "--worker" ]]; then worker; exit; fi
[[ ! -e "$ROOT" && ! -e "$STATE" ]] || { echo "refusing to overwrite Stage115" >&2; exit 66; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists" >&2; exit 67; }
tmux new-session -d -s "$SESSION" "bash '$0' --worker >'$LOG' 2>&1"
echo "started tmux:$SESSION"
echo "state=$STATE"
