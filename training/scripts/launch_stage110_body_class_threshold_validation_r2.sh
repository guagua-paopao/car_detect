#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage97_eval_r1"
EVAL="$CODE/scripts/evaluate_stage110_body_class_thresholds.py"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
MANIFEST="$BASE/datasets/attribute-domain-v2/stage64-validation-views-v1/vfg.validation-taxonomy-v2.csv"
BODY="$BASE/runs/attributes/ATTR-STAGE84-BODY-V2-CHAIN-R2/ATTR-STAGE84-BODY-CONVNEXT-256-V2-MIO-CCTV-R2/best.pt"
SPECIALIST="$BASE/runs/attributes/ATTR-STAGE99-TWO-AXLE-SPECIALIST-R1/ATTR-STAGE99-TRUCK-SUBTYPE-CONVNEXT-256-TWO-AXLE-R1/best.pt"
COLOR="$BASE/runs/attributes/ATTR-STAGE103-COLOR-V2-CCTV-PARTIAL-R2/ATTR-STAGE103-COLOR-CONVNEXT-256-V2-CCTV-PARTIAL-R2/gate-best.pt"
BASELINE="$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt"
ROOT="$BASE/runs/attributes/ATTR-STAGE110-BODY-CLASS-THRESHOLD-VALIDATION-R2"
STATE="$ROOT.state.json"
LOG="$ROOT.log"
SESSION=VCAS-STAGE110-BODY-CLASS-THRESHOLD-R2

worker() {
  [[ ! -e "$ROOT" && ! -e "$STATE" ]] || { echo "refusing to overwrite Stage110 R2" >&2; exit 66; }
  [[ "$(sha256sum "$EVAL" | awk '{print $1}')" == 0c7d105a54101ed2c8e8b9f803339f10ae7c46644479a36f40107b0a0a269f17 ]] || exit 64
  set +e
  "$PY" "$EVAL" \
    --manifest "$MANIFEST" --expected-manifest-sha256 70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6 \
    --labels "$LABELS" --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
    --body-checkpoint "$BODY" --expected-body-checkpoint-sha256 e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec \
    --body-specialist-checkpoint "$SPECIALIST" --expected-body-specialist-checkpoint-sha256 0da63e45ffe4c5af4594288b10ed0ea209bee51993c676083d01847fc2ca4fae \
    --body-specialist-subtype-threshold 0.50 \
    --color-checkpoint "$COLOR" --expected-color-checkpoint-sha256 811881e434d69a2956de80b88546644722bd1a2e0fc1490e03545fda3a5b7ef2 \
    --baseline-checkpoint "$BASELINE" --expected-baseline-checkpoint-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
    --datasets-safety-root "$BASE/datasets" --precision-target 0.930 \
    --device cuda --batch-size 64 --workers 8 --output "$ROOT/report.json"
  code=$?
  set -e
  [[ "$code" == 0 || "$code" == 2 ]] || exit "$code"
  "$PY" - "$ROOT/report.json" "$STATE" <<'PY'
import hashlib,json,sys
from pathlib import Path
report_path,state_path=map(Path,sys.argv[1:]); d=json.load(open(report_path,encoding='utf-8'))
assert d['policy']['split']=='validation' and d['policy']['test_accessed'] is False and d['policy']['frozen_video_used'] is False
out={'status':'complete_validation_only','decision':d['decision'],'qualified':bool(d['gates']['all_pass']),
     'static':d['static']['candidate'],'complex_coverage_gain':d['comparison']['body_complex_static_coverage_gain'],
     'track_final':d['track_fusion']['candidate']['track_final'],'stability':d['track_fusion']['candidate']['stability'],
     'thresholds':d['selection']['thresholds'],'precision_target':d['selection']['precision_target'],
     'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),'test_accessed':False,
     'frozen_video_used':False,'production_model_modified':False,'backend_gates_run':False,'deployment_performed':False}
state_path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
Path(str(state_path)+'.sha256').write_text(hashlib.sha256(state_path.read_bytes()).hexdigest()+'  '+state_path.name+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
PY
}

if [[ "${1:-}" == "--worker" ]]; then worker; exit; fi
[[ ! -e "$ROOT" && ! -e "$STATE" ]] || { echo "refusing to overwrite Stage110 R2" >&2; exit 66; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists" >&2; exit 67; }
tmux new-session -d -s "$SESSION" "bash '$0' --worker >'$LOG' 2>&1"
echo "started tmux:$SESSION"
