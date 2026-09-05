#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage97_eval_r1"
ANALYZER="$CODE/scripts/analyze_stage111_body_router_errors.py"
MANIFEST="$BASE/datasets/attribute-domain-v2/stage64-validation-views-v1/vfg.validation-taxonomy-v2.csv"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
BODY="$BASE/runs/attributes/ATTR-STAGE84-BODY-V2-CHAIN-R2/ATTR-STAGE84-BODY-CONVNEXT-256-V2-MIO-CCTV-R2/best.pt"
ROOT="$BASE/runs/attributes/ATTR-STAGE135-SPECIALIST-ROUTER-COMPARISON-R1"
LOG="$ROOT.log"
SESSION=VCAS-STAGE135-SPECIALIST-ROUTER-COMPARISON-R1

declare -A CHECKPOINTS=(
  [stage99]="$BASE/runs/attributes/ATTR-STAGE99-TWO-AXLE-SPECIALIST-R1/ATTR-STAGE99-TRUCK-SUBTYPE-CONVNEXT-256-TWO-AXLE-R1/best.pt"
  [stage121]="$BASE/runs/attributes/ATTR-STAGE121-AXLE-SEMANTIC-CLEAN-R1/ATTR-STAGE121-TRUCK-SUBTYPE-CONVNEXT-256-AXLE-CLEAN-R1/best.pt"
  [stage129]="$BASE/runs/attributes/ATTR-STAGE129-BACKBONE-PRESERVED-HEAD-RESET-R1/ATTR-STAGE129-TRUCK-SUBTYPE-CONVNEXT-256-HEAD-ONLY-R1/best.pt"
  [stage132_best]="$BASE/runs/attributes/ATTR-STAGE132-BACKBONE-PRESERVED-FULL-FINETUNE-R1/ATTR-STAGE132-TRUCK-SUBTYPE-CONVNEXT-256-FULL-FINETUNE-R1/best.pt"
  [stage132_last]="$BASE/runs/attributes/ATTR-STAGE132-BACKBONE-PRESERVED-FULL-FINETUNE-R1/ATTR-STAGE132-TRUCK-SUBTYPE-CONVNEXT-256-FULL-FINETUNE-R1/last.pt"
)
declare -A HASHES=(
  [stage99]=0da63e45ffe4c5af4594288b10ed0ea209bee51993c676083d01847fc2ca4fae
  [stage121]=fa80144ffc84886188d8268935f5cdadb5d74cda3af170b44cf80a4d3f060545
  [stage129]=1cb1bb40c5e8cde2244b8dcce992239e758af6ca5cf074f4d53f5761b2ada2a7
  [stage132_best]=d596185fd49735eb8156afc55480883171afe48e131f1abd6b358b4ff63963e4
  [stage132_last]=4d8bd78da0c27c1a9ec272b72a64d566de8559dc28fb13a4aa0299c4333edcec
)

assert_sha256() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  [[ "$actual" == "$expected" ]] || { echo "SHA256 mismatch: $path" >&2; exit 64; }
}

worker() {
  for required in "$PY" "$ANALYZER" "$MANIFEST" "$LABELS" "$BODY"; do
    [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 65; }
  done
  assert_sha256 "$ANALYZER" 8836590153a00ca6ea49c9fad6c45224d3572716121ba400399597a66675d1b0
  assert_sha256 "$MANIFEST" 70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6
  assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
  assert_sha256 "$BODY" e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec
  [[ ! -e "$ROOT" ]] || { echo "refusing to overwrite Stage135" >&2; exit 66; }
  for name in stage99 stage121 stage129 stage132_best stage132_last; do
    checkpoint="${CHECKPOINTS[$name]}"
    expected="${HASHES[$name]}"
    [[ -f "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 65; }
    assert_sha256 "$checkpoint" "$expected"
    "$PY" "$ANALYZER" \
      --manifest "$MANIFEST" --expected-manifest-sha256 70d0295eabedfab9370a946b390201dcf1099a6d741e585ed322b2649401fec6 \
      --labels "$LABELS" --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
      --body-checkpoint "$BODY" --expected-body-checkpoint-sha256 e9fae2b82039e69d18274156b1477dc6d849fb104acf6c8d32e76968db85aaec \
      --specialist-checkpoint "$checkpoint" --expected-specialist-checkpoint-sha256 "$expected" \
      --datasets-safety-root "$BASE/datasets" --device cuda --batch-size 64 --workers 8 \
      --output "$ROOT/$name/report.json"
  done
  "$PY" - "$ROOT" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for name in ('stage99','stage121','stage129','stage132_best','stage132_last'):
    p=root/name/'report.json'; d=json.load(open(p,encoding='utf-8'))
    assert d['policy']['split']=='validation'
    assert d['policy']['test_accessed'] is False and d['policy']['frozen_video_used'] is False
    route=d['aggregate']['truck_family_route']
    rows.append({'variant':name,**route,'diagnosis':d['aggregate']['diagnosis'],
                 'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
out={'schema_version':'stage135-specialist-router-comparison-v1','created_at':datetime.now(timezone.utc).isoformat(),
     'status':'complete_validation_only_error_attribution','variants':rows,
     'best_light_precision_variant':max(rows,key=lambda x:x['light_specialist_precision_given_family_route'])['variant'],
     'best_light_recall_variant':max(rows,key=lambda x:x['light_specialist_recall_given_family_route'])['variant'],
     'test_accessed':False,'frozen_video_used':False,'production_model_modified':False,
     'backend_gates_run':False,'deployment_performed':False}
p=root/'comparison.json'; p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
Path(str(p)+'.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
PY
}

if [[ "${1:-}" == "--worker" ]]; then worker; exit; fi
[[ ! -e "$ROOT" && ! -e "$LOG" ]] || { echo "refusing to overwrite Stage135" >&2; exit 66; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists" >&2; exit 67; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
  echo "GPU is occupied" >&2; exit 68
fi
tmux new-session -d -s "$SESSION" "bash '$0' --worker >'$LOG' 2>&1"
echo "started tmux:$SESSION"
echo "output=$ROOT/comparison.json"
