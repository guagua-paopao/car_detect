#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE=$BASE/code/stage164-persistent-fusion-r1
EVAL_ROOT=$BASE/code/stage159_component_validation_r6/scripts
EVALUATOR=$CODE/scripts/evaluate_stage164_persistent_fusion.py
EVALUATOR_TEST=$CODE/scripts/test_evaluate_stage164_persistent_fusion.py
LABELS=$BASE/code/config/vehicle_labels.v2.json
BASELINE=$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt
PRODUCTION_CONFIG=$BASE/code/stage159_component_validation_r6/production_vehicle_analytics.yaml
BODY_MANIFEST=$BASE/datasets/attribute-domain-v2/stage159-validation-views-r3/body.validation-only.csv
STAGE163_ROOT=$BASE/runs/attributes/ATTR-STAGE163-BODY-HARDCLASS-R1
STAGE163_STATE=$STAGE163_ROOT.state.json
OUTPUT_ROOT=$BASE/runs/attributes/ATTR-STAGE164-BODY-VALIDATION-R1
OUTER_LOG=$BASE/runs/attributes/ATTR-STAGE164-BODY-VALIDATION-R1.log
STATE=$BASE/runs/attributes/ATTR-STAGE164-BODY-VALIDATION-R1.state.json

test ! -e "$OUTPUT_ROOT"
test ! -e "$STATE"
exec > >(tee -a "$OUTER_LOG") 2>&1

assert_sha256() {
  local path="$1" expected="$2" actual
  test -f "$path"
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  test "$actual" = "${expected,,}"
}

assert_sha256 "$EVALUATOR" c7d976e6ccd8604cc0c019567c2c5d01549854cca66de09a6b1065fb3306f05e
assert_sha256 "$EVALUATOR_TEST" 88352437d0a693a6d3901a63b68da37d4be4ad9c2c8ad1a2270179ca41cf0be6
assert_sha256 "$EVAL_ROOT/evaluate_stage159_component_validation.py" e8147f470e7a3d48f46d67c4b41f32bc7963c7248da1a0f640f069a9ed037e1d
assert_sha256 "$EVAL_ROOT/evaluate_stage109_color_class_thresholds.py" 5409b78eec92ddce15969b5eb6d57330037dc1a4f5b9f8cf2a918b742c7db3d4
assert_sha256 "$EVAL_ROOT/evaluate_v2_decoupled_shared_validation.py" 35d19635280299e1a894647a7b98803aca13be308a59117f15aa3c61032b0e12
assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
assert_sha256 "$BASELINE" 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383
assert_sha256 "$PRODUCTION_CONFIG" fb64dde5ef7cd2dc71188388387558169eb613f70bf9f209b44aecbc92a503fa
assert_sha256 "$BODY_MANIFEST" 061dba9efbca98a2f6f7968c4c9d38123d516fdbc408a16b3ba6f2fc2a06b406

export PYTHONPATH="$EVAL_ROOT:$CODE/scripts:${PYTHONPATH:-}"
cd "$CODE/scripts"
"$PY" -m unittest -v test_evaluate_stage164_persistent_fusion.py
"$PY" -m py_compile "$EVALUATOR"

while tmux has-session -t VCAS-STAGE163-BODY-HARDCLASS-R1 2>/dev/null; do sleep 30; done
while pgrep -f '[t]rain_attribute.py .*stage163-body-hardclass' >/dev/null 2>&1; do sleep 15; done

test -s "$STAGE163_STATE"
"$PY" - "$STAGE163_STATE" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
if data.get('status') != 'training_complete_pending_stage164_validation':
    raise SystemExit('Stage163 state is not complete')
for key in ('test_accessed', 'frozen_video_used', 'production_model_modified', 'deployment_performed'):
    if data.get(key) is not False:
        raise SystemExit(f'Stage163 isolation failed: {key}')
PY

mkdir -p "$OUTPUT_ROOT/body"
for variant in ATTR-STAGE163-BODY-CONVNEXT-256-HARD-R1 ATTR-STAGE163-BODY-CONVNEXT-256-BALANCED-R1; do
  for artifact in best last; do
    checkpoint=$STAGE163_ROOT/$variant/$artifact.pt
    test -s "$checkpoint"
    test -s "$STAGE163_ROOT/$variant/metrics.json"
    test -s "$STAGE163_ROOT/$variant/model_card.json"
    checkpoint_sha=$(sha256sum "$checkpoint" | awk '{print tolower($1)}')
    report_dir=$OUTPUT_ROOT/body/${variant}__${artifact}
    set +e
    "$PY" "$EVALUATOR" --head body \
      --manifest "$BODY_MANIFEST" --expected-manifest-sha256 061dba9efbca98a2f6f7968c4c9d38123d516fdbc408a16b3ba6f2fc2a06b406 \
      --labels "$LABELS" --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
      --candidate-checkpoint "$checkpoint" --expected-candidate-sha256 "$checkpoint_sha" \
      --baseline-checkpoint "$BASELINE" --expected-baseline-sha256 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 \
      --production-config "$PRODUCTION_CONFIG" --expected-production-config-sha256 fb64dde5ef7cd2dc71188388387558169eb613f70bf9f209b44aecbc92a503fa \
      --resolution-root "$BASE" --allowed-image-root "$BASE/datasets" --allowed-image-root "$BASE/sources" \
      --precision-target 0.935 --device cuda --batch-size 64 --workers 8 --output "$report_dir/report.json"
    code=$?
    set -e
    test "$code" = 0 -o "$code" = 2
  done
done

"$PY" - "$OUTPUT_ROOT" "$STATE" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

root, state = map(Path, sys.argv[1:])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
reports = []
for path in sorted((root / 'body').glob('*/report.json')):
    data = json.loads(path.read_text())
    policy = data.get('policy', {})
    if policy.get('split') != 'validation' or policy.get('test_accessed') is not False or policy.get('frozen_video_used') is not False:
        raise SystemExit(f'validation isolation failed: {path}')
    reports.append({
        'variant': path.parent.name,
        'qualified': bool(data['gates']['all_pass']),
        'candidate_checkpoint': data['inputs']['candidate_checkpoint'],
        'candidate_checkpoint_sha256': data['inputs']['candidate_checkpoint_sha256'],
        'overall': data['static']['candidate_overall'],
        'complex': data['static']['candidate_complex'],
        'comparison': data['comparison'],
        'track': data['track_fusion'],
        'gates': data['gates']['gates'],
        'report_sha256': sha(path),
    })
if len(reports) != 4:
    raise SystemExit(f'expected 4 reports, got {len(reports)}')
qualified = [report for report in reports if report['qualified']]
selected = max(qualified, key=lambda report: (
    report['complex']['coverage'], report['overall']['coverage'],
    report['track']['track_final']['coverage'], report['overall']['precision'],
)) if qualified else None
result = {
    'schema_version': 'stage164-body-validation-state-v1',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'status': 'body_candidate_pass' if selected else 'body_candidates_rejected_fail_closed',
    'selected': selected,
    'candidates': reports,
    'selection_split': 'validation',
    'precision_target': 0.935,
    'test_accessed': False,
    'frozen_video_used': False,
    'production_model_modified': False,
    'backend_gates_run': False,
    'deployment_performed': False,
}
state.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
Path(str(state) + '.sha256').write_text(sha(state) + '  ' + state.name + '\n')
print(json.dumps({'status': result['status'], 'selected': selected['variant'] if selected else None}))
PY

find "$OUTPUT_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$OUTPUT_ROOT/SHA256SUMS"
