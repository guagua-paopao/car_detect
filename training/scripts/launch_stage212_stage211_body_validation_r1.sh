#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE=$BASE/code/stage168-body-validation-r1
EVAL_ROOT=$BASE/code/stage159_component_validation_r6/scripts
EVALUATOR=$CODE/scripts/evaluate_stage166_selective_short_tracks.py
STAGE165_EVALUATOR=$CODE/scripts/evaluate_stage165_track_frontier.py
EVALUATOR_TEST=$CODE/scripts/test_evaluate_stage166_selective_short_tracks.py
LABELS=$BASE/code/config/vehicle_labels.v2.json
BASELINE=$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt
PRODUCTION_CONFIG=$BASE/code/stage159_component_validation_r6/production_vehicle_analytics.yaml
BODY_MANIFEST=$BASE/datasets/attribute-domain-v2/stage159-validation-views-r3/body.validation-only.csv
STAGE211_ROOT=$BASE/runs/attributes/ATTR-STAGE211-INATRC-TOLL-HARDCLASS-R1
STAGE211_RUN=$STAGE211_ROOT/ATTR-STAGE211-BODY-CONVNEXT-256-INATRC-HARDCLASS-R1
STAGE211_STATE=$STAGE211_ROOT.state.json
OUTPUT_ROOT=$BASE/runs/attributes/ATTR-STAGE212-STAGE211-BODY-VALIDATION-R1
OUTER_LOG=$BASE/runs/attributes/ATTR-STAGE212-STAGE211-BODY-VALIDATION-R1.log
STATE=$BASE/runs/attributes/ATTR-STAGE212-STAGE211-BODY-VALIDATION-R1.state.json

test ! -e "$OUTPUT_ROOT"
test ! -e "$STATE"
exec > >(tee -a "$OUTER_LOG") 2>&1

assert_sha256() {
  local path="$1" expected="$2" actual
  test -f "$path"
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  test "$actual" = "${expected,,}"
}

assert_sha256 "$EVALUATOR" b5eab219f060ee6bc6c7ef4d31dbb80694af145801c27c7ebb5d56ba92728cd9
assert_sha256 "$STAGE165_EVALUATOR" 99274dee327d618f1b30066c18f2080c92516765dd8eafa243c6cee57854cf87
assert_sha256 "$EVALUATOR_TEST" db0841fa57c9747d591db34428f7fa354c667df5b93c4a7627d0f625cb3234af
assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
assert_sha256 "$BASELINE" 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383
assert_sha256 "$PRODUCTION_CONFIG" fb64dde5ef7cd2dc71188388387558169eb613f70bf9f209b44aecbc92a503fa
assert_sha256 "$BODY_MANIFEST" 061dba9efbca98a2f6f7968c4c9d38123d516fdbc408a16b3ba6f2fc2a06b406

export PYTHONPATH="$EVAL_ROOT:$CODE/scripts:${PYTHONPATH:-}"
cd "$CODE/scripts"
"$PY" -m unittest -v test_evaluate_stage166_selective_short_tracks.py
"$PY" -m py_compile "$EVALUATOR" "$STAGE165_EVALUATOR"

while tmux has-session -t VCAS-STAGE211-INATRC-TOLL-HARDCLASS-R1 2>/dev/null; do sleep 30; done
while pgrep -f '[t]rain_attribute.py .*stage211-inatrc-toll-hardclass' >/dev/null 2>&1; do sleep 15; done

test -s "$STAGE211_STATE"
"$PY" - "$STAGE211_STATE" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
if data.get('status') != 'training_complete_pending_stage212_validation':
    raise SystemExit('Stage211 state is not complete')
for key in ('test_accessed', 'frozen_video_used', 'production_model_modified', 'deployment_performed'):
    if data.get(key) is not False:
        raise SystemExit(f'Stage211 isolation failed: {key}')
PY

mkdir -p "$OUTPUT_ROOT/body"
for artifact in best last; do
  checkpoint=$STAGE211_RUN/$artifact.pt
  checkpoint_sha=$(sha256sum "$checkpoint" | awk '{print tolower($1)}')
  report_dir=$OUTPUT_ROOT/body/ATTR-STAGE211-BODY-CONVNEXT-256-INATRC-HARDCLASS-R1__${artifact}
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
    track = data['track_fusion']
    if policy.get('split') != 'validation' or policy.get('test_accessed') is not False or policy.get('frozen_video_used') is not False:
        raise SystemExit(f'validation isolation failed: {path}')
    if track['selection_search']['base_thresholds_never_lowered'] is not True:
        raise SystemExit(f'threshold policy failed: {path}')
    reports.append({
        'variant': path.parent.name,
        'qualified': bool(data['gates']['all_pass']),
        'candidate_checkpoint': data['inputs']['candidate_checkpoint'],
        'candidate_checkpoint_sha256': data['inputs']['candidate_checkpoint_sha256'],
        'overall': data['static']['candidate_overall'],
        'complex': data['static']['candidate_complex'],
        'comparison': data['comparison'],
        'track': track,
        'gates': data['gates']['gates'],
        'report_path': str(path),
        'report_sha256': sha(path),
    })
if len(reports) != 2:
    raise SystemExit(f'expected 2 reports, got {len(reports)}')
qualified = [report for report in reports if report['qualified']]
selected = max(qualified, key=lambda report: (
    report['complex']['coverage'], report['overall']['coverage'],
    report['track']['track_final']['coverage'], report['overall']['precision'],
)) if qualified else None
result = {
    'schema_version': 'stage212-stage211-body-validation-state-v1',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'status': 'body_candidate_pass' if selected else 'body_candidate_rejected_fail_closed',
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
