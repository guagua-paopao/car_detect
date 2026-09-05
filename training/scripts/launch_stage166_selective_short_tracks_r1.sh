#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE=$BASE/code/stage166-selective-short-tracks-r1
EVAL_ROOT=$BASE/code/stage159_component_validation_r6/scripts
EVALUATOR=$CODE/scripts/evaluate_stage166_selective_short_tracks.py
STAGE165_EVALUATOR=$CODE/scripts/evaluate_stage165_track_frontier.py
EVALUATOR_TEST=$CODE/scripts/test_evaluate_stage166_selective_short_tracks.py
LABELS=$BASE/code/config/vehicle_labels.v2.json
BASELINE=$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt
PRODUCTION_CONFIG=$BASE/code/stage159_component_validation_r6/production_vehicle_analytics.yaml
BODY_MANIFEST=$BASE/datasets/attribute-domain-v2/stage159-validation-views-r3/body.validation-only.csv
STAGE163_ROOT=$BASE/runs/attributes/ATTR-STAGE163-BODY-HARDCLASS-R1
OUTPUT_ROOT=$BASE/runs/attributes/ATTR-STAGE166-SELECTIVE-SHORT-TRACKS-R1
OUTER_LOG=$BASE/runs/attributes/ATTR-STAGE166-SELECTIVE-SHORT-TRACKS-R1.log
STATE=$BASE/runs/attributes/ATTR-STAGE166-SELECTIVE-SHORT-TRACKS-R1.state.json

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
assert_sha256 "$EVAL_ROOT/evaluate_stage159_component_validation.py" e8147f470e7a3d48f46d67c4b41f32bc7963c7248da1a0f640f069a9ed037e1d
assert_sha256 "$EVAL_ROOT/evaluate_stage109_color_class_thresholds.py" 5409b78eec92ddce15969b5eb6d57330037dc1a4f5b9f8cf2a918b742c7db3d4
assert_sha256 "$EVAL_ROOT/evaluate_v2_decoupled_shared_validation.py" 35d19635280299e1a894647a7b98803aca13be308a59117f15aa3c61032b0e12
assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
assert_sha256 "$BASELINE" 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383
assert_sha256 "$PRODUCTION_CONFIG" fb64dde5ef7cd2dc71188388387558169eb613f70bf9f209b44aecbc92a503fa
assert_sha256 "$BODY_MANIFEST" 061dba9efbca98a2f6f7968c4c9d38123d516fdbc408a16b3ba6f2fc2a06b406

export PYTHONPATH="$EVAL_ROOT:$CODE/scripts:${PYTHONPATH:-}"
cd "$CODE/scripts"
"$PY" -m unittest -v test_evaluate_stage166_selective_short_tracks.py
"$PY" -m py_compile "$EVALUATOR" "$STAGE165_EVALUATOR"

mkdir -p "$OUTPUT_ROOT/body"
for artifact in best last; do
  checkpoint=$STAGE163_ROOT/ATTR-STAGE163-BODY-CONVNEXT-256-BALANCED-R1/$artifact.pt
  expected_sha=
  case "$artifact" in
    best) expected_sha=5e8b95922cb0a2fac2f3cd289be7b00f30018cf2ccc8d0c43376308346720d5b ;;
    last) expected_sha=7b7af2b27066f6eb0c60278347c90873813bf8f2b063e4823b7e3d3a9eddfd15 ;;
  esac
  assert_sha256 "$checkpoint" "$expected_sha"
  report_dir=$OUTPUT_ROOT/body/balanced-$artifact
  set +e
  "$PY" "$EVALUATOR" --head body \
    --manifest "$BODY_MANIFEST" --expected-manifest-sha256 061dba9efbca98a2f6f7968c4c9d38123d516fdbc408a16b3ba6f2fc2a06b406 \
    --labels "$LABELS" --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
    --candidate-checkpoint "$checkpoint" --expected-candidate-sha256 "$expected_sha" \
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
    if policy.get('split') != 'validation' or policy.get('test_accessed') is not False or policy.get('frozen_video_used') is not False:
        raise SystemExit(f'validation isolation failed: {path}')
    track = data['track_fusion']
    if track['selection_search']['base_thresholds_never_lowered'] is not True:
        raise SystemExit(f'threshold policy failed: {path}')
    reports.append({
        'variant': path.parent.name,
        'qualified': bool(data['gates']['all_pass']),
        'candidate_checkpoint_sha256': data['inputs']['candidate_checkpoint_sha256'],
        'overall': data['static']['candidate_overall'],
        'complex': data['static']['candidate_complex'],
        'comparison': data['comparison'],
        'track': track,
        'gates': data['gates']['gates'],
        'report_sha256': sha(path),
    })
if len(reports) != 2:
    raise SystemExit(f'expected 2 reports, got {len(reports)}')
qualified = [report for report in reports if report['qualified']]
selected = max(qualified, key=lambda report: (
    report['complex']['coverage'], report['track']['track_final']['coverage'],
    report['overall']['coverage'], report['overall']['precision'],
)) if qualified else None
result = {
    'schema_version': 'stage166-selective-short-tracks-state-v1',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'status': 'body_candidate_pass' if selected else 'body_candidates_rejected_fail_closed',
    'selected': selected,
    'candidates': reports,
    'selection_split': 'validation',
    'purpose': 'selectively admit reliable singleton and short tracks without lowering static class thresholds',
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
