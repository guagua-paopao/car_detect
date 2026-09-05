#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE=$BASE/code/stage167-complex-body-r1
TRAINER=$BASE/code/training_stage141_letterbox_r1/scripts/train_attribute.py
BUILDER=$CODE/scripts/build_stage167_complex_body_manifest.py
BUILDER_TEST=$CODE/scripts/test_build_stage167_complex_body_manifest.py
LABELS=$BASE/code/config/vehicle_labels.v2.json
SOURCE=$BASE/datasets/attribute-domain-v2/stage163-body-hardclass-r1/attribute_manifest.stage163-body-hardclass.csv
MANIFEST=$BASE/datasets/attribute-domain-v2/stage167-complex-body-r1/attribute_manifest.stage167-complex-body.csv
MANIFEST_REPORT=$BASE/datasets/attribute-domain-v2/stage167-complex-body-r1/stage167-complex-body-report.json
UNLABELED=$BASE/datasets/attribute-domain-v2/stage70-specialist-manifests-v4-no-color-pseudo/attribute_manifest.stage70-unlabeled.csv
STAGE163=$BASE/runs/attributes/ATTR-STAGE163-BODY-HARDCLASS-R1/ATTR-STAGE163-BODY-CONVNEXT-256-BALANCED-R1
INIT=$STAGE163/best.pt
TEACHER=$STAGE163/best.pt
RUN_ROOT=$BASE/runs/attributes/ATTR-STAGE167-COMPLEX-BODY-R1
RUN_A=$RUN_ROOT/ATTR-STAGE167-BODY-CONVNEXT-256-STRETCH-R1
RUN_B=$RUN_ROOT/ATTR-STAGE167-BODY-CONVNEXT-256-LETTERBOX-R1
STATE=$RUN_ROOT.state.json

assert_sha256() {
  local path="$1" expected="$2" actual
  test -f "$path"
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  test "$actual" = "${expected,,}"
}

test ! -e "$RUN_ROOT"
test ! -e "$STATE"
test ! -e "$STATE.sha256"
test ! -e "$(dirname "$MANIFEST")"
assert_sha256 "$TRAINER" ca31774366304d04e43425e38ce5d209ac85b8732d6aaa706ba66abe1bc6c6a1
assert_sha256 "$BUILDER" a0684847ac764842dfe12bbdb561e5ae462554c57525fac085c7414ea9e33e71
assert_sha256 "$BUILDER_TEST" 14e43e750ab5e484faf3088543fcd0394d50330414839653a05c28da8e4996b0
assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
assert_sha256 "$SOURCE" 8963c1fcd60abb7553a108483b775d6d3fe084f44443a766ad568242c43065d9
assert_sha256 "$UNLABELED" f715881cfdc2a3d9005038822036588f612d887c6697af966abd5b7d0ebc5c04
assert_sha256 "$INIT" 5e8b95922cb0a2fac2f3cd289be7b00f30018cf2ccc8d0c43376308346720d5b

cd "$CODE/scripts"
"$PY" -m unittest -v test_build_stage167_complex_body_manifest.py
"$PY" -m py_compile "$BUILDER"
"$PY" "$BUILDER" \
  --source "$SOURCE" \
  --expected-source-sha256 8963c1fcd60abb7553a108483b775d6d3fe084f44443a766ad568242c43065d9 \
  --output "$MANIFEST" \
  --report "$MANIFEST_REPORT"

while tmux has-session -t VCAS-STAGE166-SELECTIVE-SHORT-TRACKS-R1 2>/dev/null; do sleep 15; done
while pgrep -f '[e]valuate_stage166_selective_short_tracks.py' >/dev/null 2>&1; do sleep 15; done

common=(
  --manifest "$MANIFEST" --labels "$LABELS" --architecture convnext_tiny
  --epochs 6 --workers 8 --weight-decay 0.0001
  --body-loss-weight 1.0 --color-loss-weight 0.0 --color-focal-gamma 0.0
  --class-weighting inverse_sqrt --gradient-clip-norm 5.0 --freeze-backbone-epochs 0 --patience 3
  --type-threshold 0.80 --color-threshold 0.99
  --gate-type-precision 0.93 --gate-type-coverage 0.45 --gate-color-precision 0.93 --gate-color-coverage 0.25
  --seed 20260902 --augmentation-profile hard_scene --selection-head body
  --body-hierarchy truck_family --truck-subtype-threshold 0.65
  --coarse-car-loss-weight 0.25 --coarse-truck-loss-weight 0.10
  --night-sample-weight 1.0 --occlusion-sample-weight 1.0 --hard-sample-weight 0.50 --small-sample-weight 0.0
  --color-sample-weight 0.0 --pseudo-label-weight 1.0
  --body-teacher-checkpoint "$TEACHER" --distill-body-weight 1.0 --distill-color-weight 0.0 --distill-temperature 2.0
  --unlabeled-manifest "$UNLABELED" --unlabeled-root "$BASE/datasets" --unlabeled-batch-size 64
  --unlabeled-consistency-mode feature --unlabeled-body-weight 1.0 --unlabeled-color-weight 0.0
  --unlabeled-consistency-temperature 1.0 --unlabeled-ema-decay 0.999 --unlabeled-night-sample-weight 3.0
  --input-size 256 --batch-size 40 --run-kind formal --skip-test
  --focal-gamma 1.0 --label-smoothing 0.005 --init-checkpoint "$INIT"
)

"$PY" "$TRAINER" "${common[@]}" \
  --resize-mode stretch --learning-rate 0.0000005 \
  --distill-weight 0.15 --unlabeled-consistency-weight 0.03 \
  --dataset-version attribute-domain-v2-stage167-complex-body-r1 \
  --code-revision stage167-complex-body-256-stretch-r1 --output-dir "$RUN_A"

"$PY" "$TRAINER" "${common[@]}" \
  --resize-mode letterbox --learning-rate 0.0000008 \
  --distill-weight 0.08 --unlabeled-consistency-weight 0.03 \
  --dataset-version attribute-domain-v2-stage167-complex-body-r1 \
  --code-revision stage167-complex-body-256-letterbox-r1 --output-dir "$RUN_B"

"$PY" - "$STATE" "$MANIFEST" "$MANIFEST_REPORT" "$RUN_A" "$RUN_B" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

state, manifest, manifest_report, run_a, run_b = map(Path, sys.argv[1:])
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
def candidate(run):
    artifacts = {}
    for name in ('best.pt', 'last.pt', 'metrics.json', 'model_card.json'):
        path = run / name
        if not path.is_file(): raise SystemExit(f'missing artifact: {path}')
        artifacts[name] = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
    return {'run_dir': str(run), 'metrics': json.loads((run / 'metrics.json').read_text()), 'artifacts': artifacts}
result = {
    'schema_version': 'stage167-complex-body-state-v1',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'status': 'training_complete_pending_stage168_validation',
    'research_only': True,
    'manifest': {'path': str(manifest), 'sha256': sha(manifest)},
    'manifest_report': {'path': str(manifest_report), 'sha256': sha(manifest_report)},
    'candidates': {'stretch': candidate(run_a), 'letterbox': candidate(run_b)},
    'test_accessed': False,
    'frozen_video_used': False,
    'production_model_modified': False,
    'deployment_performed': False,
}
state.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
Path(str(state) + '.sha256').write_text(sha(state) + '  ' + state.name + '\n')
print(json.dumps({'status': result['status'], 'state': str(state)}))
PY
