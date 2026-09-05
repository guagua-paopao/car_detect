#!/usr/bin/env bash
set -euo pipefail

# Stage161 is a research-only body repair run.  It never touches production
# artifacts and never reads the frozen video or any independent test split.
BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE=/root/autodl-tmp/vcas/code/stage161-body-focus-distill-r1
TRAINER=/root/autodl-tmp/vcas/code/training_stage141_letterbox_r1/scripts/train_attribute.py
LABELS=$BASE/code/config/vehicle_labels.v2.json
SOURCE_MANIFEST=$BASE/datasets/attribute-domain-v2/stage156-fullscale-body-repair-r1/attribute_manifest.stage156-fullscale-body-repair.csv
MANIFEST=$BASE/datasets/attribute-domain-v2/stage161-body-focus-r1/attribute_manifest.stage161-body-focus.csv
MANIFEST_REPORT=$BASE/datasets/attribute-domain-v2/stage161-body-focus-r1/stage161-body-focus-report.json
UNLABELED=$BASE/datasets/attribute-domain-v2/stage70-specialist-manifests-v4-no-color-pseudo/attribute_manifest.stage70-unlabeled.csv
INIT=$BASE/runs/attributes/ATTR-STAGE156-FULLSCALE-BODY-TRAIN-R1/ATTR-STAGE156-BODY-CONVNEXT-256-STRETCH-R1/best.pt
TEACHER=$INIT
RUN_ROOT=$BASE/runs/attributes/ATTR-STAGE161-BODY-FOCUS-DISTILL-R1
RUN256=$RUN_ROOT/ATTR-STAGE161-BODY-CONVNEXT-256-FOCUS-R1
RUN288=$RUN_ROOT/ATTR-STAGE161-BODY-CONVNEXT-288-FOCUS-R1
STATE=$RUN_ROOT.state.json

assert_sha256() {
  local path="$1" expected="$2" actual
  test -f "$path"
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  test "$actual" = "${expected,,}"
}

test ! -e "$RUN_ROOT" && test ! -e "$STATE" && test ! -e "$STATE.sha256"
assert_sha256 "$TRAINER" ca31774366304d04e43425e38ce5d209ac85b8732d6aaa706ba66abe1bc6c6a1
assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
assert_sha256 "$SOURCE_MANIFEST" d9f4e6f8718f389f1669b20933bb5089382bc84fba23dcd0f89a0bb5166194d5
assert_sha256 "$UNLABELED" f715881cfdc2a3d9005038822036588f612d887c6697af966abd5b7d0ebc5c04
assert_sha256 "$INIT" 3f92de00f0d77b73ce8fedb423a743e4770c0befca6311f49003bc80a5f3a338
mkdir -p "$CODE/scripts" "$RUN_ROOT"

assert_sha256 "$CODE/scripts/build_stage161_body_focus_manifest.py" 3124e581b4e65b06c4f12b7f00b630d0e4348cd44a0af982f6e6bddd4ba40246
"$PY" "$CODE/scripts/build_stage161_body_focus_manifest.py" \
  --source "$SOURCE_MANIFEST" --expected-source-sha256 d9f4e6f8718f389f1669b20933bb5089382bc84fba23dcd0f89a0bb5166194d5 \
  --output "$MANIFEST" --report "$MANIFEST_REPORT"

common=(
  --manifest "$MANIFEST" --labels "$LABELS" --architecture convnext_tiny
  --epochs 8 --workers 8 --learning-rate 0.000001 --weight-decay 0.0001
  --body-loss-weight 1.0 --color-loss-weight 0.0 --focal-gamma 1.5 --color-focal-gamma 0.0
  --class-weighting inverse_sqrt --label-smoothing 0.01 --gradient-clip-norm 5.0
  --freeze-backbone-epochs 0 --patience 4 --type-threshold 0.80 --color-threshold 0.99
  --gate-type-precision 0.93 --gate-type-coverage 0.45 --gate-color-precision 0.93 --gate-color-coverage 0.25
  --seed 20260830 --augmentation-profile hard_scene --selection-head body
  --body-hierarchy truck_family --truck-subtype-threshold 0.65
  --coarse-car-loss-weight 0.25 --coarse-truck-loss-weight 0.10
  --night-sample-weight 6.0 --occlusion-sample-weight 1.5 --hard-sample-weight 0.25 --small-sample-weight 1.5
  --color-sample-weight 0.0 --pseudo-label-weight 1.0
  --body-teacher-checkpoint "$TEACHER" --distill-weight 0.20 --distill-body-weight 1.0 --distill-color-weight 0.0
  --distill-temperature 2.0
  --unlabeled-manifest "$UNLABELED" --unlabeled-root "$BASE/datasets" --unlabeled-batch-size 64
  --unlabeled-consistency-weight 0.05 --unlabeled-consistency-mode feature --unlabeled-body-weight 1.0
  --unlabeled-color-weight 0.0 --unlabeled-consistency-temperature 1.0 --unlabeled-ema-decay 0.999
  --unlabeled-night-sample-weight 4.0 --init-checkpoint "$INIT" --run-kind formal --skip-test
)

"$PY" "$TRAINER" "${common[@]}" --input-size 256 --resize-mode stretch --batch-size 40 \
  --dataset-version attribute-domain-v2-stage161-body-focus-r1 --code-revision stage161-body-focus-distill-256-r1 \
  --output-dir "$RUN256"
"$PY" "$TRAINER" "${common[@]}" --input-size 288 --resize-mode letterbox --batch-size 32 \
  --dataset-version attribute-domain-v2-stage161-body-focus-r1 --code-revision stage161-body-focus-distill-288-r1 \
  --output-dir "$RUN288"

"$PY" - "$STATE" "$MANIFEST" "$MANIFEST_REPORT" "$RUN256" "$RUN288" "$TRAINER" "$LABELS" "$INIT" "$UNLABELED" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

state, manifest, manifest_report, run256, run288, trainer, labels, init, unlabeled = map(Path, sys.argv[1:])
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()
def artifact(run, name):
    p = run / name
    if not p.is_file(): raise SystemExit(f'missing artifact: {p}')
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}
def metrics(run):
    p = run / 'metrics.json'
    return json.loads(p.read_text()) if p.is_file() else {}
out = {
  'schema_version': 'stage161-body-focus-distill-state-v1',
  'created_at': datetime.now(timezone.utc).isoformat(),
  'status': 'training_complete_pending_stage162_validation',
  'research_only': True,
  'strategy': {'class_focus': 'suv/mpv/van/other/pickup', 'body_teacher': str(init), 'distill_weight': 0.20,
               'unlabeled_feature_consistency_weight': 0.05, 'sequence_stability_proxy': 'EMA feature consistency'},
  'inputs': {k: {'path': str(p), 'sha256': sha(p)} for k,p in {
      'manifest': manifest, 'manifest_report': manifest_report, 'trainer': trainer,
      'labels': labels, 'initialization': init, 'unlabeled_manifest': unlabeled}.items()},
  'candidates': {},
  'test_accessed': False, 'stage148_test_reused': False, 'stage155_holdout_reused': False,
  'frozen_video_used': False, 'production_model_modified': False, 'deployment_performed': False,
}
for name, run in [('convnext_tiny_256_focus', run256), ('convnext_tiny_288_focus', run288)]:
    out['candidates'][name] = {'run_dir': str(run), 'metrics': metrics(run),
      'artifacts': {n: artifact(run, n) for n in ('best.pt','last.pt','metrics.json','model_card.json')},
      'test_status': 'not_run'}
state.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
(Path(str(state) + '.sha256')).write_text(sha(state) + '  ' + state.name + '\n')
print(json.dumps({'status': out['status'], 'state': str(state)}, ensure_ascii=False))
PY
