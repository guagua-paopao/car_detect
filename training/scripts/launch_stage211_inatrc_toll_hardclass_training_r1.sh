#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
TRAINER=$BASE/code/training_stage141_letterbox_r1/scripts/train_attribute.py
LABELS=$BASE/code/config/vehicle_labels.v2.json
DATA_ROOT=$BASE/datasets/attribute-domain-v2/stage211-inatrc-toll-hardclass-r2
MANIFEST=$DATA_ROOT/attribute_manifest.stage211-inatrc-toll-hardclass.csv
UNLABELED=$DATA_ROOT/attribute_manifest.stage211-unlabeled.csv
MANIFEST_REPORT=$DATA_ROOT/stage211-inatrc-toll-hardclass-manifest-report.json
INIT=$BASE/runs/attributes/ATTR-STAGE167-COMPLEX-BODY-R1/ATTR-STAGE167-BODY-CONVNEXT-256-STRETCH-R1/best.pt
RUN_ROOT=$BASE/runs/attributes/ATTR-STAGE211-INATRC-TOLL-HARDCLASS-R1
RUN=$RUN_ROOT/ATTR-STAGE211-BODY-CONVNEXT-256-INATRC-HARDCLASS-R1
STATE=$RUN_ROOT.state.json
LOG=$RUN_ROOT.log

test ! -e "$RUN_ROOT"
test ! -e "$STATE"
test ! -e "$LOG"

assert_sha256() {
  local path="$1" expected="$2" actual
  test -f "$path"
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  test "$actual" = "${expected,,}"
}

assert_sha256 "$TRAINER" ca31774366304d04e43425e38ce5d209ac85b8732d6aaa706ba66abe1bc6c6a1
assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
assert_sha256 "$MANIFEST" a6344c55b2a6ce98533e0234f7cfd0699224ae2c1d44609067eced907dc3b69f
assert_sha256 "$UNLABELED" ffd93dafa0898e9e8652a6bb90fa8a68fd4287c55ce6877455009f06d5db20b6
assert_sha256 "$MANIFEST_REPORT" b0e9ea46ddd1e011ebf24317cb34a516f9bef3632375c0bff35f3c2bae26818c
assert_sha256 "$INIT" c6dea9d50a01fd6395e403606083e955d8880f9f9e77048d0d5f28c487b11f0b

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$LOG") 2>&1

"$PY" "$TRAINER" \
  --manifest "$MANIFEST" --labels "$LABELS" --architecture convnext_tiny \
  --epochs 6 --workers 8 --batch-size 40 --input-size 256 --resize-mode stretch \
  --learning-rate 0.0000007 --weight-decay 0.0001 \
  --body-loss-weight 1.0 --color-loss-weight 0.0 --color-focal-gamma 0.0 \
  --class-weighting inverse_sqrt --focal-gamma 1.0 --label-smoothing 0.005 \
  --gradient-clip-norm 5.0 --freeze-backbone-epochs 0 --patience 3 \
  --type-threshold 0.80 --color-threshold 0.99 \
  --gate-type-precision 0.93 --gate-type-coverage 0.45 \
  --gate-color-precision 0.93 --gate-color-coverage 0.25 \
  --seed 20260903 --augmentation-profile hard_scene --selection-head body \
  --body-hierarchy truck_family --truck-subtype-threshold 0.65 \
  --coarse-car-loss-weight 0.25 --coarse-truck-loss-weight 0.10 \
  --night-sample-weight 1.0 --occlusion-sample-weight 1.0 \
  --hard-sample-weight 0.50 --small-sample-weight 0.0 \
  --color-sample-weight 0.0 --pseudo-label-weight 1.0 \
  --body-teacher-checkpoint "$INIT" --distill-weight 0.10 \
  --distill-body-weight 1.0 --distill-color-weight 0.0 --distill-temperature 2.0 \
  --unlabeled-manifest "$UNLABELED" --unlabeled-root "$BASE/datasets" \
  --unlabeled-batch-size 64 --unlabeled-consistency-mode feature \
  --unlabeled-body-weight 1.0 --unlabeled-color-weight 0.0 \
  --unlabeled-consistency-weight 0.04 --unlabeled-consistency-temperature 1.0 \
  --unlabeled-ema-decay 0.999 --unlabeled-night-sample-weight 4.0 \
  --init-checkpoint "$INIT" --run-kind formal --skip-test \
  --dataset-version attribute-domain-v2-stage211-inatrc-toll-hardclass-r2 \
  --code-revision stage211-inatrc-toll-hardclass-256-stretch-r1 \
  --output-dir "$RUN"

"$PY" - "$STATE" "$MANIFEST" "$UNLABELED" "$MANIFEST_REPORT" "$RUN" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

state, manifest, unlabeled, manifest_report, run = map(Path, sys.argv[1:])
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
artifacts = {}
for name in ('best.pt', 'last.pt', 'metrics.json', 'model_card.json'):
    path = run / name
    if not path.is_file():
        raise SystemExit(f'missing artifact: {path}')
    artifacts[name] = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
result = {
    'schema_version': 'stage211-inatrc-toll-hardclass-training-state-v1',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'status': 'training_complete_pending_stage212_validation',
    'research_only': True,
    'deployment_eligible': False,
    'candidate': str(run),
    'manifest': {'path': str(manifest), 'sha256': sha(manifest)},
    'unlabeled_manifest': {'path': str(unlabeled), 'sha256': sha(unlabeled)},
    'manifest_report': {'path': str(manifest_report), 'sha256': sha(manifest_report)},
    'metrics': json.loads((run / 'metrics.json').read_text()),
    'artifacts': artifacts,
    'test_accessed': False,
    'frozen_video_used': False,
    'production_model_modified': False,
    'backend_gates_run': False,
    'deployment_performed': False,
}
state.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
Path(str(state) + '.sha256').write_text(sha(state) + '  ' + state.name + '\n')
print(json.dumps({'status': result['status'], 'state': str(state), 'best_sha256': artifacts['best.pt']['sha256']}))
PY
