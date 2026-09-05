#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
TRAINER="$BASE/code/stage158_fresh_color_train_r2/scripts/train_attribute.py"
LABELS="$BASE/code/config/vehicle_labels.v2.json"
MANIFEST="$BASE/datasets/attribute-domain-v2/stage242-nightowls-color-repair-r1/attribute_manifest.stage242-nightowls-color-repair.csv"
MANIFEST_REPORT="$BASE/datasets/attribute-domain-v2/stage242-nightowls-color-repair-r1/stage242-nightowls-color-repair-manifest.json"
INIT="$BASE/runs/attributes/ATTR-STAGE158-FRESH-COLOR-TRAIN-R2/ATTR-STAGE158-COLOR-CONVNEXT-256-FRESH-R1/best.pt"
ROOT="$BASE/runs/attributes/ATTR-STAGE243-NIGHTOWLS-COLOR-REPAIR-R1"
RUN_A="$ROOT/ATTR-STAGE243-COLOR-CONVNEXT-256-NIGHT-W12-R1"
RUN_B="$ROOT/ATTR-STAGE243-COLOR-CONVNEXT-256-NIGHT-W20-R1"
STATE="$ROOT/state.json"
LOG="$BASE/logs/stage243-nightowls-color-repair-r1.log"
SESSION=VCAS-STAGE243-NIGHTOWLS-COLOR-REPAIR-R1

assert_sha256() {
  local path="$1" expected="$2" actual
  test -f "$path"
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  test "$actual" = "${expected,,}"
}

assert_sha256 "$TRAINER" 8f657b3a4dde5cee366fb0f3af159d93698daac09d4ae6103f68435c2d64bd0c
assert_sha256 "$LABELS" 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f
assert_sha256 "$MANIFEST" 445babeb94649f32b208db080e2c2085451c26ca25ad9187b37ce6c2dd9112f8
assert_sha256 "$MANIFEST_REPORT" 99d765053f15a9d9fa2ce9c53c812430b6b090e73cb4aca6bbc34e2dc3f85353
assert_sha256 "$INIT" f03835963fc1fc48129de1004d52057b9677d02afc8bb33b8f1401881e4cd10d
"$PY" -m py_compile "$TRAINER"
"$PY" - "$MANIFEST_REPORT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding='utf-8'))
assert report['status'] == 'pass_research_only_targeted_night_repair'
assert report['added']['rows'] == 264
assert report['added']['tracks'] == 88
assert report['policy']['test_rows_used'] is False
assert report['policy']['frozen_video_used'] is False
assert report['policy']['production_model_modified'] is False
PY

if [[ "${STAGE243_WORKER:-0}" != "1" ]]; then
  test ! -e "$ROOT"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session already exists: $SESSION" >&2
    exit 67
  fi
  tmux new-session -d -s "$SESSION" \
    "STAGE243_WORKER=1 bash '$BASE/code/stage243-nightowls-color-repair-r1/launch_stage243_nightowls_color_repair_train_r1.sh'"
  echo "started tmux:$SESSION"
  echo "root=$ROOT"
  echo "log=$LOG"
  exit 0
fi

test ! -e "$ROOT"
available_kib="$(df --output=avail -k "$BASE" | tail -n 1 | tr -d ' ')"
test "$available_kib" -ge 10485760
mkdir -p "$(dirname "$LOG")"

common=(
  --manifest "$MANIFEST" --labels "$LABELS" --architecture convnext_tiny
  --input-size 256 --resize-mode stretch --epochs 4 --batch-size 40 --workers 8
  --weight-decay 0.0001 --body-loss-weight 0.0 --color-loss-weight 1.3
  --focal-gamma 0.0 --color-focal-gamma 1.5 --class-weighting inverse_sqrt
  --sampler-color-class-weighting inverse_sqrt --sampler-color-class-weight-cap 4.0
  --label-smoothing 0.02 --gradient-clip-norm 5.0 --freeze-backbone-epochs 0 --patience 2
  --type-threshold 0.99 --color-threshold 0.70
  --gate-type-precision 0.93 --gate-type-coverage 0.45
  --gate-color-precision 0.93 --gate-color-coverage 0.25
  --seed 20260903 --augmentation-profile color_scene --selection-head color
  --body-hierarchy none --coarse-car-loss-weight 0.0 --coarse-truck-loss-weight 0.0
  --coarse-color-loss-weight 0.5 --occlusion-sample-weight 1.1
  --hard-sample-weight 0.1 --small-sample-weight 0.5 --color-sample-weight 0.25
  --pseudo-label-weight 0.5 --init-checkpoint "$INIT"
  --color-teacher-checkpoint "$INIT" --distill-body-weight 0.0 --distill-color-weight 1.0
  --distill-temperature 2.0 --run-kind formal --skip-test
)

mkdir -p "$ROOT"
exec > >(tee -a "$LOG") 2>&1
echo "Stage243 candidate A: conservative night repair"
"$PY" "$TRAINER" "${common[@]}" \
  --learning-rate 0.000001 --night-sample-weight 12.0 --distill-weight 0.15 \
  --dataset-version attribute-domain-v2-stage243-nightowls-color-w12-r1 \
  --code-revision stage243-nightowls-color-w12-r1 --output-dir "$RUN_A"

echo "Stage243 candidate B: moderate night repair"
"$PY" "$TRAINER" "${common[@]}" \
  --learning-rate 0.000002 --night-sample-weight 20.0 --distill-weight 0.08 \
  --dataset-version attribute-domain-v2-stage243-nightowls-color-w20-r1 \
  --code-revision stage243-nightowls-color-w20-r1 --output-dir "$RUN_B"

"$PY" - "$STATE" "$MANIFEST" "$MANIFEST_REPORT" "$INIT" "$RUN_A" "$RUN_B" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

state, manifest, report, init, run_a, run_b = map(Path, sys.argv[1:])
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
def candidate(path):
    artifacts = {}
    for name in ('best.pt', 'last.pt', 'metrics.json', 'model_card.json'):
        item = path / name
        if not item.is_file():
            raise SystemExit(f'missing artifact: {item}')
        artifacts[name] = {'path': str(item), 'sha256': sha(item), 'bytes': item.stat().st_size}
    return {'path': str(path), 'metrics': json.loads((path / 'metrics.json').read_text()), 'artifacts': artifacts}
result = {
    'schema_version': 'stage243-nightowls-color-repair-state-v1',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'status': 'training_complete_pending_stage244_validation',
    'research_only': True,
    'deployment_eligible': False,
    'manifest': {'path': str(manifest), 'sha256': sha(manifest)},
    'manifest_report': {'path': str(report), 'sha256': sha(report)},
    'initial_checkpoint': {'path': str(init), 'sha256': sha(init)},
    'candidates': {'night_weight_12': candidate(run_a), 'night_weight_20': candidate(run_b)},
    'test_accessed': False,
    'frozen_video_used': False,
    'production_model_modified': False,
    'deployment_performed': False,
}
state.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
Path(str(state) + '.sha256').write_text(sha(state) + '  ' + state.name + '\n')
print(json.dumps({'status': result['status'], 'state': str(state)}))
PY
