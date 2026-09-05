#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
SCRIPT="$BASE/code/stage246-nightowls-validation-pilot-r1/download_stage246_nightowls_validation_pilot.py"
ANNOTATIONS="$BASE/sources/nightowls-stage245-validation-metadata/nightowls_validation.json"
LICENSE_HTML="$BASE/sources/nightowls-stage232/official-download.html"
STAGE244_STATE="$BASE/runs/attributes/ATTR-STAGE244-COLOR-VALIDATION-R2/state.json"
STAGE244_SESSION=VCAS-STAGE244-COLOR-VALIDATION-R2
SOURCE_ROOT="$BASE/sources/nightowls-stage246-validation-pilot"
RUN_ROOT="$BASE/runs/attributes/ATTR-STAGE246-NIGHTOWLS-VALIDATION-PILOT-R1"
STATE="$RUN_ROOT/state.json"
LOG="$BASE/logs/stage246-nightowls-validation-pilot-r1.log"
SESSION=VCAS-STAGE246-NIGHTOWLS-VALIDATION-PILOT-R1

assert_sha256() {
  local path="$1" expected="$2" actual
  test -f "$path"
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  test "$actual" = "${expected,,}"
}

assert_sha256 "$SCRIPT" f8dca17aa489f7318f647e395d39692671613657125abcd1eccfa6dc303739f1
assert_sha256 "$ANNOTATIONS" 584c0dc11f0d086fc5bcbaca0385cf2dd9794074f815f2727fbf9a093fd9a9b6
assert_sha256 "$LICENSE_HTML" a6d9ba9b25c371632e9674a8cdcfa6c2417b8b59468c09775d9d91a2ba26533e
"$PY" -m py_compile "$SCRIPT"

if [[ "${STAGE246_WORKER:-0}" != "1" ]]; then
  test ! -e "$RUN_ROOT"
  test ! -e "$SOURCE_ROOT"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session already exists: $SESSION" >&2
    exit 67
  fi
  tmux new-session -d -s "$SESSION" \
    "STAGE246_WORKER=1 bash '$BASE/code/stage246-nightowls-validation-pilot-r1/launch_stage246_nightowls_validation_pilot_r1.sh'"
  echo "started tmux:$SESSION"
  echo "log=$LOG"
  exit 0
fi

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "Stage246 waiting for Stage244 decision"
while tmux has-session -t "$STAGE244_SESSION" 2>/dev/null || \
      pgrep -af '[e]valuate_stage159_component_validation.py' >/dev/null 2>&1; do
  sleep 20
done

test -f "$STAGE244_STATE"
test -f "$STAGE244_STATE.sha256"
expected_state_sha="$(awk 'NR==1 {print tolower($1)}' "$STAGE244_STATE.sha256")"
assert_sha256 "$STAGE244_STATE" "$expected_state_sha"
decision="$($PY - "$STAGE244_STATE" <<'PY'
import json, sys
state=json.load(open(sys.argv[1], encoding='utf-8'))
policy=state['policy']
assert policy['test_accessed'] is False
assert policy['frozen_video_used'] is False
assert policy['production_model_modified'] is False
assert policy['deployment_performed'] is False
print('pass' if state['status']=='pass_research_candidate_selected' and state.get('selected') else 'reject')
PY
)"

test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
if [[ "$decision" != "pass" ]]; then
  "$PY" - "$STAGE244_STATE" "$STATE" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
source, target=map(Path, sys.argv[1:])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
result={
  'schema_version':'stage246-nightowls-validation-pilot-state-v1',
  'created_at':datetime.now(timezone.utc).isoformat(),
  'status':'not_started_stage244_rejected',
  'stage244_state':str(source),
  'stage244_state_sha256':sha(source),
  'validation_image_pixels_downloaded':0,
  'test_accessed':False,
  'frozen_video_used':False,
  'production_model_modified':False,
  'deployment_performed':False,
}
target.write_text(json.dumps(result, indent=2)+'\n')
Path(str(target)+'.sha256').write_text(f'{sha(target)}  {target.name}\n')
print(json.dumps(result))
PY
  exit 0
fi

"$PY" "$SCRIPT" \
  --url https://thor.robots.ox.ac.uk/nightowls/python/nightowls_validation.zip \
  --annotations "$ANNOTATIONS" \
  --license-html "$LICENSE_HTML" \
  --output-root "$SOURCE_ROOT" \
  --blocks 10 --block-size 20 --max-compressed-gib 3.0 --workers 2

REPORT="$SOURCE_ROOT/stage246-nightowls-validation-pilot-report.json"
test -f "$REPORT"
"$PY" - "$STAGE244_STATE" "$REPORT" "$SOURCE_ROOT" "$STATE" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
stage244, report_path, source_root, target=map(Path, sys.argv[1:])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
s244=json.loads(stage244.read_text(encoding='utf-8'))
report=json.loads(report_path.read_text(encoding='utf-8'))
assert s244['status']=='pass_research_candidate_selected' and s244['selected']
assert report['official_split']=='validation'
assert report['policy']['training_or_threshold_selection_allowed'] is False
assert report['policy']['color_labels_generated'] is False
assert report['policy']['test_pixels_opened']==0
assert report['policy']['frozen_video_used'] is False
result={
  'schema_version':'stage246-nightowls-validation-pilot-state-v1',
  'created_at':datetime.now(timezone.utc).isoformat(),
  'status':'complete_pixels_unlabeled_pending_detector_tracks_and_agent_audit',
  'stage244_selected':s244['selected']['variant'],
  'stage244_checkpoint':s244['selected']['checkpoint'],
  'stage244_checkpoint_sha256':s244['selected']['checkpoint_sha256'],
  'stage244_state_sha256':sha(stage244),
  'source_root':str(source_root),
  'report':str(report_path),
  'report_sha256':sha(report_path),
  'selected_frames':report['selected_frames'],
  'recording_groups':report['recording_groups'],
  'official_split':'validation',
  'used_for_training_or_threshold_selection':False,
  'test_accessed':False,
  'frozen_video_used':False,
  'production_model_modified':False,
  'deployment_performed':False,
}
target.write_text(json.dumps(result, indent=2)+'\n')
Path(str(target)+'.sha256').write_text(f'{sha(target)}  {target.name}\n')
print(json.dumps(result))
PY
