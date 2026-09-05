#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
CODE="$BASE/code/training"
SOURCE_ROOT="$BASE/sources/vehicle-rear-stage149"
SELECTED="$SOURCE_ROOT/selected-r3"
SELECTION_MANIFEST="$SOURCE_ROOT/stage149-selection-plan-r2.csv"
R3_STATE="$BASE/runs/attributes/ATTR-STAGE149-VEHICLE-REAR-PARALLEL-EXTRACT-R3.state.json"
OUT="$BASE/datasets/attribute-domain-v2/stage149-vehicle-rear-audited-r1"
MANIFEST="$OUT/attribute_manifest.stage149-vehicle-rear.csv"
REPORT="$OUT/stage149-vehicle-rear-audit-report.json"
STATE="$BASE/runs/attributes/ATTR-STAGE149-VEHICLE-REAR-AUDIT-R1.state.json"
LOG="$BASE/runs/attributes/ATTR-STAGE149-VEHICLE-REAR-AUDIT-R1.log"
SESSION=VCAS-STAGE149-VEHICLEREAR-AUDIT-R1
TRANSFER_SESSION=VCAS-DL-VEHICLEREAR149-R3
AUDITOR="$CODE/scripts/audit_stage149_vehicle_rear_selected.py"
README="$SOURCE_ROOT/OFFICIAL_README.md"
LICENSE="$SOURCE_ROOT/OFFICIAL_LICENSE"
README_SHA=f02ddc132d210f3378b30048daa8840cd62453b12a83d9b402e54bab6909cc1f
LICENSE_SHA=c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4

if [[ "${1:-}" != "--worker" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "$SESSION already running"
    exit 0
  fi
  tmux new-session -d -s "$SESSION" "bash '$0' --worker >>'$LOG' 2>&1"
  echo "started tmux:$SESSION"
  exit 0
fi

for _ in $(seq 1 240); do
  if ! tmux has-session -t "$TRANSFER_SESSION" 2>/dev/null; then
    break
  fi
  sleep 30
done
if tmux has-session -t "$TRANSFER_SESSION" 2>/dev/null; then
  echo "Stage149 R3 did not finish within two hours" >&2
  exit 2
fi
for required in "$R3_STATE" "$SELECTED" "$SELECTION_MANIFEST" "$AUDITOR" "$README" "$LICENSE"; do
  [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 3; }
done
/root/miniconda3/bin/python - "$R3_STATE" <<'PY'
import json,sys
state=json.load(open(sys.argv[1],encoding='utf-8'))
if state.get('status')!='selective_extraction_complete_pending_license_dedup_and_label_audit':
    raise SystemExit(f"R3 is not audit-ready: {state.get('status')}")
if state.get('policy',{}).get('future_holdout_images_extracted') is not False:
    raise SystemExit('sealed holdout policy failed')
PY

[[ ! -e "$OUT" ]] || { echo "refusing to overwrite audit output: $OUT" >&2; exit 4; }
mkdir -p "$OUT"
/root/miniconda3/bin/python "$AUDITOR" \
  --selection-manifest "$SELECTION_MANIFEST" \
  --selected-root "$SELECTED" \
  --official-readme "$README" \
  --official-license "$LICENSE" \
  --expected-readme-sha256 "$README_SHA" \
  --expected-license-sha256 "$LICENSE_SHA" \
  --output-manifest "$MANIFEST" \
  --output-report "$REPORT" \
  --workers 12 \
  --minimum-train-rows 4000

sha256sum "$MANIFEST" "$REPORT" > "$OUT/SHA256SUMS"
/root/miniconda3/bin/python - "$REPORT" "$STATE" <<'PY'
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,sys
report_path,state_path=map(Path,sys.argv[1:])
report=json.loads(report_path.read_text(encoding='utf-8'))
payload={
  'schema_version':'stage149-vehicle-rear-audit-state-v1',
  'created_at':datetime.now(timezone.utc).isoformat(),
  'status':'pass_research_only_pending_stage150' if report.get('status')=='pass_research_only' else 'fail_closed',
  'report':str(report_path),
  'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
  'manifest':report.get('output',{}).get('manifest'),
  'manifest_sha256':report.get('output',{}).get('manifest_sha256'),
  'train_rows':report.get('output',{}).get('train_rows'),
  'validation_rows':report.get('output',{}).get('validation_rows'),
  'policy':report.get('policy',{}),
}
state_path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
state_path.with_suffix(state_path.suffix+'.sha256').write_text(
  f"{hashlib.sha256(state_path.read_bytes()).hexdigest()}  {state_path.name}\n",encoding='utf-8')
print(json.dumps(payload,ensure_ascii=False))
PY

echo "Stage149 Vehicle-Rear audit complete"
