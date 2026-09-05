#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
BUILDER="$BASE/code/training/scripts/build_stage150_vehicle_rear_color_manifest.py"
BASE_MANIFEST="$BASE/datasets/attribute-domain-v2/stage102-v2-partial-color-r1/attribute_manifest.stage102-v2-partial-color.csv"
AUDIT_DIR="$BASE/datasets/attribute-domain-v2/stage149-vehicle-rear-audited-r1"
SUPPLEMENT_MANIFEST="$AUDIT_DIR/attribute_manifest.stage149-vehicle-rear.csv"
SUPPLEMENT_REPORT="$AUDIT_DIR/stage149-vehicle-rear-audit-report.json"
AUDIT_STATE="$BASE/runs/attributes/ATTR-STAGE149-VEHICLE-REAR-AUDIT-R1.state.json"
OUT="$BASE/datasets/attribute-domain-v2/stage150-vehicle-rear-color-domain-r2"
MANIFEST="$OUT/attribute_manifest.stage150-vehicle-rear-color-domain.csv"
REPORT="$OUT/stage150-vehicle-rear-color-domain-manifest-report.json"
STATE="$BASE/runs/attributes/ATTR-STAGE150-VEHICLE-REAR-COLOR-MANIFEST-R2.state.json"
LOG="$BASE/runs/attributes/ATTR-STAGE150-VEHICLE-REAR-COLOR-MANIFEST-R2.log"
SESSION=VCAS-STAGE150-COLOR-MANIFEST-R2
BASE_SHA=aac74a71cdbfb9ace020654973ecc443537b67ef30d6135f2fe321caf3d1d21c
BUILDER_SHA=a91327b19c447bb45ce984a59a2a7ae6fa8c0e457357296d6ab06a72df0a27a2

if [[ "${1:-}" != "--worker" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "$SESSION already running"
    exit 0
  fi
  tmux new-session -d -s "$SESSION" "bash '$0' --worker >>'$LOG' 2>&1"
  echo "started tmux:$SESSION"
  exit 0
fi

for required in "$PY" "$BUILDER" "$BASE_MANIFEST" "$SUPPLEMENT_MANIFEST" "$SUPPLEMENT_REPORT" "$AUDIT_STATE"; do
  [[ -f "$required" ]] || { echo "missing required input: $required" >&2; exit 3; }
done
[[ "$(sha256sum "$BUILDER" | awk '{print tolower($1)}')" == "$BUILDER_SHA" ]] || {
  echo "Stage150 builder SHA256 mismatch" >&2
  exit 4
}
[[ "$(sha256sum "$BASE_MANIFEST" | awk '{print tolower($1)}')" == "$BASE_SHA" ]] || {
  echo "Stage102 base manifest SHA256 mismatch" >&2
  exit 5
}

"$PY" - "$AUDIT_STATE" "$SUPPLEMENT_MANIFEST" "$SUPPLEMENT_REPORT" <<'PY'
from pathlib import Path
import hashlib,json,sys
state_path,manifest_path,report_path=map(Path,sys.argv[1:])
state=json.loads(state_path.read_text(encoding='utf-8'))
if state.get('status')!='pass_research_only_pending_stage150':
    raise SystemExit(f"Stage149 audit is not admissible: {state.get('status')}")
manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
report_sha=hashlib.sha256(report_path.read_bytes()).hexdigest()
if manifest_sha!=state.get('manifest_sha256') or report_sha!=state.get('report_sha256'):
    raise SystemExit('Stage149 audit evidence SHA256 mismatch')
policy=state.get('policy',{})
if policy.get('future_holdout_images_opened') is not False:
    raise SystemExit('Stage149 future holdout seal failed')
if policy.get('stage148_test_reused') is not False or policy.get('frozen_video_used') is not False:
    raise SystemExit('forbidden evaluation data reuse detected')
PY

[[ ! -e "$OUT" ]] || { echo "refusing to overwrite Stage150 R2 output: $OUT" >&2; exit 6; }
[[ ! -e "$STATE" ]] || { echo "refusing to overwrite Stage150 R2 state: $STATE" >&2; exit 7; }
mkdir -p "$OUT"
"$PY" "$BUILDER" \
  --base-manifest "$BASE_MANIFEST" \
  --expected-base-sha256 "$BASE_SHA" \
  --supplement-manifest "$SUPPLEMENT_MANIFEST" \
  --supplement-report "$SUPPLEMENT_REPORT" \
  --output-manifest "$MANIFEST" \
  --output-report "$REPORT" \
  --minimum-supplement-train 3500 \
  --minimum-total-train 118338 \
  --minimum-validation 500 \
  --workers 12

sha256sum "$MANIFEST" "$REPORT" > "$OUT/SHA256SUMS"
"$PY" - "$REPORT" "$STATE" <<'PY'
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,sys
report_path,state_path=map(Path,sys.argv[1:])
report=json.loads(report_path.read_text(encoding='utf-8'))
integrity=report.get('integrity',{})
policy=report.get('policy',{})
passed=(
  report.get('status')=='pass_research_only'
  and integrity.get('train_validation_group_overlap')==0
  and integrity.get('train_validation_exact_sha_overlap')==0
  and integrity.get('train_validation_dhash_distance_le_4_rows')==0
  and policy.get('canonical_dhash_recomputed_from_saved_pixels') is True
  and policy.get('legacy_stage102_validation_imported') is False
  and policy.get('stage148_test_reused') is False
  and policy.get('frozen_video_used') is False
)
status='pass_research_only_pending_training' if passed else 'fail_closed'
payload={
  'schema_version':'stage150-vehicle-rear-color-manifest-state-v2',
  'created_at':datetime.now(timezone.utc).isoformat(),
  'status':status,
  'report':str(report_path),
  'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
  'manifest':report.get('output',{}).get('manifest'),
  'manifest_sha256':report.get('output',{}).get('manifest_sha256'),
  'rows':report.get('output',{}).get('rows'),
  'train_rows':report.get('output',{}).get('train_rows'),
  'validation_rows':report.get('output',{}).get('validation_rows'),
  'accepted_supplement_train':report.get('output',{}).get('accepted_supplement_train'),
  'legacy_base_validation_rows_quarantined':report.get('inputs',{}).get('legacy_base_validation_rows_quarantined'),
  'legacy_base_train_rows_near_legacy_validation':integrity.get('legacy_base_train_rows_near_legacy_validation'),
  'policy':policy,
}
state_path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
state_path.with_suffix(state_path.suffix+'.sha256').write_text(
  f"{hashlib.sha256(state_path.read_bytes()).hexdigest()}  {state_path.name}\n",encoding='utf-8')
print(json.dumps(payload,ensure_ascii=False))
if not passed:
    raise SystemExit(8)
PY

echo "Stage150 R2 research-only color manifest complete"
