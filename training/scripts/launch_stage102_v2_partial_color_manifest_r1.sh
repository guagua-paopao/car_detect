#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
CODE="$BASE/code/training_stage102_partial_color_r1"
BUILDER="$CODE/scripts/build_stage102_v2_partial_color_manifest.py"
BASE_MANIFEST="$BASE/datasets/attribute-domain-v2/stage78-dvm-fine-color-runtime-v2/attribute_manifest.stage78-dvm-fine-color-runtime.csv"
STAGE74_MANIFEST="$BASE/datasets/attribute-domain-v2/stage71-teacher-manifests-v5/attribute_manifest.stage74-joint-color-runtime-v4.csv"
OUTPUT="$BASE/datasets/attribute-domain-v2/stage102-v2-partial-color-r1"
MANIFEST="$OUTPUT/attribute_manifest.stage102-v2-partial-color.csv"
REPORT="$OUTPUT/stage102-v2-partial-color-report.json"
STATE="$BASE/runs/attributes/ATTR-STAGE102-V2-PARTIAL-COLOR-MANIFEST-R1.state.json"
LOG="$BASE/runs/attributes/ATTR-STAGE102-V2-PARTIAL-COLOR-MANIFEST-R1.log"
SESSION=VCAS-STAGE102-COLOR-MANIFEST-R1
SELF="$CODE/scripts/launch_stage102_v2_partial_color_manifest_r1.sh"

run_worker() {
  exec >"$LOG" 2>&1
  "$PY" -c 'import json,sys; from datetime import datetime,timezone; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"status":"running_image_and_leakage_audit","started_at":datetime.now(timezone.utc).isoformat(),"test_accessed":False,"frozen_video_used":False,"production_model_modified":False,"deployment_performed":False},indent=2)+"\n",encoding="utf-8")' "$STATE"
  cd "$CODE"
  "$PY" "$BUILDER" \
    --base-v2-manifest "$BASE_MANIFEST" \
    --stage74-manifest "$STAGE74_MANIFEST" \
    --safety-root "$BASE" \
    --output-dir "$OUTPUT" \
    --expected-base-sha256 dc06fe38daa5aaea1daf0c38133265e9d7f0650b5aed32f613462a1d94f1edb5 \
    --expected-stage74-sha256 698db00b15d602a79e9c5fc318811bb8908109f148d85890167f58c4146cd5a9 \
    --near-duplicate-hamming 2
  "$PY" -c 'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8")); assert r["status"]=="pass"; assert r["rows"]["added_cctv_train"]>=1500; assert r["rows"]["base_validation"]>0; assert len(r["added_supervision_counts"])>=5; assert r["leakage"]["exact_train_validation"]==0; assert r["leakage"]["group_train_validation"]==0; assert r["leakage"]["near_train_validation"]==0; assert r["policy"]["test_accessed"] is False; assert r["policy"]["frozen_video_used"] is False; assert r["policy"]["validation_rows_added_to_training"]==0; assert r["policy"]["merged_silver_gray_exact_class_fabricated"] is False; assert r["policy"]["production_model_modified"] is False' "$REPORT"
  sha256sum "$MANIFEST" "$REPORT" > "$OUTPUT/artifacts.sha256"
  "$PY" -c 'import hashlib,json,sys; from datetime import datetime,timezone; from pathlib import Path; report=json.load(open(sys.argv[2],encoding="utf-8")); digest=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); state={"status":"pass_training_manifest_ready","completed_at":datetime.now(timezone.utc).isoformat(),"manifest":sys.argv[3],"manifest_sha256":digest(sys.argv[3]),"report":sys.argv[2],"report_sha256":digest(sys.argv[2]),"rows":report["rows"],"added_supervision_counts":report["added_supervision_counts"],"added_source_counts":report["added_source_counts"],"added_license_counts":report["added_license_counts"],"leakage":report["leakage"],"test_accessed":False,"frozen_video_used":False,"production_model_modified":False,"deployment_performed":False}; Path(sys.argv[1]).write_text(json.dumps(state,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")' "$STATE" "$REPORT" "$MANIFEST"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker
  exit 0
fi

for absent in "$OUTPUT" "$STATE" "$LOG"; do
  [[ ! -e "$absent" ]] || { echo "refusing to overwrite Stage102 evidence: $absent" >&2; exit 66; }
done
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION" >&2
  exit 67
fi
echo "22ecd4b1eb0f0832f9b962bd3aa56e06043f9d1658ddcd21a2dcf3e197db5fab  $BUILDER" | sha256sum -c -
echo "dc06fe38daa5aaea1daf0c38133265e9d7f0650b5aed32f613462a1d94f1edb5  $BASE_MANIFEST" | sha256sum -c -
echo "698db00b15d602a79e9c5fc318811bb8908109f148d85890167f58c4146cd5a9  $STAGE74_MANIFEST" | sha256sum -c -
tmux new-session -d -s "$SESSION" "bash '$SELF' --worker"

echo "started tmux:$SESSION"
echo "state=$STATE"
echo "output=$OUTPUT"
echo "log=$LOG"
