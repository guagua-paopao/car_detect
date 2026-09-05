#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
TRAIN=$BASE/code/training_stage66/scripts/train_attribute.py
LABELS=$BASE/code/config/vehicle_labels.v1.json
DVM=$BASE/datasets/attribute-domain-v2/stage68-dvm-color-pool-v1/attribute_manifest.audit-clean-v2.csv
DOMAIN=$BASE/datasets/attribute-domain-v2/attribute_manifest.stage66-licensed-adverse-clean-v3.csv
AUDIT=$BASE/runs/attributes/ATTR-STAGE68-DVM-COLOR-FINALIZE-V3.report.json
LICENSE=$BASE/sources/color-scale-stage68/SOURCE_LICENSE_EVIDENCE.json
BASELINE=$BASE/runs/stage3/ATTR-AGENT-E-DISTILL-MNV3-224/best.pt
PRETRAIN=$BASE/runs/attributes/ATTR-STAGE68-DVM-COLOR-PRETRAIN-V1
FINETUNE=$BASE/runs/attributes/ATTR-STAGE68-CCTV-DOMAIN-FINETUNE-V1
STATE=$BASE/runs/attributes/ATTR-STAGE68-COLOR-TRAINING-V1.state.json

check_hash() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  test "$actual" = "$expected"
}

check_hash 9c7bc1296e7c5d2caa7cabfc8d34b031e1dbdbccb9be355ce36b41d7d9f0910b "$DVM"
check_hash 3ff1dc6fc565a43127e0db55cb14797bb3a661512d4341d9a843a380e81a38ee "$AUDIT"
check_hash 74c8be98927abe9b323974a5ca0da956d241d6b26306eaa843bc54be7acd6819 "$LICENSE"
check_hash e362136d56ca707d6de738d312b8df7cf5d72cdd05d8b2efdab1b6b36db8a0ed "$DOMAIN"
check_hash f00eabfec69cb66bb9936baef17c1c7ec293d2b9bd603ed84ce7ec75b586a8f4 "$TRAIN"
check_hash c71675a0e2de950880f76f088c108c6fb286561a2399cb5a182055289fb5eb4f "$LABELS"
check_hash 6f651bba1c62082f13740728c96c82a074fbb3a8ecaf27fac1f90504c6bb7383 "$BASELINE"

test ! -e "$PRETRAIN"
test ! -e "$FINETUNE"
test ! -e "$STATE"

"$PY" - "$AUDIT" "$LICENSE" "$DVM" "$STATE" <<'PY'
import csv, json, sys
from datetime import datetime, timezone
from pathlib import Path
audit_path, license_path, manifest_path, state_path = map(Path, sys.argv[1:])
audit = json.loads(audit_path.read_text())
evidence = json.loads(license_path.read_text())
if audit.get("status") != "pass" or audit.get("output_train_rows") != 118338:
    raise SystemExit("Stage68 data gate failed")
if evidence.get("sources", [])[0].get("training_decision") != "countable_research_only_non_deployable":
    raise SystemExit("DVM research-only license gate failed")
with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
train = [row for row in rows if row.get("split") == "train"]
if len(train) < 118338 or any(row.get("color_supervised") != "true" for row in train):
    raise SystemExit("effective color row gate failed")
if any("vcas_rtsp_demo_60s" in " ".join(row.values()).lower() or "frozen_video" in " ".join(row.values()).lower() for row in rows):
    raise SystemExit("frozen marker gate failed")
state = {
    "schema_version": "attribute-stage68-research-color-training-state-v1",
    "status": "running_dvm_color_pretraining",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "training_eligibility": "research-only; non-deployable",
    "dvm_train_rows": len(train),
    "test_accessed": False,
    "frozen_video_used": False,
    "production_model_modified": False,
    "deployment_performed": False
}
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
PY

"$PY" "$TRAIN" \
  --manifest "$DVM" --labels "$LABELS" \
  --input-size 224 --architecture mobilenet_v3_large --resize-mode stretch \
  --epochs 8 --batch-size 128 --workers 8 --learning-rate 4e-5 --weight-decay 1e-4 \
  --color-loss-weight 1.0 --body-loss-weight 0.0 --color-focal-gamma 1.0 \
  --class-weighting inverse_sqrt --label-smoothing 0.02 --gradient-clip-norm 5.0 \
  --freeze-backbone-epochs 1 --patience 4 --type-threshold 0.75 --color-threshold 0.70 \
  --device cuda --augmentation-profile color_scene --selection-head color \
  --init-checkpoint "$BASELINE" --run-kind formal --skip-test \
  --dataset-version attribute-domain-v2-stage68-dvm-color-research-v1 \
  --code-revision stage68-dvm-color-pretrain-v1 --output-dir "$PRETRAIN"

test -f "$PRETRAIN/best.pt"
"$PY" - "$STATE" "$PRETRAIN/best.pt" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
state_path, checkpoint = map(Path, sys.argv[1:])
state = json.loads(state_path.read_text())
state.update({
    "status": "running_real_cctv_domain_finetuning",
    "dvm_pretraining_completed_at": datetime.now(timezone.utc).isoformat(),
    "dvm_pretrained_checkpoint": str(checkpoint),
    "dvm_pretrained_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
})
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
PY

"$PY" "$TRAIN" \
  --manifest "$DOMAIN" --labels "$LABELS" \
  --input-size 224 --architecture mobilenet_v3_large --resize-mode stretch \
  --epochs 12 --batch-size 128 --workers 8 --learning-rate 2e-5 --weight-decay 1e-4 \
  --color-loss-weight 1.5 --body-loss-weight 1.0 --focal-gamma 1.5 --color-focal-gamma 1.0 \
  --class-weighting inverse_sqrt --label-smoothing 0.03 --gradient-clip-norm 5.0 \
  --freeze-backbone-epochs 1 --patience 5 --type-threshold 0.75 --color-threshold 0.70 \
  --device cuda --augmentation-profile color_scene --selection-head joint \
  --hard-sample-weight 0.2 --small-sample-weight 1.0 --night-sample-weight 20.0 \
  --occlusion-sample-weight 1.5 --color-sample-weight 6.0 \
  --init-checkpoint "$PRETRAIN/best.pt" --teacher-checkpoint "$PRETRAIN/best.pt" \
  --distill-weight 0.12 --distill-temperature 2.5 --run-kind formal --skip-test \
  --dataset-version attribute-domain-v2-stage68-real-cctv-domain-research-v1 \
  --code-revision stage68-cctv-domain-finetune-v1 --output-dir "$FINETUNE"

test -f "$FINETUNE/best.pt"
"$PY" - "$STATE" "$FINETUNE/best.pt" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
state_path, checkpoint = map(Path, sys.argv[1:])
state = json.loads(state_path.read_text())
state.update({
    "status": "training_complete_validation_pending",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "candidate_checkpoint": str(checkpoint),
    "candidate_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "deployment_performed": False
})
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
PY
