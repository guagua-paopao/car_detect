#!/usr/bin/env bash
set -euo pipefail

base="/root/autodl-tmp/vcas"
audit_root="${base}/runs/attributes/ATTR-STAGE70-UA-COLOR-PSEUDO-V2"
pool="${base}/datasets/attribute-domain-v2/stage70-uadetrac-training-pool-v2"
output="${base}/datasets/attribute-domain-v2/stage70-specialist-manifests-v4-no-color-pseudo"
builder="${base}/code/training/scripts/build_stage70_specialist_manifests.py"

if [[ -e "${output}" ]]; then
  echo "fail_closed: refusing to overwrite ${output}" >&2
  exit 21
fi

/root/miniconda3/bin/python - "${audit_root}/color-pseudolabel-report.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file():
    raise SystemExit("fail_closed: Stage70 color audit report is missing")
x=json.loads(p.read_text(encoding="utf-8"))
if x.get("status") != "fail":
    raise SystemExit("fail_closed: no-pseudo path is allowed only after a preserved failed audit")
if x.get("accepted_rows", 0) >= x.get("parameters", {}).get("minimum_total", 500):
    raise SystemExit("fail_closed: failed audit reason is inconsistent with the fixed total gate")
policy=x.get("policy", {})
for key,value in {
    "train_split_only": True, "validation_or_test_used": False,
    "frozen_video_used": False, "production_model_modified": False,
    "deployment_performed": False,
}.items():
    if policy.get(key) is not value:
        raise SystemExit(f"fail_closed: color audit policy mismatch for {key}")
PY

exec /root/miniconda3/bin/python "${builder}" \
  --stage67-manifest "${base}/datasets/attribute-domain-v2/attribute_manifest.stage67-uvh26-controlled-clean-v2.csv" \
  --stage67-root "${base}/datasets/attribute-domain-v2" \
  --ua-manifest "${pool}/attribute_manifest.training-only.csv" \
  --ua-root "${pool}" \
  --omit-color-pseudo \
  --stage69-unlabeled-manifest "${base}/datasets/attribute-domain-v2/attribute_manifest.stage69-real-domain-unlabeled-v1.csv" \
  --stage69-unlabeled-root "${base}/datasets/attribute-domain-v2" \
  --datasets-safety-root "${base}/datasets" \
  --output-root "${output}" \
  --near-duplicate-hamming 3
