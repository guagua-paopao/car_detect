#!/usr/bin/env bash
set -euo pipefail

audit_session="VCAS-AUDIT-STAGE70-UA-COLOR-V2"
base="/root/autodl-tmp/vcas"
audit_root="${base}/runs/attributes/ATTR-STAGE70-UA-COLOR-PSEUDO-V2"
pool="${base}/datasets/attribute-domain-v2/stage70-uadetrac-training-pool-v2"
output="${base}/datasets/attribute-domain-v2/stage70-specialist-manifests-v2"
builder="${base}/code/training/scripts/build_stage70_specialist_manifests.py"

while tmux has-session -t "=${audit_session}" 2>/dev/null; do
  sleep 30
done

if [[ -e "${output}" ]]; then
  echo "fail_closed: refusing to overwrite ${output}" >&2
  exit 21
fi

/root/miniconda3/bin/python - "${audit_root}/color-pseudolabel-report.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file():
    raise SystemExit("fail_closed: Stage70 color audit report is missing")
x=json.loads(p.read_text(encoding="utf-8"))
if x.get("status") != "pass":
    raise SystemExit("fail_closed: Stage70 color audit did not pass")
if x.get("accepted_rows", 0) < x.get("parameters", {}).get("minimum_total", 500):
    raise SystemExit("fail_closed: accepted color rows are below the fixed minimum")
if len(x.get("accepted_color_counts", {})) < x.get("parameters", {}).get("minimum_classes", 5):
    raise SystemExit("fail_closed: accepted color classes are below the fixed minimum")
policy=x.get("policy", {})
required={
    "model_predictions_replace_proposals": False,
    "train_split_only": True,
    "validation_or_test_used": False,
    "frozen_video_used": False,
    "production_model_modified": False,
    "deployment_performed": False,
}
for key,value in required.items():
    if policy.get(key) is not value:
        raise SystemExit(f"fail_closed: color audit policy mismatch for {key}")
m=Path(x["output_manifest"])
if hashlib.sha256(m.read_bytes()).hexdigest() != x["output_manifest_sha256"]:
    raise SystemExit("fail_closed: color pseudo manifest SHA256 mismatch")
PY

[[ "$(sha256sum "${builder}" | awk '{print $1}')" == "d923bbee8099ceb6cbfa9e4f74a5b7b5107cc60b110ef003e4971ba718b261e2" ]]

exec /root/miniconda3/bin/python "${builder}" \
  --stage67-manifest "${base}/datasets/attribute-domain-v2/attribute_manifest.stage67-uvh26-controlled-clean-v2.csv" \
  --stage67-root "${base}/datasets/attribute-domain-v2" \
  --ua-manifest "${pool}/attribute_manifest.training-only.csv" \
  --ua-root "${pool}" \
  --color-pseudo-manifest "${audit_root}/attribute_manifest.color-pseudo.train-only.csv" \
  --stage69-unlabeled-manifest "${base}/datasets/attribute-domain-v2/attribute_manifest.stage69-real-domain-unlabeled-v1.csv" \
  --stage69-unlabeled-root "${base}/datasets/attribute-domain-v2" \
  --datasets-safety-root "${base}/datasets" \
  --output-root "${output}" \
  --near-duplicate-hamming 3
