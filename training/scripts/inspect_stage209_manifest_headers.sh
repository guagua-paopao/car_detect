#!/usr/bin/env bash
set -euo pipefail

files=(
  /root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage167-complex-body-r1/attribute_manifest.stage167-complex-body.csv
  /root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE192-TESLA-TEACHER-QUOTA-R2/stage192-tesla-reviewed-manifest.csv
  /root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE196-ABTD-BODY-TEACHER-R2/body-teacher-reviewed.csv
  /root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE203-JUNCTION-ORIGINAL-AUDIT-R1/stage203-original-image-audit.csv
  /root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage156-fullscale-body-repair-r1/attribute_manifest.stage156-fullscale-body-repair.csv
  /root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage153-lvad-body-repair-r3/attribute_manifest.stage153-lvad-body-repair.csv
)

for file in "${files[@]}"; do
  echo "=== ${file} ==="
  wc -l "${file}"
  sed -n '1,3p' "${file}"
done

echo '=== STAGE167 MODEL CARD FILTERED ==='
python3 - <<'PY'
import json
from pathlib import Path
p = Path('/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE167-COMPLEX-BODY-R1/ATTR-STAGE167-BODY-CONVNEXT-256-STRETCH-R1/model_card.json')
d = json.loads(p.read_text())
for key in sorted(d):
    if any(token in key.lower() for token in ('manifest', 'unlabel', 'night', 'complex', 'sample', 'weight', 'row')):
        print(key, json.dumps(d[key], ensure_ascii=False)[:3000])
PY
