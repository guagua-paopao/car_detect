#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/bin/python
ARCHIVE=/root/autodl-tmp/vcas/sources/lvad-stage92/lvad-v2.zip
EVIDENCE=/root/autodl-tmp/vcas/sources/lvad-stage92/lvad-source-evidence-v2.json
AUDITOR=/root/autodl-tmp/vcas/code/training_stage92/scripts/audit_stage92_lvad_archive.py
TEST=/root/autodl-tmp/vcas/code/training_stage92/tests/test_audit_stage92_lvad_archive.py
REPORT=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE92-LVAD-AUDIT-V1/report.json

test ! -e "$REPORT"
echo "e9219543e93af7f2771b2743441af0b1a9bd2ccf6de4165cddc22a506d4d0e67  $ARCHIVE" | sha256sum -c -
echo "48aa7de0a60af503a67dbd0b88754b20c64ddc98c1b926b20cc1657b61e2933b  $EVIDENCE" | sha256sum -c -
echo "53a6d0a92aee2af9eda6500545cac774af476d80f3583e8b38e342186e5af4b8  $AUDITOR" | sha256sum -c -
echo "87d2a552aa17205ef5079ece1adaa7e82855cc337c04037b1cbee142467d1f0d  $TEST" | sha256sum -c -

"$PYTHON" -m py_compile "$AUDITOR"
"$PYTHON" -m unittest "$TEST"
"$PYTHON" "$AUDITOR" \
  --archive "$ARCHIVE" \
  --expected-sha256 e9219543e93af7f2771b2743441af0b1a9bd2ccf6de4165cddc22a506d4d0e67 \
  --expected-size 292339160 \
  --source-evidence "$EVIDENCE" \
  --output-report "$REPORT"
sha256sum "$REPORT" > "$REPORT.sha256"
