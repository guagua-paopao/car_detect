#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/bin/python
ARCHIVE=/root/autodl-tmp/vcas/sources/lvad-stage92/lvad-v2.zip
EVIDENCE=/root/autodl-tmp/vcas/sources/lvad-stage92/lvad-source-evidence-v2-r2.json
AUDITOR=/root/autodl-tmp/vcas/code/training_stage92_r2/scripts/audit_stage92_lvad_archive.py
TEST=/root/autodl-tmp/vcas/code/training_stage92_r2/tests/test_audit_stage92_lvad_archive.py
REPORT=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE92-LVAD-AUDIT-R2/report.json

test ! -e "$REPORT"
echo "e9219543e93af7f2771b2743441af0b1a9bd2ccf6de4165cddc22a506d4d0e67  $ARCHIVE" | sha256sum -c -
echo "ab98375d6f0449dda85a8d95922e20f30b5391172aa3e17ef0910b57a9d7c513  $EVIDENCE" | sha256sum -c -
echo "eeb36537b227900b5fee1b7bf93a72c5635741f509da22c40993692a1b5bb86d  $AUDITOR" | sha256sum -c -
echo "1eeb4a36c7c4957a2dfd1145bdd66d6837c8caf0dfb83451884ad079cc47bcb5  $TEST" | sha256sum -c -

"$PYTHON" -m py_compile "$AUDITOR"
"$PYTHON" -m unittest "$TEST"
"$PYTHON" "$AUDITOR" \
  --archive "$ARCHIVE" \
  --expected-sha256 e9219543e93af7f2771b2743441af0b1a9bd2ccf6de4165cddc22a506d4d0e67 \
  --expected-size 292339160 \
  --source-evidence "$EVIDENCE" \
  --output-report "$REPORT"
sha256sum "$REPORT" > "$REPORT.sha256"
