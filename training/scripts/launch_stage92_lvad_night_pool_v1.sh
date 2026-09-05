#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/vcas/code/training_stage92_pool_v1
BUILDER=$ROOT/scripts/build_stage92_lvad_night_pool.py
TEST=$ROOT/tests/test_build_stage92_lvad_night_pool.py
ARCHIVE=/root/autodl-tmp/vcas/sources/lvad-stage92/lvad-v2.zip
AUDIT=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE92-LVAD-AUDIT-R2/report.json
EVIDENCE=/root/autodl-tmp/vcas/sources/lvad-stage92/lvad-source-evidence-v2-r2.json
BASE83=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage83-body-v2-merged-v5/attribute_manifest.stage83-body-v2-merged.csv
BASE89=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage89-mio-balanced-expansion-r4/attribute_manifest.stage89-mio-balanced.csv
BASE91=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage91-inatrc-train-crops-v1/attribute_manifest.stage91-all.csv
OUTPUT=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage92-lvad-night-pool-v1

test ! -e "$OUTPUT"
echo "456f5c26ae52d7463bdfb732046eda079e287d4505e721d507d2946875a536a8  $BUILDER" | sha256sum -c -
echo "52736d65af1d9b6202f0d143e2df38477a7bb75a9939aef29f5e3bc93a8cdb41  $TEST" | sha256sum -c -
echo "e9219543e93af7f2771b2743441af0b1a9bd2ccf6de4165cddc22a506d4d0e67  $ARCHIVE" | sha256sum -c -
echo "f61fa3582f47cdcda3bbceef1f3ea08daef30ac2cbbe170464f8c8a855da4a68  $AUDIT" | sha256sum -c -
echo "ab98375d6f0449dda85a8d95922e20f30b5391172aa3e17ef0910b57a9d7c513  $EVIDENCE" | sha256sum -c -
echo "f0d95b424926a38b6e162fd0fa78da1329e7eab180e2a2dd8df42f26e06fcdef  $BASE83" | sha256sum -c -
echo "acb0e4d42aeb79d429dee4e864d41939a84f7b1d72298c1c830a589be90bddf7  $BASE89" | sha256sum -c -
echo "d9cf4d5c8469a5da47d5b0e7b5df78f628b5084738f0de62037993bee3e6acf1  $BASE91" | sha256sum -c -

"$PYTHON" -m py_compile "$BUILDER"
"$PYTHON" -m unittest "$TEST"
"$PYTHON" "$BUILDER" \
  --archive "$ARCHIVE" \
  --expected-archive-sha256 e9219543e93af7f2771b2743441af0b1a9bd2ccf6de4165cddc22a506d4d0e67 \
  --audit-report "$AUDIT" \
  --expected-audit-report-sha256 f61fa3582f47cdcda3bbceef1f3ea08daef30ac2cbbe170464f8c8a855da4a68 \
  --source-evidence "$EVIDENCE" \
  --expected-source-evidence-sha256 ab98375d6f0449dda85a8d95922e20f30b5391172aa3e17ef0910b57a9d7c513 \
  --base-manifest "$BASE83" \
  --expected-base-manifest-sha256 f0d95b424926a38b6e162fd0fa78da1329e7eab180e2a2dd8df42f26e06fcdef \
  --base-manifest "$BASE89" \
  --expected-base-manifest-sha256 acb0e4d42aeb79d429dee4e864d41939a84f7b1d72298c1c830a589be90bddf7 \
  --base-manifest "$BASE91" \
  --expected-base-manifest-sha256 d9cf4d5c8469a5da47d5b0e7b5df78f628b5084738f0de62037993bee3e6acf1 \
  --output-images "$OUTPUT/images" \
  --output-manifest "$OUTPUT/attribute_manifest.stage92-lvad-night.csv" \
  --output-report "$OUTPUT/stage92-lvad-night-pool-report.json" \
  --minimum-temporal-gap 15 \
  --track-window-frames 30 \
  --frame-near-duplicate-hamming 4 \
  --crop-near-duplicate-hamming 4 \
  --minimum-rows 1000 \
  --minimum-car-rows 800 \
  --minimum-truck-rows 100
sha256sum "$OUTPUT/attribute_manifest.stage92-lvad-night.csv" "$OUTPUT/stage92-lvad-night-pool-report.json" > "$OUTPUT/SHA256SUMS"
