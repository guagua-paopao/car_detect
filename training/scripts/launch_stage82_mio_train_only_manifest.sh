#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/vcas
OUT="$ROOT/datasets/attribute-domain-v2/stage82-mio-train-only-v1"

exec nice -n 15 ionice -c 3 /root/miniconda3/bin/python \
  "$ROOT/code/training/scripts/build_stage82_mio_train_only_manifest.py" \
  --archive "$ROOT/sources/mio-tcd/MIO-TCD-Classification.tar" \
  --expected-archive-sha256 0cf70c660f399aef05d069d9c993e2e3f89e887c823c781d3064fee3b2e6979c \
  --staging-root "$ROOT/datasets/dataset-large-v1-mio-attributes-r2/_selected" \
  --successor-manifest "$ROOT/datasets/attribute-domain-v2/stage72-body-taxonomy-v2-v1/attribute_manifest.stage72-body-taxonomy-v2.csv" \
  --expected-successor-sha256 0b28f8bab925c5e5c6b680af1fcb6d35a4a76a09d7ca1f47ec0f02e8c74a4aed \
  --labels "$ROOT/code/config/vehicle_labels.v2.json" \
  --expected-labels-sha256 22a701568d3e7626e4ff61c2170aff730377d2b9db5d2901b439d5b4e102de0f \
  --output-manifest "$OUT/attribute_manifest.stage82-mio-train-only.csv" \
  --output-report "$OUT/stage82-mio-train-only-report.json" \
  --near-duplicate-hamming 4 \
  --minimum-rows 25000 \
  --minimum-per-class 3000
