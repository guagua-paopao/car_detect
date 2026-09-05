#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
RUNNER=$BASE/code/training/scripts/recover_stage71_validation_chain_v2.py

[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "0ff1f89982f08a33cfafff66495100b63282d57c7bdc4db430ce2196bdb0aa22" ]]
exec "$PY" "$RUNNER" --timeout-hours 24
