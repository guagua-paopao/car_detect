#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python
RUNNER=$BASE/code/training/scripts/recover_stage71_validation_chain_v3.py

[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "3721ab2c3e64cb2059e063bac14cc169658deb7cbd4ef1246386fe2428a4d97d" ]]
exec "$PY" "$RUNNER" --timeout-hours 24
