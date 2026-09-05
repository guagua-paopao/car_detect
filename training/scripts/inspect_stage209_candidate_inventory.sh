#!/usr/bin/env bash
set -euo pipefail

echo '=== DISK ==='
df -h /root/autodl-tmp

echo '=== GPU ==='
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader

echo '=== CANDIDATE MANIFESTS ==='
find /root/autodl-tmp/vcas/datasets/attribute-domain-v2 -maxdepth 4 -type f \
  \( -iname '*manifest*.csv' -o -iname '*accepted*.csv' -o -iname '*eligible*.csv' \) \
  | sort | tail -n 180

echo '=== RELEVANT RUN OUTPUTS ==='
find /root/autodl-tmp/vcas/runs/attributes -maxdepth 3 -type f \
  \( -path '*STAGE167*/*manifest*.csv' -o -path '*STAGE191*/*.csv' -o -path '*STAGE192*/*.csv' \
     -o -path '*STAGE196*/*.csv' -o -path '*STAGE203*/*.csv' -o -path '*STAGE206*/*.csv' \) \
  | sort | tail -n 180
