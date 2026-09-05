#!/usr/bin/env bash
set -euo pipefail
find /root/autodl-tmp/vcas/sources -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
