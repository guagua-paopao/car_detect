#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
PY=/root/miniconda3/bin/python3
CODE="$BASE/code/stage242-nightowls-color-repair-r1"
SCRIPT="$CODE/build_stage242_nightowls_color_repair_manifest.py"
BASE_MANIFEST="$BASE/datasets/attribute-domain-v2/stage157-fresh-color-combined-r2/attribute_manifest.stage157-fresh-color-combined.csv"
STAGE241_MANIFEST="$CODE/evidence/stage241-nightowls-color-supervised-research-only.csv"
STAGE241_REPORT="$CODE/evidence/stage241-nightowls-color-agent-audit.json"
OUTPUT="$BASE/datasets/attribute-domain-v2/stage242-nightowls-color-repair-r1"

assert_sha256() {
  local path="$1" expected="$2" actual
  test -f "$path"
  actual="$(sha256sum "$path" | awk '{print tolower($1)}')"
  test "$actual" = "${expected,,}"
}

assert_sha256 "$SCRIPT" ab49f8c2d8b598e5abb3cf254181cf13c3c56d92292a1abcb4870bbca4daca15
assert_sha256 "$BASE_MANIFEST" db92921c1633183b018be70091dfe9f26cc55fa942b844c4749b2add5e430f23
assert_sha256 "$STAGE241_MANIFEST" 132b96feb420c2d5e7e8158e98c94d139311975db7605f9566bcad6da5d21bee
assert_sha256 "$STAGE241_REPORT" 81cb70f40d4cb2228b403578e82b4e55c4bca6763eca67c85b8be1b84dc8f673
"$PY" -m py_compile "$SCRIPT"
test ! -e "$OUTPUT"

"$PY" "$SCRIPT" \
  --base-manifest "$BASE_MANIFEST" \
  --expected-base-sha256 db92921c1633183b018be70091dfe9f26cc55fa942b844c4749b2add5e430f23 \
  --stage241-manifest "$STAGE241_MANIFEST" \
  --expected-stage241-sha256 132b96feb420c2d5e7e8158e98c94d139311975db7605f9566bcad6da5d21bee \
  --stage241-report "$STAGE241_REPORT" \
  --expected-stage241-report-sha256 81cb70f40d4cb2228b403578e82b4e55c4bca6763eca67c85b8be1b84dc8f673 \
  --output-dir "$OUTPUT"
