#!/usr/bin/env bash
set -euo pipefail

session_name="VCAS-STAGE71-RECOVERED-CHAIN"
python_bin="/root/miniconda3/bin/python"
chain_script="/root/autodl-tmp/vcas/code/training/scripts/recover_stage71_validation_chain.py"
chain_script_sha="62d8c681a394c6b488e9bff2bf3919311844798a011b2ba580e8f45ecb278e20"
chain_output="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE71-RECOVERED-CHAIN-V1"
chain_state="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE71-RECOVERED-CHAIN-V1.state.json"
archive_root="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE71-STALE-WAITERS-ARCHIVE-20260829T1537Z"
launcher_log="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE71-RECOVERED-CHAIN-V1.launcher.log"

[[ -f "$python_bin" ]] || { echo "missing Python runtime" >&2; exit 2; }
[[ -f "$chain_script" ]] || { echo "missing Stage71 chain script" >&2; exit 2; }
[[ "$(sha256sum "$chain_script" | awk '{print $1}')" == "$chain_script_sha" ]] || {
  echo "Stage71 chain script SHA256 mismatch" >&2
  exit 3
}
for absent_path in "$chain_output" "$chain_state" "$archive_root"; do
  [[ ! -e "$absent_path" ]] || { echo "refusing to overwrite $absent_path" >&2; exit 4; }
done
if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "session already exists: $session_name" >&2
  exit 5
fi
if pgrep -f '^/root/miniconda3/bin/python .*/recover_stage71_validation_chain.py' >/dev/null; then
  echo "Stage71 recovered chain process is already active" >&2
  exit 6
fi

tmux new-session -d -s "$session_name" \
  "$python_bin '$chain_script' > '$launcher_log' 2>&1"

echo "started tmux:$session_name"
echo "log=$launcher_log"
