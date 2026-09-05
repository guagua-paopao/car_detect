#!/usr/bin/env bash
set -euo pipefail

session_name="VCAS-RECOVER-STAGE71-TEACHERS"
python_bin="/root/miniconda3/bin/python"
recovery_script="/root/autodl-tmp/vcas/code/training/scripts/recover_stage71_teacher_matrix.py"
matrix_path="/root/autodl-tmp/vcas/code/training/stage71-convnext-teacher-matrix-v1.json"
output_root="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE71-CONVNEXT-TEACHERS-V1"
state_path="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE71-CONVNEXT-TEACHERS-V1.state.json"
original_runner="/root/autodl-tmp/vcas/code/training/scripts/run_stage71_teacher_matrix.py"
recovery_log="/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE71-RECOVERY-V1.log"
expected_state_sha="a11ae392a61ce31820d9cfb9b7069556d7d64c106ea4dfc033a184138d7d9f8f"
expected_last_sha="32990bf78003a54a23077c7ac1b9d81839ea331d5ffac97f6daee10fe821027b"

for required_file in \
  "$python_bin" \
  "$recovery_script" \
  "$matrix_path" \
  "$state_path" \
  "$original_runner"; do
  [[ -f "$required_file" ]] || { echo "missing required file: $required_file" >&2; exit 2; }
done

[[ "$(sha256sum "$state_path" | awk '{print $1}')" == "$expected_state_sha" ]] || {
  echo "interrupted state SHA256 changed" >&2
  exit 3
}
[[ "$(sha256sum "$output_root/ATTR-STAGE71-COLOR-CONVNEXT-256-DVM-REPLAY-CCTV/last.pt" | awk '{print $1}')" == "$expected_last_sha" ]] || {
  echo "interrupted checkpoint SHA256 changed" >&2
  exit 4
}
if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "session already exists: $session_name" >&2
  exit 5
fi
if pgrep -f '^/root/miniconda3/bin/python .*/train_attribute.py.*ATTR-STAGE71' >/dev/null \
  || pgrep -f '^/root/miniconda3/bin/python .*/recover_stage71_teacher_matrix.py' >/dev/null; then
  echo "a Stage71 training/recovery process is already active" >&2
  exit 6
fi

tmux new-session -d -s "$session_name" \
  "$python_bin '$recovery_script' \
    --matrix '$matrix_path' \
    --output-root '$output_root' \
    --state '$state_path' \
    --original-runner '$original_runner' \
    --expected-state-sha256 '$expected_state_sha' \
    --expected-last-sha256 '$expected_last_sha' \
    --device cuda \
    --code-revision stage71-convnext-teachers-research-v1 \
    > '$recovery_log' 2>&1"

echo "started tmux:$session_name"
echo "log=$recovery_log"
