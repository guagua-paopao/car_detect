#!/usr/bin/env bash
set -euo pipefail

archive=/root/autodl-tmp/vcas/sources/stage193-abtd-v3/images.zip
expected_path=/root/autodl-tmp/vcas/sources/stage193-abtd-v3/images.zip
expected_bytes=1885610631
expected_sha=40c57c19488e05ac6cc50b724b493f7a0827a45b85c20dbdd79c02cee120ac28
stage195=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE195-ABTD-TRAIN-IMAGE-AUDIT-R2
stage196=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE196-ABTD-BODY-TEACHER-R2

resolved=$(readlink -f -- "${archive:?}")
[[ "${resolved:?}" == "${expected_path}" ]]
[[ -f "${resolved}" && ! -L "${archive}" ]]
[[ "$(stat -c %s -- "${resolved}")" == "${expected_bytes}" ]]
[[ "$(sha256sum -- "${resolved}" | awk '{print $1}')" == "${expected_sha}" ]]
[[ -s "${stage195}/stage195-abtd-train-image-audit.json" ]]
[[ -s "${stage196}/body-teacher-report.json" ]]
(cd "${stage195}" && sha256sum -c SHA256SUMS >/dev/null)
(cd "${stage196}" && sha256sum -c SHA256SUMS >/dev/null)

if pgrep -f 'audit_stage195_abtd_train_images.py|launch_stage194_abtd_train_images_download_r1.sh|audit_body_multiteacher_consensus.py' >/dev/null; then
  echo "relevant process still active; refusing cleanup" >&2
  exit 9
fi

printf 'verified cleanup target: %s (%s bytes)\n' "${resolved}" "${expected_bytes}"
rm -- "${resolved}"
[[ ! -e "${resolved}" ]]
echo "archive removed; metadata, labels, hashes, crops, manifests, reports and source URL remain"
df -h /root/autodl-tmp
