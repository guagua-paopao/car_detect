#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/vcas/code/training_stage153_manifest_r2
BUILDER=$ROOT/scripts/build_stage153_lvad_body_repair_manifest.py
TEST=$ROOT/tests/test_build_stage153_lvad_body_repair_manifest.py
BASE=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage83-body-v2-merged-v5/attribute_manifest.stage83-body-v2-merged.csv
NIGHT=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage152-lvad-night-pool-r3/attribute_manifest.stage152-lvad-night.csv
NIGHT_REPORT=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage152-lvad-night-pool-r3/stage152-lvad-night-pool-report.json
OUTPUT=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage153-lvad-body-repair-r2
MANIFEST=$OUTPUT/attribute_manifest.stage153-lvad-body-repair.csv
REPORT=$OUTPUT/stage153-lvad-body-repair-manifest-report.json
STATE=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE153-LVAD-BODY-REPAIR-MANIFEST-R2.state.json

test ! -e "$OUTPUT"
test ! -e "$STATE"
echo "1dac30c8c3fc74b3d77426f5f1ed038ec2e3c7bc31c9bfa508cc06181d553dd8  $BUILDER" | sha256sum -c -
echo "26b3740d004d5ea30d11c45dc16d02c27148815f4c9b25fa5674778b4121cd96  $TEST" | sha256sum -c -
echo "f0d95b424926a38b6e162fd0fa78da1329e7eab180e2a2dd8df42f26e06fcdef  $BASE" | sha256sum -c -
echo "25d8e8ff7b586cbee2622ba90eb08ab23fbcbf2acff5e496feedaff7348a6cdb  $NIGHT" | sha256sum -c -
echo "f6c8672e00db713ec3005226246150b5730a0a6b3c05c5eae8717056edf0bbc4  $NIGHT_REPORT" | sha256sum -c -

"$PYTHON" -m py_compile "$BUILDER"
"$PYTHON" -m unittest "$TEST"
set +e
"$PYTHON" "$BUILDER" \
  --base-manifest "$BASE" \
  --expected-base-sha256 f0d95b424926a38b6e162fd0fa78da1329e7eab180e2a2dd8df42f26e06fcdef \
  --night-manifest "$NIGHT" \
  --expected-night-sha256 25d8e8ff7b586cbee2622ba90eb08ab23fbcbf2acff5e496feedaff7348a6cdb \
  --night-report "$NIGHT_REPORT" \
  --expected-night-report-sha256 f6c8672e00db713ec3005226246150b5730a0a6b3c05c5eae8717056edf0bbc4 \
  --output-manifest "$MANIFEST" \
  --output-report "$REPORT" \
  --exact-per-class 1750 \
  --occluded-per-eligible-class 525 \
  --maximum-cross-split-dhash 4
BUILD_RC=$?
set -e

"$PYTHON" - "$REPORT" "$MANIFEST" "$STATE" "$BUILD_RC" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

report_path, manifest, state, return_code = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
report = json.loads(report_path.read_text(encoding="utf-8"))

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

payload = {
    "schema_version": "stage153-lvad-body-repair-manifest-state-v2",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": "pass_training_manifest_ready" if return_code == 0 and report.get("status") == "pass_training_manifest_ready" else "fail_closed",
    "return_code": return_code,
    "report": str(report_path),
    "report_sha256": sha(report_path),
    "manifest": str(manifest) if manifest.is_file() else None,
    "manifest_sha256": sha(manifest) if manifest.is_file() else None,
    "output": report.get("output", {}),
    "leakage": report.get("leakage", {}),
    "policy": report.get("policy", {}),
    "failures": report.get("failures", []),
    "test_accessed": False,
    "frozen_video_used": False,
    "production_model_modified": False,
    "training_started": False,
    "deployment_performed": False,
}
state.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
PY

exit "$BUILD_RC"
