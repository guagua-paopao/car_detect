#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/vcas/code/training_stage152_lvad_pool_r3
BUILDER=$ROOT/scripts/build_stage92_lvad_night_pool.py
TEST=$ROOT/tests/test_build_stage92_lvad_night_pool.py
ARCHIVE=/root/autodl-tmp/vcas/sources/lvad-stage92/lvad-v2.zip
AUDIT=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE92-LVAD-AUDIT-R2/report.json
EVIDENCE=/root/autodl-tmp/vcas/sources/lvad-stage92/lvad-source-evidence-v2-r2.json
BASE83=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage83-body-v2-merged-v5/attribute_manifest.stage83-body-v2-merged.csv
BASE89=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage89-mio-balanced-expansion-r4/attribute_manifest.stage89-mio-balanced.csv
BASE91=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage91-inatrc-train-crops-v1/attribute_manifest.stage91-all.csv
OUTPUT=/root/autodl-tmp/vcas/datasets/attribute-domain-v2/stage152-lvad-night-pool-r3
STATE=/root/autodl-tmp/vcas/runs/attributes/ATTR-STAGE152-LVAD-NIGHT-POOL-R3.state.json

test ! -e "$OUTPUT"
test ! -e "$STATE"
echo "9ded6c9641f5293f2c3c9818b7f17c89adcba7eac04a032b08ead6777d92966c  $BUILDER" | sha256sum -c -
echo "bdadd73fbf79061f36543964712af908d385a88eeae3f2c63cd45a6a4618c6d4  $TEST" | sha256sum -c -
echo "e9219543e93af7f2771b2743441af0b1a9bd2ccf6de4165cddc22a506d4d0e67  $ARCHIVE" | sha256sum -c -
echo "f61fa3582f47cdcda3bbceef1f3ea08daef30ac2cbbe170464f8c8a855da4a68  $AUDIT" | sha256sum -c -
echo "ab98375d6f0449dda85a8d95922e20f30b5391172aa3e17ef0910b57a9d7c513  $EVIDENCE" | sha256sum -c -
echo "f0d95b424926a38b6e162fd0fa78da1329e7eab180e2a2dd8df42f26e06fcdef  $BASE83" | sha256sum -c -
echo "acb0e4d42aeb79d429dee4e864d41939a84f7b1d72298c1c830a589be90bddf7  $BASE89" | sha256sum -c -
echo "d9cf4d5c8469a5da47d5b0e7b5df78f628b5084738f0de62037993bee3e6acf1  $BASE91" | sha256sum -c -

"$PYTHON" -m py_compile "$BUILDER"
"$PYTHON" -m unittest "$TEST"
set +e
"$PYTHON" "$BUILDER" \
  --archive "$ARCHIVE" \
  --expected-archive-sha256 e9219543e93af7f2771b2743441af0b1a9bd2ccf6de4165cddc22a506d4d0e67 \
  --audit-report "$AUDIT" \
  --expected-audit-report-sha256 f61fa3582f47cdcda3bbceef1f3ea08daef30ac2cbbe170464f8c8a855da4a68 \
  --source-evidence "$EVIDENCE" \
  --expected-source-evidence-sha256 ab98375d6f0449dda85a8d95922e20f30b5391172aa3e17ef0910b57a9d7c513 \
  --base-manifest "$BASE83" \
  --expected-base-manifest-sha256 f0d95b424926a38b6e162fd0fa78da1329e7eab180e2a2dd8df42f26e06fcdef \
  --base-manifest "$BASE89" \
  --expected-base-manifest-sha256 acb0e4d42aeb79d429dee4e864d41939a84f7b1d72298c1c830a589be90bddf7 \
  --base-manifest "$BASE91" \
  --expected-base-manifest-sha256 d9cf4d5c8469a5da47d5b0e7b5df78f628b5084738f0de62037993bee3e6acf1 \
  --output-images "$OUTPUT/images" \
  --output-manifest "$OUTPUT/attribute_manifest.stage152-lvad-night.csv" \
  --output-report "$OUTPUT/stage152-lvad-night-pool-report.json" \
  --minimum-temporal-gap 1 \
  --track-window-frames 30 \
  --frame-near-duplicate-hamming 0 \
  --crop-near-duplicate-hamming 4 \
  --minimum-rows 2500 \
  --minimum-car-rows 2100 \
  --minimum-truck-rows 200
BUILD_RC=$?
set -e

"$PYTHON" - "$OUTPUT" "$STATE" "$BUILD_RC" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

output, state, return_code = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
report_path = output / "stage152-lvad-night-pool-report.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
manifest = output / "attribute_manifest.stage152-lvad-night.csv"

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

payload = {
    "schema_version": "stage152-lvad-night-pool-state-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": "pass_training_supplement_ready" if return_code == 0 and report.get("status") == "pass" else "fail_closed",
    "return_code": return_code,
    "report": str(report_path),
    "report_sha256": sha(report_path),
    "manifest": str(manifest) if manifest.is_file() else None,
    "manifest_sha256": sha(manifest) if manifest.is_file() else None,
    "rows": report.get("output", {}).get("rows"),
    "coarse_family_counts": report.get("output", {}).get("coarse_family_counts", {}),
    "selection": report.get("selection", {}),
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
