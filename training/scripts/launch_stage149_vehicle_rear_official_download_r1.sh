#!/usr/bin/env bash
set -euo pipefail

BASE=/root/autodl-tmp/vcas
SOURCE_ROOT="$BASE/sources/vehicle-rear-stage149"
PARTIAL="$SOURCE_ROOT/extracted.partial"
EXTRACTED="$SOURCE_ROOT/extracted"
STATE="$BASE/runs/attributes/ATTR-STAGE149-VEHICLE-REAR-OFFICIAL-DOWNLOAD-R1.state.json"
LOG="$BASE/runs/attributes/ATTR-STAGE149-VEHICLE-REAR-OFFICIAL-DOWNLOAD-R1.log"
SESSION=VCAS-DL-VEHICLEREAR149
DATA_URL=https://www.inf.ufpr.br/vri/databases/vehicle-reid/data.tgz
README_URL=https://raw.githubusercontent.com/icarofua/vehicle-rear/master/README.md
LICENSE_URL=https://raw.githubusercontent.com/icarofua/vehicle-rear/master/LICENSE
MIN_FREE_BYTES=13000000000

if [[ "${1:-}" != "--worker" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "$SESSION already running"
    exit 0
  fi
  tmux new-session -d -s "$SESSION" "bash '$0' --worker >>'$LOG' 2>&1"
  echo "started tmux:$SESSION"
  exit 0
fi

mkdir -p "$SOURCE_ROOT" "$(dirname "$STATE")"
if [[ -e "$PARTIAL" ]]; then
  echo "refusing to reuse partial extraction: $PARTIAL" >&2
  exit 2
fi
if [[ -e "$EXTRACTED" ]]; then
  echo "refusing to overwrite extraction: $EXTRACTED" >&2
  exit 2
fi

free_bytes=$(df -B1 --output=avail "$BASE" | tail -n 1 | tr -d ' ')
if [[ ! "$free_bytes" =~ ^[0-9]+$ ]] || (( free_bytes < MIN_FREE_BYTES )); then
  echo "insufficient free space: $free_bytes bytes" >&2
  exit 3
fi

curl -LIsS --max-time 60 "$DATA_URL" > "$SOURCE_ROOT/data.headers.txt"
curl -LfsS --retry 5 --retry-delay 5 "$README_URL" > "$SOURCE_ROOT/OFFICIAL_README.md"
curl -LfsS --retry 5 --retry-delay 5 "$LICENSE_URL" > "$SOURCE_ROOT/OFFICIAL_LICENSE"
sha256sum "$SOURCE_ROOT/OFFICIAL_README.md" "$SOURCE_ROOT/OFFICIAL_LICENSE" > "$SOURCE_ROOT/official-evidence.sha256"

mkdir "$PARTIAL"
curl -LfsS "$DATA_URL" \
  | tee >(sha256sum > "$SOURCE_ROOT/data.tgz.stream.sha256") \
  | tar -xzf - -C "$PARTIAL"
mv "$PARTIAL" "$EXTRACTED"

/root/miniconda3/bin/python - "$SOURCE_ROOT" "$EXTRACTED" "$STATE" <<'PY'
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import sys

source_root = Path(sys.argv[1])
extracted = Path(sys.argv[2])
state = Path(sys.argv[3])
files = [p for p in extracted.rglob("*") if p.is_file()]
extensions = Counter((p.suffix.lower() or "<none>") for p in files)
license_candidates = [
    str(p.relative_to(extracted))
    for p in files
    if any(token in p.name.lower() for token in ("license", "readme", "copying", "terms"))
]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

payload = {
    "schema_version": "stage149-vehicle-rear-official-download-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": "download_extraction_complete_pending_license_structure_audit",
    "source": {
        "official_repository": "https://github.com/icarofua/vehicle-rear",
        "official_dataset_url": "https://www.inf.ufpr.br/vri/databases/vehicle-reid/data.tgz",
        "paper_doi": "10.1109/ACCESS.2021.3097964",
        "repository_license": "Apache-2.0",
        "dataset_license_scope": "pending embedded archive and official-project scope audit",
    },
    "evidence": {
        "readme_sha256": sha256(source_root / "OFFICIAL_README.md"),
        "license_sha256": sha256(source_root / "OFFICIAL_LICENSE"),
        "archive_stream_sha256": (source_root / "data.tgz.stream.sha256").read_text().split()[0],
        "http_headers_sha256": sha256(source_root / "data.headers.txt"),
    },
    "extraction": {
        "root": str(extracted),
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "extensions": dict(sorted(extensions.items())),
        "embedded_license_candidates": sorted(license_candidates)[:200],
    },
    "policy": {
        "training_rows_created": 0,
        "test_accessed": False,
        "frozen_video_used": False,
        "production_model_modified": False,
        "backend_gates_run": False,
        "deployment_performed": False,
        "default_eligibility": "quarantined_until_audit",
    },
}
state.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
(state.with_suffix(state.suffix + ".sha256")).write_text(
    f"{sha256(state)}  {state.name}\n", encoding="utf-8"
)
print(json.dumps(payload, ensure_ascii=False))
PY

echo "Stage149 official Vehicle-Rear download and extraction complete; audit still required"
