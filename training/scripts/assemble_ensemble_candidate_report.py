#!/usr/bin/env python3
"""Fail-closed decision report for independent type/color ensemble."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def load(p: Path): return json.loads(p.read_text(encoding="utf-8"))
def sha(p: Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ensemble",type=Path,required=True); ap.add_argument("--baseline-hard",type=Path,required=True); ap.add_argument("--body-hard",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--candidate",default="ATTR-ENSEMBLE-BODY-CANDIDATE"); args=ap.parse_args()
    e=load(args.ensemble); b=load(args.baseline_hard); bh=load(args.body_hard)
    small_base=b["stratified"]["body_type"]["vehicle_size=small"]["high_confidence_coverage"]
    small_candidate=bh["stratified"]["body_type"]["vehicle_size=small"]["high_confidence_coverage"]
    unknown_base=b["color"]["predicted_unknown_rate"]
    unknown_candidate=e["test"]["color"]["predicted_unknown_rate"]
    gates={
      "body_precision":e["test"]["body_type"]["high_confidence_precision"]>=.93,
      "body_coverage":e["test"]["body_type"]["high_confidence_coverage"]>=.45,
      "color_precision":e["test"]["color"]["high_confidence_precision"]>=.93,
      "color_coverage":e["test"]["color"]["high_confidence_coverage"]>=.25,
      "complex_type_coverage_plus_15pp":small_candidate-small_base>=.15,
      "complex_color_unknown_minus_20pct":unknown_base>0 and (unknown_base-unknown_candidate)/unknown_base>=.20,
      "trajectory_stability":False,
      "onnx_pytorch_parity":False,
      "tensorrt_smoke":False,
      "frozen_video_not_used":True,
    }
    report={"schema_version":"attribute-ensemble-candidate-report-v1","candidate":args.candidate,"status":"rejected_fail_closed" if not all(gates.values()) else "eligible_pending_backend_gates","production_model_unchanged":True,"production_artifact":"vehicle-attr-agent-e-224","frozen_video_used":False,"ensemble":e,"complex_scene_comparison":{"baseline_small_type_coverage":small_base,"candidate_small_type_coverage":small_candidate,"delta":small_candidate-small_base,"baseline_color_unknown_rate":unknown_base,"candidate_color_unknown_rate":unknown_candidate},"gates":gates,"failure_reasons":[k for k,v in gates.items() if not v],"artifacts":{"ensemble":{"path":str(args.ensemble),"sha256":sha(args.ensemble)},"baseline_hard":{"path":str(args.baseline_hard),"sha256":sha(args.baseline_hard)},"body_hard":{"path":str(args.body_hard),"sha256":sha(args.body_hard)}},"next_action":"retain production baseline; redesign small-target feature learning and continue isolated calibration"}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
