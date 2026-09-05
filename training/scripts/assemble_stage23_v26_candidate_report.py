#!/usr/bin/env python3
"""Assemble independent-v26 candidate evidence and apply fail-closed gates."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha(p):
    h=hashlib.sha256();
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--candidate',type=Path,required=True); parser.add_argument('--baseline',type=Path,required=True); parser.add_argument('--manifest-report',type=Path,required=True); parser.add_argument('--output',type=Path,required=True); parser.add_argument('--onnx-parity',type=Path); parser.add_argument('--tensorrt-smoke',type=Path); parser.add_argument('--unknown-proxy',type=Path); ns=parser.parse_args()
    cand=json.loads(ns.candidate.read_text(encoding='utf-8')); base=json.loads(ns.baseline.read_text(encoding='utf-8'))
    small_key='vehicle_size=small'; ct=cand['test']['stratified']['body_type'].get(small_key,{}); bt=base['test']['stratified']['body_type'].get(small_key,{})
    body_delta=ct.get('high_confidence_coverage',0.0)-bt.get('high_confidence_coverage',0.0)
    color_unknown_delta=base['test']['color'].get('predicted_unknown_rate',0.0)-cand['test']['color'].get('predicted_unknown_rate',0.0)
    gates={
      'body_precision':cand['test']['body_type']['high_confidence_precision']>=.93,
      'body_coverage':cand['test']['body_type']['high_confidence_coverage']>=.45,
      'color_precision':cand['test']['color']['high_confidence_precision']>=.93,
      'color_coverage':cand['test']['color']['high_confidence_coverage']>=.25,
      'complex_type_coverage_plus_15pp':body_delta>=.15,
      'complex_color_unknown_minus_20pct':color_unknown_delta>=.20,
      'trajectory_stability':False,
      'onnx_pytorch_parity':bool(ns.onnx_parity and json.loads(ns.onnx_parity.read_text(encoding='utf-8')).get('gate') is True),
      'tensorrt_smoke':bool(ns.tensorrt_smoke and json.loads(ns.tensorrt_smoke.read_text(encoding='utf-8')).get('build')=='pass' and json.loads(ns.tensorrt_smoke.read_text(encoding='utf-8')).get('inference')=='pass'),
      'deployment_paused_by_user':True,
    }
    report={'schema_version':'attribute-ensemble-candidate-v26','candidate':'ATTR-V23-BODY-BMDRAW-TYPE+ATTR-V28-COLOR-MERGED-INPUT224','status':'rejected_fail_closed','production_artifact':'vehicle-attr-agent-e-224','production_model_unchanged':True,'deployment_performed':False,'frozen_video_used':False,'manifest_report':json.loads(ns.manifest_report.read_text(encoding='utf-8')),'candidate_ensemble':str(ns.candidate),'candidate_ensemble_sha256':sha(ns.candidate),'baseline_ensemble':str(ns.baseline),'baseline_ensemble_sha256':sha(ns.baseline),'validation':cand['validation'],'test':cand['test'],'comparison':{'small_body_coverage_candidate':ct.get('high_confidence_coverage',0.0),'small_body_coverage_baseline':bt.get('high_confidence_coverage',0.0),'small_body_coverage_delta':body_delta,'test_color_predicted_unknown_rate_candidate':cand['test']['color'].get('predicted_unknown_rate'),'test_color_predicted_unknown_rate_baseline':base['test']['color'].get('predicted_unknown_rate'),'color_unknown_rate_reduction':color_unknown_delta},'gates':gates,'backend_artifacts':{'onnx_parity':str(ns.onnx_parity) if ns.onnx_parity else None,'tensorrt_smoke':str(ns.tensorrt_smoke) if ns.tensorrt_smoke else None},'unknown_proxy':json.loads(ns.unknown_proxy.read_text(encoding='utf-8')) if ns.unknown_proxy else None,'failure_reasons':[k for k,v in gates.items() if not v],'next_action':'retain production baseline; complete C++ candidate adapter/real-engine smoke and independent track-fusion validation; use the unlabeled color proxy only for diagnosis, not as the release gate; deployment remains paused until frontend/backend changes are complete'}
    ns.output.parent.mkdir(parents=True,exist_ok=True); ns.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'status':report['status'],'gates':gates,'comparison':report['comparison']},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
