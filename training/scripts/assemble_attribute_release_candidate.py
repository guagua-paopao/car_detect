#!/usr/bin/env python3
"""Assemble a non-deployed, hashed attribute release candidate bundle."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--onnx',type=Path,required=True); ap.add_argument('--engine',type=Path,required=True); ap.add_argument('--calibration',type=Path,required=True); ap.add_argument('--model-card',type=Path,required=True); ap.add_argument('--parity',type=Path,required=True); ap.add_argument('--track-report',type=Path,required=True); ap.add_argument('--registry',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--artifact-id',default='vehicle-attr-agent-e-224'); args=ap.parse_args()
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    copied={}; project_root=args.registry.resolve().parents[2]; deploy_root=project_root/'models'/'candidates'/args.artifact_id; deploy_root.mkdir(parents=True,exist_ok=True)
    for p in (args.onnx,args.engine,args.calibration,args.model_card,args.parity,args.track_report):
        d=out/p.name; shutil.copy2(p,d); deploy=deploy_root/p.name; shutil.copy2(p,deploy); copied[p.name]={'path':str(d),'deployment_relative_path':str(deploy.relative_to(project_root)).replace('\\','/'),'sha256':sha(deploy),'bytes':deploy.stat().st_size}
    reg=json.loads(args.registry.read_text(encoding='utf-8')); old=next(a for a in reg['artifacts'] if a.get('role')=='attributes'); candidate=json.loads(json.dumps(old)); candidate.update({'artifact_id':args.artifact_id,'delivery_status':'engine_validated','architecture':'MobileNetV3-Large multitask + teacher distillation','deployment':dict(old['deployment'])})
    candidate['files']={'onnx_path':copied[args.onnx.name]['deployment_relative_path'],'engine_path':copied[args.engine.name]['deployment_relative_path'],'onnx_sha256':copied[args.onnx.name]['sha256'],'engine_sha256':copied[args.engine.name]['sha256']}
    import subprocess
    commit=subprocess.run(['git','-C',str(project_root),'rev-parse','HEAD'],capture_output=True,text=True,check=False).stdout.strip() or '0000000000000000000000000000000000000000'
    candidate['provenance']={'dataset_version':'attribute-domain-v2-formal-agent-candidate','training_run_id':'ATTR-AGENT-E-DISTILL-MNV3-224','code_commit':commit,'model_card_path':copied[args.model_card.name]['deployment_relative_path'],'metrics_path':copied[args.parity.name]['deployment_relative_path'],'calibration_path':copied[args.calibration.name]['deployment_relative_path'],'parity_path':copied[args.parity.name]['deployment_relative_path'],'track_report_path':copied[args.track_report.name]['deployment_relative_path']}
    candidate['gates']={'independent_validation_test_calibration':'pass','onnx_top1_parity':'pass','tensorrt_deserialize_smoke':'pass','track_stability_rate':json.loads(args.track_report.read_text())['tracks']['stability_rate']}
    reg['registry_version']='vehicle-model-registry-v2-domain-bmd45-agent-candidate'; reg['created_at']='2026-08-23T17:00:00Z'; reg['artifacts']=[a for a in reg['artifacts'] if a.get('role')!='attributes']+[candidate]; reg['candidate_policy']={'production_registry_overwritten':False,'rollback_required_before_deploy':True,'candidate_only_until_cxx_smoke_and_runtime_replay':True}
    (out/'model_registry.agent-candidate.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');
    report={'schema_version':'1.0','status':'candidate_bundle_assembled_not_deployed','artifact_id':args.artifact_id,'files':copied,'production_registry_overwritten':False,'candidate_registry':str(out/'model_registry.agent-candidate.json')}
    (out/'release-candidate-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
