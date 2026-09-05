#!/usr/bin/env python3
"""Promote a validated attribute candidate with a recoverable snapshot."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil
from datetime import datetime, timezone
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(path, value):
    tmp=Path(str(path)+'.tmp'); tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); os.replace(tmp,path)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',type=Path,required=True); ap.add_argument('--candidate-registry',type=Path,required=True); ap.add_argument('--candidate-config',type=Path,required=True); ap.add_argument('--rollback-dir',type=Path,required=True); ap.add_argument('--report',type=Path,required=True); args=ap.parse_args(); root=args.project_root.resolve(); reg_path=root/'models'/'manifests'/'model_registry.v1.json'; cfg_path=root/'config'/'vehicle_analytics.yaml';
    if args.rollback_dir.exists(): raise SystemExit(f'rollback target already exists: {args.rollback_dir}')
    cand=json.loads(args.candidate_registry.read_text(encoding='utf-8')); attr=next(a for a in cand['artifacts'] if a.get('role')=='attributes'); files=attr['files']; onnx=root/files['onnx_path']; engine=root/files['engine_path'];
    if not onnx.exists() or not engine.exists(): raise SystemExit('candidate payload missing')
    if sha(onnx)!=files['onnx_sha256'] or sha(engine)!=files['engine_sha256']: raise SystemExit('candidate hash mismatch')
    # Snapshot every production input needed for an exact rollback.
    args.rollback_dir.mkdir(parents=True,exist_ok=False); (args.rollback_dir/'config').mkdir(); (args.rollback_dir/'models'/'manifests').mkdir(parents=True); (args.rollback_dir/'models').mkdir(exist_ok=True); (args.rollback_dir/'engines').mkdir(exist_ok=True)
    for src,dst in ((cfg_path,args.rollback_dir/'config'/'vehicle_analytics.yaml'),(reg_path,args.rollback_dir/'models'/'manifests'/'model_registry.v1.json'),(root/'models'/'vehicle-attr-v1.onnx',args.rollback_dir/'models'/'vehicle-attr-v1.onnx'),(root/'engines'/'vehicle-attr-v1.engine',args.rollback_dir/'engines'/'vehicle-attr-v1.engine')): shutil.copy2(src,dst)
    deployed=json.loads(json.dumps(cand)); deployed['registry_version']='vehicle-model-registry-v2-domain-bmd45-agent-e'; deployed['created_at']=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'); deployed['candidate_policy']={'production_registry_overwritten':True,'rollback_path':str(args.rollback_dir),'source_candidate':str(args.candidate_registry)}
    for a in deployed['artifacts']: a['delivery_status']='deployed'
    cfg=json.loads(args.candidate_config.read_text(encoding='utf-8')); cfg['model_registry_path']='./models/manifests/model_registry.v1.json'; cfg['models']['vehicle_attribute']['delivery_status']='deployed';
    atomic_json(reg_path,deployed); atomic_json(cfg_path,cfg)
    report={'schema_version':'1.0','status':'deployed','artifact_id':attr['artifact_id'],'production_registry':str(reg_path),'production_config':str(cfg_path),'onnx':{'path':str(onnx),'sha256':sha(onnx)},'engine':{'path':str(engine),'sha256':sha(engine)},'rollback_dir':str(args.rollback_dir),'rollback_files_sha256':{str(p.relative_to(args.rollback_dir)):sha(p) for p in args.rollback_dir.rglob('*') if p.is_file()}}
    args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
