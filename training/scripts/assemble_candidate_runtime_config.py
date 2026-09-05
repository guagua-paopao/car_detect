#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--registry',type=Path,required=True); ap.add_argument('--artifact-id',required=True); ap.add_argument('--onnx',required=True); ap.add_argument('--engine',required=True); args=ap.parse_args()
    cfg=json.loads(args.input.read_text(encoding='utf-8')); reg_path=str(args.registry).replace('\\','/'); cfg['model_registry_path']=reg_path if reg_path.startswith('./') else './'+reg_path; model=cfg['models']['vehicle_attribute']; model.update({'artifact':args.artifact_id,'delivery_status':'engine_validated','onnx_path':'./'+args.onnx.lstrip('./'),'engine_path':'./'+args.engine.lstrip('./')}); args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'output':str(args.output),'artifact':args.artifact_id},ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())
