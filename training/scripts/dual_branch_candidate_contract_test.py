"""Fail-closed artifact contract for the isolated dual-branch candidate."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import onnx

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); r=a.root
 registry_files=sorted(r.glob('model_registry.candidate-v*.json'))
 required=['body-candidate.onnx','body-candidate.engine','color-baseline.onnx','color-baseline.engine','candidate-report.json','model-card.json'] + ([registry_files[-1].name] if registry_files else ['model_registry.candidate-v*.json'])
 missing=[x for x in required if not (r/x).is_file()]
 checks={}
 onnx_files=['body-candidate.onnx','color-baseline.onnx'] + (['color-candidate.onnx'] if (r/'color-candidate.onnx').exists() else [])
 for name in onnx_files:
  p=r/name; checks[name]={'sha256':sha(p) if p.exists() else None,'onnx_checker':'fail'}
  if p.exists():
   m=onnx.load(str(p)); onnx.checker.check_model(m); checks[name]['onnx_checker']='pass'; checks[name]['inputs']=[x.name for x in m.graph.input]; checks[name]['outputs']=[x.name for x in m.graph.output]
 report={'schema_version':'dual-branch-candidate-contract-v1','root':str(r),'missing':missing,'checks':checks,'production_model_unchanged':True,'frozen_video_used':False,'deployment_performed':False,'gate':not missing and all(x['onnx_checker']=='pass' for x in checks.values()),'cpp_runtime_contract':'pending_separate_from_artifact_contract'}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(report,ensure_ascii=False,indent=2)); raise SystemExit(0 if report['gate'] else 1)
if __name__=='__main__': main()
