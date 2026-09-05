#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--log',type=Path,required=True); ap.add_argument('--manifest-report',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
 # training_log entries are Python repr, normalize safely via literal_eval.
 import ast
 log=[ast.literal_eval(x.split(' metrics=',1)[1]) for x in args.log.read_text(encoding='utf-8').splitlines() if ' metrics=' in x]
 best=max(log,key=lambda x:x.get('selection_score',-1)) if log else {}
 report={'schema_version':'smallboost-candidate-report-v1','candidate':'ATTR-V6-MNV3-256-SMALLBOOST','status':'rejected_fail_closed','production_model_unchanged':True,'production_artifact':'vehicle-attr-agent-e-224','frozen_video_used':False,'manifest_report':json.loads(args.manifest_report.read_text(encoding='utf-8')),'epochs_completed':len(log),'best_epoch':best,'gates':{'validation_type_precision':best.get('body_type_high_confidence_precision',0)>=.93,'validation_color_precision':best.get('color_high_confidence_precision',0)>=.93,'independent_test':False,'complex_scene_improvement':False,'trajectory_stability':False,'onnx_parity':False,'tensorrt_smoke':False,'frozen_video_not_used':True},'failure_reasons':['early_validation_type_precision_below_0.93','independent_test_not_run','complex_scene_improvement_not_proven','backend_gates_not_run'],'checkpoint':{'path':str(args.checkpoint),'sha256':sha(args.checkpoint)},'next_action':'retain production baseline; redesign small-target feature learning rather than simple duplication'}
 args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
