#!/usr/bin/env python3
"""Combine per-split stratified reports emitted by evaluate_attribute_baseline."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--validation',type=Path,required=True); ap.add_argument('--test',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); v=json.loads(a.validation.read_text(encoding='utf-8')); t=json.loads(a.test.read_text(encoding='utf-8')); out={'schema_version':'attribute-evaluation-combined-v1','manifest':v.get('manifest'),'checkpoint':v.get('checkpoint'),'frozen_video_used':False,'validation':v,'test':t,'release_target_met':all(x[h]['high_confidence_precision']>=p and x[h]['high_confidence_coverage']>=c for x in (v,t) for h,p,c in (('body_type',.93,.45),('color',.93,.25)))}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'output':str(a.output),'release_target_met':out['release_target_met']},ensure_ascii=False))
if __name__=='__main__': main()
