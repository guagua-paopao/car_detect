#!/usr/bin/env python3
"""Keep only color-supervised training rows while preserving held-out splits."""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import Counter
from pathlib import Path

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--report',type=Path,required=True); a=ap.parse_args()
 with a.input.open('r',encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 out=[r for r in rows if r.get('split')!='train' or (str(r.get('color_supervised','true')).lower() not in {'false','0','no'} and r.get('color','').lower() not in {'','unknown'})]
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(out)
 report={'schema_version':'color-only-manifest-v28','input_rows':len(rows),'output_rows':len(out),'train_rows':sum(r.get('split')=='train' for r in out),'train_color_counts':dict(sorted(Counter(r.get('color') for r in out if r.get('split')=='train').items())),'validation_rows':sum(r.get('split')=='validation' for r in out),'test_rows':sum(r.get('split')=='test' for r in out),'policy':'only explicitly supervised non-unknown color rows are used for training; validation/test are untouched','frozen_video_used':False,'output_manifest':str(a.output),'output_manifest_sha256':sha(a.output)}
 a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
