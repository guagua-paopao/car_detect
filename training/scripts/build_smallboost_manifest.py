#!/usr/bin/env python3
"""Oversample approved train rows marked small_target without touching val/test."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--repeat',type=int,default=4); args=ap.parse_args()
    with args.input.open('r',encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f); rows=list(r); fields=r.fieldnames or []
    train=[x for x in rows if x.get('split')=='train']; extra=[]
    for x in train:
        if str(x.get('small_target','')).strip().lower() in {'1','true','yes'} or str(x.get('vehicle_size','')).strip().lower() == 'small':
            for _ in range(max(0,args.repeat-1)): extra.append(dict(x))
    out=rows+extra; args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(out)
    rep={'schema_version':'smallboost-manifest-v1','input_rows':len(rows),'output_rows':len(out),'train_rows':len(train),'small_train_rows':len(extra)//max(1,args.repeat-1) if args.repeat>1 else 0,'repeat':args.repeat,'validation_test_unchanged':True,'frozen_video_used':False}
    args.output.with_suffix('.report.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(rep,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
