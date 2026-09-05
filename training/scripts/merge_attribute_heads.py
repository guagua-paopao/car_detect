#!/usr/bin/env python3
"""Create an isolated single-checkpoint candidate from independent head weights."""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True); ap.add_argument('--body-source',type=Path,required=True); ap.add_argument('--color-source',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    import torch
    base=torch.load(args.base,map_location='cpu'); body=torch.load(args.body_source,map_location='cpu'); color=torch.load(args.color_source,map_location='cpu')
    if base['body_types'] != body['body_types'] or base['colors'] != color['colors']:
        raise RuntimeError('label order mismatch')
    out=copy.deepcopy(base); state=out['model_state']; bstate=body['model_state']; cstate=color['model_state']
    for key in ('body_type_head.weight','body_type_head.bias'): state[key]=bstate[key].clone()
    for key in ('color_head.weight','color_head.bias'): state[key]=cstate[key].clone()
    out['model_state']=state; out['input_size']=int(base.get('input_size',224)); out['run_id']=args.output.stem; out['model_card']={'merge_policy':'independent body/color head merge; no production overwrite','base_checkpoint':str(args.base),'body_source':str(args.body_source),'color_source':str(args.color_source),'base_sha256':sha(args.base),'body_source_sha256':sha(args.body_source),'color_source_sha256':sha(args.color_source),'frozen_video_used':False}
    args.output.parent.mkdir(parents=True,exist_ok=True); torch.save(out,args.output); print(json.dumps({'output':str(args.output),'sha256':sha(args.output),'input_size':out['input_size'],'base':str(args.base)},ensure_ascii=False))
if __name__=='__main__': main()
