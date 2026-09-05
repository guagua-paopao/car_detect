#!/usr/bin/env python3
"""Change only the declared evaluation input size for an isolated checkpoint."""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--size',type=int,required=True); a=ap.parse_args(); import torch
 x=torch.load(a.input,map_location='cpu'); x=copy.deepcopy(x); x['input_size']=a.size; x.setdefault('model_card',{})['retargeted_input_size']=a.size; x['model_card']['retarget_policy']='metadata-only isolated evaluation; retraining required before deployment'; a.output.parent.mkdir(parents=True,exist_ok=True); torch.save(x,a.output); print(json.dumps({'output':str(a.output),'input_size':a.size}))
if __name__=='__main__': main()
