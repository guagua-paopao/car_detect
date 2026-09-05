#!/usr/bin/env python3
"""Search a validation-only color router; unlabeled rows are proxy evidence only."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.multitask_mobilenet_v3 import model_from_checkpoint,IMAGENET_MEAN,IMAGENET_STD

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--candidate',type=Path,required=True); ap.add_argument('--baseline',type=Path,required=True)
 ap.add_argument('--candidate-temperature',type=float,default=.825); ap.add_argument('--baseline-temperature',type=float,default=.75)
 ap.add_argument('--threshold',type=float,default=.8); ap.add_argument('--device',default='cuda'); ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--validation-precision-floor',type=float,default=.93)
 a=ap.parse_args()
 with a.manifest.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 cck=torch.load(a.candidate,map_location='cpu'); bck=torch.load(a.baseline,map_location='cpu')
 cm=model_from_checkpoint(cck,pretrained=False); bm=model_from_checkpoint(bck,pretrained=False); cm.load_state_dict(cck['model_state']); bm.load_state_dict(bck['model_state']); cm.eval(); bm.eval()
 dev=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu'); cm.to(dev); bm.to(dev)
 colors=list(cck['colors']); unknown=colors.index('unknown'); size=int(cck.get('input_size',224)); tf=transforms.Compose([transforms.Resize((size,size),antialias=True),transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
 def collect(split, unlabeled=False):
  if unlabeled:
   rs=[r for r in rows if r.get('split') in {'validation','test'} and r.get('source_dataset')=='BMD-45-RAW-VAL' and r.get('color_supervised','false').lower() in {'false','0','no'}]
  else:
   rs=[r for r in rows if r.get('split')==split and r.get('review_status','approved')=='approved' and r.get('color') in colors and r.get('color_supervised','true').lower() not in {'false','0','no'}]
  class D(Dataset):
   def __len__(self): return len(rs)
   def __getitem__(self,i):
    with Image.open((a.manifest.parent/rs[i]['image_path']).resolve()) as im: im=im.convert('RGB')
    return tf(im), (colors.index(rs[i]['color']) if not unlabeled else -1)
  cl=[]; bl=[]; yy=[]; loader=DataLoader(D(),batch_size=128,shuffle=False,num_workers=8)
  with torch.no_grad():
    for x,y in loader:
     q=x.to(dev); _,co=cm(q); _,bo=bm(q); cl.append((co/a.candidate_temperature).cpu()); bl.append((bo/a.baseline_temperature).cpu()); yy.extend(y.tolist())
  return torch.cat(cl),torch.cat(bl),torch.tensor(yy),len(rs)
 val=collect('validation'); test=collect('test'); raw=collect('validation',True)
 def route(c,b,base_gate,cand_gate,margin):
  bp=b.softmax(1); cp=c.softmax(1); bconf,bpred=bp.max(1); cconf,cpred=cp.max(1)
  bnon=bp.clone(); bnon[:,unknown]=-1; bnconf,bnpred=bnon.max(1); cnon=cp.clone(); cnon[:,unknown]=-1; cnconf,cnpred=cnon.max(1)
  use_c=(bnconf<base_gate)&(cnconf>=cand_gate)&(cnconf>=bnconf+margin)
  pred=torch.where(use_c,cnpred,bnpred); conf=torch.where(use_c,cnconf,bnconf); return pred,conf,use_c
 def metric(c,b,y,base_gate,cand_gate,margin,candidate_accept_gate=.8,labeled=True):
  pred,conf,use_c=route(c,b,base_gate,cand_gate,margin)
  # Keep the baseline acceptance threshold fixed, but allow a separately
  # validated threshold for candidate-routed frames. This is selected only
  # on labeled validation; raw rows remain diagnostic evidence.
  sel=(pred!=unknown)&torch.where(use_c,conf>=candidate_accept_gate,conf>=a.threshold)
  z={'base_gate':base_gate,'candidate_gate':cand_gate,'candidate_accept_gate':candidate_accept_gate,'margin':margin,'selected':int(sel.sum()),'n':len(y),'coverage':float(sel.float().mean()),'candidate_route_rate':float(use_c.float().mean()),'abstention_rate':float((~sel).float().mean())}
  if labeled: z.update(precision=float((pred[sel]==y[sel]).float().mean()) if int(sel.sum()) else 0.0,accuracy=float((pred==y).float().mean()),predicted_unknown_rate=float((pred==unknown).float().mean()))
  return z
 grid=[]
 for bg in (.80,.85,.90,.95):
  for cg in (.45,.50,.55,.60,.65,.70,.75,.80,.85,.90):
   for margin in (-.10,-.05,0.0,.05):
    for ca in (.45,.50,.55,.60,.65,.70,.75,.80): grid.append(metric(val[0],val[1],val[2],bg,cg,margin,ca))
 passing=[z for z in grid if z['precision']>=a.validation_precision_floor]; chosen=max(passing,key=lambda z:(z['coverage'],z['precision'])) if passing else max(grid,key=lambda z:z['precision'])
 out={'schema_version':'color-adaptive-router-v2','manifest':str(a.manifest),'frozen_video_used':False,'selection_policy':'router and candidate acceptance threshold selected on labeled validation only; raw BMD rows are proxy evidence','validation_precision_floor':a.validation_precision_floor,'validation':{'chosen':chosen,'candidates':grid},'test':metric(test[0],test[1],test[2],chosen['base_gate'],chosen['candidate_gate'],chosen['margin'],chosen['candidate_accept_gate']),'unlabeled_proxy':metric(raw[0],raw[1],raw[2],chosen['base_gate'],chosen['candidate_gate'],chosen['margin'],chosen['candidate_accept_gate'],False)}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'validation':chosen,'test':out['test'],'unlabeled_proxy':out['unlabeled_proxy']},ensure_ascii=False))
if __name__=='__main__': main()
