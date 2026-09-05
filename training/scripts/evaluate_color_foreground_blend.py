#!/usr/bin/env python3
"""Validation-only foreground-aware color logits experiment."""
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
 ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--candidate',type=Path,required=True); ap.add_argument('--baseline',type=Path,required=True); ap.add_argument('--candidate-temperature',type=float,default=.825); ap.add_argument('--baseline-temperature',type=float,default=.75); ap.add_argument('--threshold',type=float,default=.8); ap.add_argument('--device',default='cuda'); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 with a.manifest.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 cck=torch.load(a.candidate,map_location='cpu'); bck=torch.load(a.baseline,map_location='cpu'); cm=model_from_checkpoint(cck,pretrained=False); bm=model_from_checkpoint(bck,pretrained=False); cm.load_state_dict(cck['model_state']); bm.load_state_dict(bck['model_state']); cm.eval(); bm.eval(); dev=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu'); cm.to(dev); bm.to(dev); colors=list(cck['colors']); unknown=colors.index('unknown'); size=int(cck.get('input_size',224)); norm=transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD); resize=transforms.Resize((size,size),antialias=True)
 def collect(split,raw=False):
  if raw: rs=[r for r in rows if r.get('split') in {'validation','test'} and r.get('source_dataset')=='BMD-45-RAW-VAL' and r.get('color_supervised','false').lower() in {'false','0','no'}]
  else: rs=[r for r in rows if r.get('split')==split and r.get('review_status','approved')=='approved' and r.get('color') in colors and r.get('color_supervised','true').lower() not in {'false','0','no'}]
  class D(Dataset):
   def __len__(self): return len(rs)
   def __getitem__(self,i):
    with Image.open((a.manifest.parent/rs[i]['image_path']).resolve()) as im: im=im.convert('RGB')
    w,h=im.size; margin_x=int(round(w*.075)); margin_y=int(round(h*.075)); inner=im.crop((margin_x,margin_y,max(margin_x+1,w-margin_x),max(margin_y+1,h-margin_y)))
    return torch.stack([norm(transforms.ToTensor()(resize(im))),norm(transforms.ToTensor()(resize(inner)))]), (colors.index(rs[i]['color']) if not raw else -1)
  fs=[]; bs=[]; yy=[]
  with torch.no_grad():
   for x,y in DataLoader(D(),batch_size=128,shuffle=False,num_workers=8):
    q=x.to(dev); c0=[]; b0=[]
    for k in range(2):
     _,cl=cm(q[:,k]); _,bl=bm(q[:,k]); c0.append((cl/a.candidate_temperature).cpu()); b0.append((bl/a.baseline_temperature).cpu())
    # Keep batch dimension first so the final concatenation works for a short
    # last batch (stacking on dim=0 would make that batch dimension 1).
    fs.append(torch.stack(c0,dim=1)); bs.append(torch.stack(b0,dim=1)); yy.extend(y.tolist())
  return torch.cat(fs),torch.cat(bs),torch.tensor(yy),len(rs)
 val=collect('validation'); test=collect('test'); raw=collect('validation',True)
 def metric(c,b,y,w,raw=False):
  # foreground blend for each branch, then conservative candidate route.
  cb=(1-w)*c[:,0]+w*c[:,1]; bb=(1-w)*b[:,0]+w*b[:,1]; cp=cb.softmax(1); bp=bb.softmax(1); bnon=bp.clone(); bnon[:,unknown]=-1; bn,bpred=bnon.max(1); cnon=cp.clone(); cnon[:,unknown]=-1; cn,cpred=cnon.max(1); use=(bn<.8)&(cn>=.85)&(cn>=bn-.1); pred=torch.where(use,cpred,bpred); conf=torch.where(use,cn,bn); sel=(pred!=unknown)&(conf>=a.threshold); z={'inner_weight':w,'n':len(y),'selected':int(sel.sum()),'coverage':float(sel.float().mean()),'abstention_rate':float((~sel).float().mean()),'candidate_route_rate':float(use.float().mean())}
  if not raw: z.update(precision=float((pred[sel]==y[sel]).float().mean()) if int(sel.sum()) else 0.,accuracy=float((pred==y).float().mean()),predicted_unknown_rate=float((pred==unknown).float().mean()))
  return z
 table=[metric(val[0],val[1],val[2],w) for w in [0,.25,.5,.75,1.0]]; passing=[z for z in table if z['precision']>=.93]; chosen=max(passing,key=lambda z:(z['coverage'],z['precision'])) if passing else max(table,key=lambda z:z.get('precision',0)); out={'schema_version':'color-foreground-blend-v1','manifest':str(a.manifest),'frozen_video_used':False,'selection_policy':'inner crop blend selected on labeled validation only; raw proxy not ground truth','validation':{'chosen':chosen,'table':table},'test':metric(test[0],test[1],test[2],chosen['inner_weight']),'unlabeled_proxy':metric(raw[0],raw[1],raw[2],chosen['inner_weight'],True)}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'validation':chosen,'test':out['test'],'unlabeled_proxy':out['unlabeled_proxy']},ensure_ascii=False))
if __name__=='__main__': main()
