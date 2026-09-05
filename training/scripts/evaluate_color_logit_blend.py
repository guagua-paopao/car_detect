"""Validation-only search for a conservative candidate/baseline color blend."""
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
 cck=torch.load(a.candidate,map_location='cpu'); bck=torch.load(a.baseline,map_location='cpu'); cm=model_from_checkpoint(cck,pretrained=False); bm=model_from_checkpoint(bck,pretrained=False); cm.load_state_dict(cck['model_state']); bm.load_state_dict(bck['model_state']); cm.eval(); bm.eval(); dev=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu'); cm.to(dev); bm.to(dev)
 colors=list(cck['colors']); unknown=colors.index('unknown'); size=int(cck.get('input_size',224)); tf=transforms.Compose([transforms.Resize((size,size),antialias=True),transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
 def collect(split):
  rs=[r for r in rows if r.get('split')==split and r.get('review_status','approved')=='approved' and r.get('color') in colors and r.get('color_supervised','true').lower() not in {'false','0','no'}]
  class D(Dataset):
   def __len__(self): return len(rs)
   def __getitem__(self,i):
    with Image.open(a.manifest.parent/rs[i]['image_path']) as im: im=im.convert('RGB')
    return tf(im), colors.index(rs[i]['color'])
  loader=DataLoader(D(),batch_size=128,shuffle=False,num_workers=8); c=[]; b=[]; y=[]
  with torch.no_grad():
   for x,t in loader:
    cl,_=cm(x.to(dev)); _,bl=bm(x.to(dev)); c.append((cl/a.candidate_temperature).cpu()); b.append((bl/a.baseline_temperature).cpu()); y.extend(t.tolist())
  return torch.cat(c),torch.cat(b),torch.tensor(y),len(rs)
 out={'schema_version':'color-logit-blend-v1','manifest':str(a.manifest),'frozen_video_used':False,'threshold':a.threshold,'validation':{},'test':{}}
 val=collect('validation'); test=collect('test')
 def metric(c,b,y,w):
  p=(w*c+(1-w)*b).softmax(1); conf,pred=p.max(1); sel=(pred!=unknown)&(conf>=a.threshold); n=int(sel.sum()); return {'weight_candidate':w,'evaluated':len(y),'selected':n,'precision':float((pred[sel]==y[sel]).float().mean()) if n else 0.0,'coverage':n/len(y) if len(y) else 0.0,'accuracy':float((pred==y).float().mean()) if len(y) else 0.0,'predicted_unknown_rate':float((pred==unknown).float().mean()) if len(y) else 0.0}
 table=[metric(val[0],val[1],val[2],round(i/20,2)) for i in range(21)]; passing=[x for x in table if x['precision']>=.93]; chosen=max(passing,key=lambda x:(x['coverage'],x['precision'])) if passing else table[10]; out['validation']={'chosen':chosen,'table':table}; out['test']=metric(test[0],test[1],test[2],chosen['weight_candidate']); out['selection_policy']='choose weight on validation only; test is final evaluation'; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'validation':chosen,'test':out['test']},ensure_ascii=False))
if __name__=='__main__': main()
