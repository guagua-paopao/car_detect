#!/usr/bin/env python3
"""Audit color confidence on unlabeled real crops; never treats them as ground truth."""
from __future__ import annotations
import argparse,csv,json,sys
from collections import Counter,defaultdict
from pathlib import Path
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--labels',type=Path,required=True); ap.add_argument('--candidate',type=Path,required=True); ap.add_argument('--baseline',type=Path,required=True); ap.add_argument('--candidate-calibration',type=Path,required=True); ap.add_argument('--baseline-calibration',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--threshold',type=float,default=.8); ap.add_argument('--device',default='cuda'); args=ap.parse_args()
 import torch
 from PIL import Image
 from torchvision import transforms
 from torch.utils.data import Dataset,DataLoader
 sys.path.insert(0,str(Path(__file__).resolve().parents[1])); from src.multitask_mobilenet_v3 import model_from_checkpoint,IMAGENET_MEAN,IMAGENET_STD
 with args.manifest.open('r',encoding='utf-8-sig',newline='') as f: rows=[r for r in csv.DictReader(f) if r.get('split') in {'validation','test'} and r.get('source_dataset')=='BMD-45-RAW-VAL' and str(r.get('color_supervised','false')).lower() in {'false','0','no'}]
 def load(p):
  ck=torch.load(p,map_location='cpu'); m=model_from_checkpoint(ck,pretrained=False); m.load_state_dict(ck['model_state']); m.eval(); size=int(ck.get('input_size',224)); tf=transforms.Compose([transforms.Resize((size,size),antialias=True),transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)]); return ck,m,tf
 bck,bm,bt=load(args.baseline); cck,cm,ct=load(args.candidate); btemp=float(json.loads(args.baseline_calibration.read_text())['color']['temperature']); ctemp=float(json.loads(args.candidate_calibration.read_text())['color']['temperature']); device=torch.device(args.device if args.device!='cuda' or torch.cuda.is_available() else 'cpu'); bm.to(device); cm.to(device)
 class Rows(Dataset):
  def __len__(self): return len(rows)
  def __getitem__(self,i):
   r=rows[i]; p=Path(r['image_path']); p=p if p.is_absolute() else (args.manifest.parent/p).resolve()
   with Image.open(p) as im: im=im.convert('RGB')
   return bt(im),ct(im),i
 loader=DataLoader(Rows(),batch_size=64,shuffle=False,num_workers=4); stats={'candidate':{'n':0,'abstained':0,'unknown_predicted':0,'confidence_sum':0.0,'by_lighting':defaultdict(lambda:[0,0,0.0])},'baseline':{'n':0,'abstained':0,'unknown_predicted':0,'confidence_sum':0.0,'by_lighting':defaultdict(lambda:[0,0,0.0])}}
 unknown_c=9
 with torch.no_grad():
  for bi,ci,indexes in loader:
   bl,_=bm(bi.to(device)); _,cl=cm(ci.to(device));
   for name,logits,temp in (('candidate',cl,ctemp),('baseline',bm(ci.to(device))[1] if False else None,btemp)):
    if name=='baseline': _,logits=bm(bi.to(device))
    probs=(logits/temp).softmax(1); conf,pred=probs.max(1); non=probs.clone(); non[:,unknown_c]=-1; nconf,npred=non.max(1)
    for confv,predv,nconfv,idx in zip(conf.tolist(),pred.tolist(),nconf.tolist(),indexes.tolist()):
     light=rows[idx].get('lighting') or 'unknown'; z=stats[name]; z['n']+=1; z['confidence_sum']+=nconfv; z['abstained']+=nconfv<args.threshold; z['unknown_predicted']+=predv==unknown_c; z['by_lighting'][light][0]+=1; z['by_lighting'][light][1]+=nconfv<args.threshold; z['by_lighting'][light][2]+=nconfv
 def finish(z):
  z['abstention_rate']=z['abstained']/max(1,z['n']); z['mean_nonunknown_confidence']=z['confidence_sum']/max(1,z['n']); z['by_lighting']={k:{'n':v[0],'abstention_rate':v[1]/max(1,v[0]),'mean_nonunknown_confidence':v[2]/max(1,v[0])} for k,v in sorted(z['by_lighting'].items())}; return z
 out={'schema_version':'unlabeled-color-abstention-audit-v1','policy':'proxy only; unlabeled BMD-45 raw validation crops are not treated as color ground truth and cannot satisfy the unknown-rate release gate','manifest':str(args.manifest),'rows':len(rows),'threshold_from_labeled_validation':args.threshold,'candidate':finish(stats['candidate']),'baseline':finish(stats['baseline']),'frozen_video_used':False}; args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'rows':len(rows),'candidate_abstention_rate':out['candidate']['abstention_rate'],'baseline_abstention_rate':out['baseline']['abstention_rate'],'proxy_only':True},ensure_ascii=False))
if __name__=='__main__': main()
