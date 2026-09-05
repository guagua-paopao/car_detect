#!/usr/bin/env python3
"""Build conservative, track-consistent, class-capped train-only color pseudo labels."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image,ImageEnhance
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.multitask_mobilenet_v3 import model_from_checkpoint,IMAGENET_MEAN,IMAGENET_STD

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--candidate',type=Path,required=True); ap.add_argument('--baseline',type=Path,required=True); ap.add_argument('--candidate-temperature',type=float,default=.825); ap.add_argument('--baseline-temperature',type=float,default=.75); ap.add_argument('--confidence',type=float,default=.88); ap.add_argument('--min-track-frames',type=int,default=3); ap.add_argument('--min-track-agree',type=float,default=.8); ap.add_argument('--max-per-class',type=int,default=12000); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--device',default='cuda'); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--report',type=Path,required=True); a=ap.parse_args()
 with a.manifest.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 targets=[(i,r) for i,r in enumerate(rows) if r.get('split')=='train' and r.get('source_dataset') in {'BMD-45-RAW','BMD-45-RAW-COCO'} and r.get('color_supervised','false').lower() in {'false','0','no'}]
 if not targets: raise RuntimeError('no unlabeled BMD raw train rows')
 cck=torch.load(a.candidate,map_location='cpu'); bck=torch.load(a.baseline,map_location='cpu'); cm=model_from_checkpoint(cck,pretrained=False); bm=model_from_checkpoint(bck,pretrained=False); cm.load_state_dict(cck['model_state']); bm.load_state_dict(bck['model_state']); cm.eval(); bm.eval(); dev=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu'); cm.to(dev); bm.to(dev)
 size=int(cck.get('input_size',224)); norm=transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD); resize=transforms.Resize((size,size),antialias=True); to_tensor=transforms.ToTensor()
 class D(Dataset):
  def __len__(self): return len(targets)
  def __getitem__(self,j):
   idx,r=targets[j]; p=(a.manifest.parent/r['image_path']).resolve()
   with Image.open(p) as im: im=im.convert('RGB')
   xs=[]
   for bright,contrast in [(1.,1.),(.78,1.08),(1.22,.92)]:
    x=ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(bright)).enhance(contrast); xs.append(norm(to_tensor(resize(x))))
   return torch.stack(xs),j
 loader=DataLoader(D(),batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=dev.type=='cuda'); predictions={}; stats=Counter()
 with torch.no_grad():
  for batch,js in loader:
   bp=[]; cp=[]; x=batch.to(dev)
   for k in range(3):
    _,bl=bm(x[:,k]); _,cl=cm(x[:,k]); bp.append((bl/a.baseline_temperature).softmax(1)); cp.append((cl/a.candidate_temperature).softmax(1))
   for n,j in enumerate(js.tolist()):
    blabel=[int(z[n].argmax()) for z in bp]; clabel=[int(z[n].argmax()) for z in cp]; bconf=[float(z[n].max()) for z in bp]; cconf=[float(z[n].max()) for z in cp]; label=clabel[0]; strong=(label==blabel[0] and label!=9 and len(set(blabel))==1 and len(set(clabel))==1 and min(bconf+cconf)>=a.confidence); stats['seen']+=1; stats['agree']+=int(label==blabel[0] and label!=9); stats['strong']+=int(strong); predictions[j]=(label,strong,min(bconf+cconf))
 groups=defaultdict(list)
 for j,(idx,r) in enumerate(targets): groups[(r.get('video_id',''),r.get('track_group',''))].append((j,idx,r))
 accepted_by_class=defaultdict(list); track_pass=0
 for key,items in groups.items():
  strong=[(j,idx,r,predictions[j][0]) for j,idx,r in items if predictions[j][1]]
  if len(items)<a.min_track_frames or len(strong)<a.min_track_frames: continue
  counts=Counter(x[3] for x in strong); label,n=counts.most_common(1)[0]
  if n/len(strong)<a.min_track_agree: continue
  track_pass+=1
  for j,idx,r,p in strong:
   if p==label: accepted_by_class[cck['colors'][p]].append((idx,j))
 accepted={}
 for label,items in accepted_by_class.items():
  for idx,j in sorted(items,key=lambda x:x[0])[:a.max_per_class]: accepted[idx]=label
 out_rows=[]
 for i,r in enumerate(rows):
  nr=dict(r)
  if i in accepted:
   nr['color']=accepted[i]; nr['color_supervised']='true'; nr['annotation_source']='auto_track_consensus_pseudo_v2'; nr['color_review_status']='auto_track_consensus'; nr['review_method']='teacher_candidate_augmented_track_consensus'; nr['pseudo_label']='true'; nr['pseudo_label_confidence']=str(a.confidence)
  out_rows.append(nr)
 a.output.parent.mkdir(parents=True,exist_ok=True); fields=list(rows[0]);
 for extra in ('pseudo_label','pseudo_label_confidence'):
  if extra not in fields: fields.append(extra)
 with a.output.open('w',encoding='utf-8',newline='') as f: w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(out_rows)
 report={'schema_version':'bmd-raw-color-balanced-track-pseudolabel-v2','input_manifest':str(a.manifest),'output_manifest':str(a.output),'input_sha256':sha(a.manifest),'output_sha256':sha(a.output),'policy':'train-only; teacher/candidate agreement, three photometric variants, track consensus, per-class cap; never validation/test ground truth','frozen_video_used':False,'test_or_validation_used':False,'confidence':a.confidence,'min_track_frames':a.min_track_frames,'min_track_agree':a.min_track_agree,'max_per_class':a.max_per_class,'rows':dict(stats),'candidate_tracks_passed':track_pass,'accepted_rows':len(accepted),'accepted_color_counts':dict(Counter(accepted.values())),'candidate_checkpoint':str(a.candidate),'baseline_checkpoint':str(a.baseline)}
 a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
