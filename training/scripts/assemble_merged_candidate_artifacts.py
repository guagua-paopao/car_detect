#!/usr/bin/env python3
"""Write an auditable candidate model card and SHA256 inventory without cutover."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--output-card',type=Path,required=True); ap.add_argument('--output-sha',type=Path,required=True); a=ap.parse_args()
 root=a.root; names=['merged-stage23base-input224.pt','merged-input224-calibrated.onnx','merged-input224.engine','merged-calibration.json','merged-input224-combined.json','merged-input224-candidate-report.json','merged-input224-onnx-parity.json','merged-input224-tensorrt-smoke.json']
 files=[]
 for n in names:
  p=root/n
  if p.exists(): files.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
 card={'schema_version':'vehicle-attribute-candidate-model-card-v1','candidate_id':'vehicle-attr-v28-merged-input224','status':'candidate_only_not_deployed','production_baseline':'vehicle-attr-agent-e-224','production_model_unchanged':True,'deployment_performed':False,'input':{'width':224,'height':224,'format':'RGB uint8 adapter input; embedded ImageNet normalization'},'outputs':{'body_type':['sedan','suv','mpv','van','pickup','bus','light_truck','heavy_truck','other','unknown'],'color':['black','white','silver_gray','red','blue','green','yellow_orange','brown_beige','other','unknown']},'training_policy':{'body_source':'BMD-45 raw detection crops with independent image-group validation/test','color_source':'licensed VCoR-LARGE plus existing approved color labels','frozen_video_used':False,'test_used_for_selection':False},'metrics_reference':str(root/'merged-input224-candidate-report.json'),'backend_reference':{'onnx_parity':str(root/'merged-input224-onnx-parity.json'),'tensorrt_smoke':str(root/'merged-input224-tensorrt-smoke.json')},'rollback':'retain vehicle-attr-agent-e-224; no registry/config cutover','artifacts':files}
 a.output_card.parent.mkdir(parents=True,exist_ok=True); a.output_card.write_text(json.dumps(card,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); a.output_sha.write_text(json.dumps({'schema_version':'candidate-sha256-inventory-v1','candidate_id':card['candidate_id'],'files':files},ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'candidate_id':card['candidate_id'],'files':len(files),'status':card['status']},ensure_ascii=False))
if __name__=='__main__': main()
