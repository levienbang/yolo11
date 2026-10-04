#!/usr/bin/env python3
"""Catalog polygon-only Vietnam images, then prepare a visually reviewed set."""
import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vehicle_pipeline.vehicle_benchmark import CLASSES, image_digest


def load_polygons(path, names, shape):
    targets=[]
    h,w=shape
    for number,line in enumerate(path.read_text().splitlines(),1):
        if not line.strip(): continue
        values=np.array([float(v) for v in line.split()])
        if not np.all(np.isfinite(values)) or values[0]!=int(values[0]) or int(values[0]) not in names:
            raise ValueError(f'Invalid class: {path}:{number}')
        name=names[int(values[0])].lower()
        name={'motorbike':'motorcycle','motor':'motorcycle'}.get(name,name)
        if name=='bicycle': continue
        if name not in CLASSES: raise ValueError(f'Unsupported class {name!r}')
        if len(values)<7 or len(values)%2!=1:
            raise ValueError(f'Non-polygon vehicle annotation: {path}:{number}')
        points=values[1:].reshape(-1,2)
        if np.any(points<0) or np.any(points>1) or len(np.unique(points,axis=0))<3:
            raise ValueError(f'Invalid polygon: {path}:{number}')
        pixels=np.round(points*np.array([w,h])).astype(np.int32)
        pixels=np.clip(pixels,[0,0],[w-1,h-1])
        if cv2.contourArea(pixels)<1: raise ValueError(f'Zero-area polygon: {path}:{number}')
        targets.append(dict(class_id=CLASSES.index(name),points=points,pixels=pixels,
                            box=np.r_[points.min(axis=0),points.max(axis=0)]))
    return targets


def names_from_yaml(path):
    names=yaml.safe_load(path.read_text())['names']
    return dict(enumerate(names)) if isinstance(names,list) else {int(k):v for k,v in names.items()}


def catalog(source):
    names=names_from_yaml(source/'data.yaml')
    rows=[]; excluded=Counter(); seen=set()
    for split in ['valid','test','train']:
        for image in sorted((source/split/'images').glob('*')):
            if image.suffix.lower() not in {'.jpg','.jpeg','.png'}: continue
            label=source/split/'labels'/image.with_suffix('.txt').name
            im=cv2.imread(str(image))
            if im is None: raise ValueError(f'Cannot decode {image}')
            try: targets=load_polygons(label,names,im.shape[:2])
            except ValueError as e:
                excluded['box_or_invalid_polygon']+=1
                continue
            if not targets: excluded['no_target_vehicles']+=1; continue
            digest=image_digest(image)
            if digest in seen: excluded['duplicate_pixels']+=1; continue
            seen.add(digest)
            boxes=np.array([t['box'] for t in targets]); wh=boxes[:,2:]-boxes[:,:2]
            small=float(np.mean(wh[:,1]<.035))
            # Pair-box intersection is a review proxy, never ground-truth occlusion.
            overlap=[]
            for i,b in enumerate(boxes):
                other=np.delete(boxes,i,axis=0)
                inter=np.maximum(np.minimum(b[2:],other[:,2:])-np.maximum(b[:2],other[:,:2]),0).prod(axis=1)
                overlap.append(float(np.max(inter/max(float(wh[i].prod()),1e-9))) if len(other) else 0)
            border=float(np.mean((boxes[:,0]<.003)|(boxes[:,2]>.997)|(boxes[:,1]<.003)|(boxes[:,3]>.997)))
            score=2*small+float(np.mean(overlap))+border
            counts=np.bincount([t['class_id'] for t in targets],minlength=4)
            match=re.match(r'(.*?)[_-](\d+)',image.name)
            sequence=match[1] if match else image.stem
            frame=int(match[2]) if match else 0
            rows.append(dict(filename=image.name,source_split=split,image=str(image.resolve()),label=str(label.resolve()),
                             sha256=digest,sequence=sequence,frame=frame,objects=len(targets),
                             **dict(zip(CLASSES,map(int,counts))),small_ratio=small,box_overlap_proxy=float(np.mean(overlap)),
                             border_ratio=border,review_score=score))
    return rows,dict(excluded),names


def suggest(rows):
    selected=[]; used=set(); totals=np.zeros(4,int)
    # Prefer complete polygons and class coverage; no penalty for object count.
    for level,count in [('easy',70),('medium',20),('hard',10)]:
        for _ in range(count):
            pool=[r for r in rows if r['filename'] not in used]
            def score(r):
                counts=np.array([r[n] for n in CLASSES])
                support=float(np.minimum(counts,3) @ (np.maximum(np.array([150,80,25,150])-totals,0)/np.array([150,80,25,150])))*.3
                nearby=sum(s['sequence']==r['sequence'] and abs(s['frame']-r['frame'])<5 for s in selected)
                quality=(-r['review_score'] if level=='easy' else -abs(r['review_score']-(.65 if level=='medium' else 1.1)))
                return quality+support-.7*nearby+.025*(r['source_split']!='train')
            chosen=max(pool,key=score).copy(); chosen['difficulty']=level
            chosen['review_notes']='PROVISIONAL: must visually inspect contact sheet'
            selected.append(chosen); used.add(chosen['filename']);totals+=np.array([chosen[n] for n in CLASSES])
    return selected


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def sheets(rows,names,out):
    out.mkdir(parents=True,exist_ok=True)
    colors=[(255,180,0),(0,140,255),(255,0,255),(0,220,70)]
    for start in range(0,len(rows),12):
        canvas=np.full((4*300,3*400,3),245,np.uint8)
        for pos,row in enumerate(rows[start:start+12]):
            image=cv2.imread(row['image']); targets=load_polygons(Path(row['label']),names,image.shape[:2])
            for t in targets: cv2.polylines(image,[t['pixels'].reshape(-1,1,2)],True,colors[t['class_id']],1)
            nonblack=np.any(image>15,axis=2); y,x=np.where(nonblack)
            crop=image[max(0,y.min()-2):min(image.shape[0],y.max()+3),max(0,x.min()-2):min(image.shape[1],x.max()+3)]
            scale=min(400/crop.shape[1],225/crop.shape[0]); tile=cv2.resize(crop,(int(crop.shape[1]*scale),int(crop.shape[0]*scale)))
            top,left=(pos//3)*300,(pos%3)*400
            canvas[top+70:top+70+tile.shape[0],left:left+tile.shape[1]]=tile
            texts=[f'{start+pos:03d} {row.get("difficulty", "candidate")} {row["source_split"]}',
                   row['filename'].split('_jpg')[0],
                   ' '.join(f'{n}:{row[n]}' for n in CLASSES)]
            for i,text in enumerate(texts):cv2.putText(canvas,text,(left+5,top+18+i*20),cv2.FONT_HERSHEY_SIMPLEX,.42,(20,20,20),1)
        cv2.imwrite(str(out/f'sheet_{start//12:02d}.jpg'),canvas)


def validate(dataset):
    with (dataset/'manifest.csv').open() as f: rows=list(csv.DictReader(f))
    if len(rows)!=100 or Counter(r['difficulty'] for r in rows)['easy']!=70:
        raise ValueError('Require exactly 100 images, 70 visually reviewed easy')
    names=names_from_yaml(dataset/'dataset.yaml'); seen=set();counts=Counter()
    if set(p.name for p in (dataset/'images').iterdir())!={r['filename'] for r in rows}:raise ValueError('Image inventory mismatch')
    if set(p.name for p in (dataset/'labels').iterdir())!={Path(r['filename']).with_suffix('.txt').name for r in rows}:raise ValueError('Label inventory mismatch')
    for row in rows:
        image=dataset/row['image_path']; digest=image_digest(image)
        if digest in seen or digest!=row['sha256']:raise ValueError('Duplicate or changed image')
        seen.add(digest); im=cv2.imread(str(image)); targets=load_polygons(dataset/row['label_path'],names,im.shape[:2])
        if len(targets)!=int(row['vehicle_instances']):raise ValueError('Instance count changed')
        c=Counter(t['class_id'] for t in targets)
        if any(c[i]!=int(row[n]) for i,n in enumerate(CLASSES)):raise ValueError('Class counts changed')
        counts.update(CLASSES[t['class_id']] for t in targets)
    return dict(images=len(rows),difficulty=dict(Counter(r['difficulty'] for r in rows)),
                source_splits=dict(Counter(r['source_split'] for r in rows)),class_instances=dict(counts))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--review-dir',type=Path,default=ROOT/'outputs/selection/vietnam-seg-100')
    p.add_argument('--review-csv',type=Path)
    p.add_argument('--output-dir',type=Path,default=ROOT/'datasets/vietnam-seg-review-100')
    args=p.parse_args(); names=names_from_yaml(args.source/'data.yaml')
    if not args.review_csv:
        rows,excluded,names=catalog(args.source)
        args.review_dir.mkdir(parents=True,exist_ok=True)
        write_csv(args.review_dir/'catalog.csv',rows)
        selected=suggest(rows);write_csv(args.review_dir/'provisional.csv',selected)
        sheets(selected,names,args.review_dir/'provisional_sheets')
        (args.review_dir/'catalog_summary.json').write_text(json.dumps(dict(candidates=len(rows),excluded=excluded),indent=2))
        print(json.dumps(dict(candidates=len(rows),excluded=excluded,provisional_instances=dict(zip(CLASSES,map(int,sum((np.array([r[n] for n in CLASSES]) for r in selected),np.zeros(4,int)))))),indent=2));return
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    with args.review_csv.open() as f: reviewed=list(csv.DictReader(f))
    if len(reviewed)!=100 or sum(r['difficulty']=='easy' for r in reviewed)!=70:raise ValueError('Need 100 reviewed images, 70 easy')
    if any(not r['review_notes'] or 'PROVISIONAL' in r['review_notes'] for r in reviewed):raise ValueError('Visual review required')
    (args.output_dir/'images').mkdir(parents=True);(args.output_dir/'labels').mkdir()
    manifest=[]
    for row in reviewed:
        image=Path(row['image']); im=cv2.imread(str(image));targets=load_polygons(Path(row['label']),names,im.shape[:2])
        filename=image.name;label_name=image.with_suffix('.txt').name
        shutil.copy2(image,args.output_dir/'images'/filename)
        lines=[str(t['class_id'])+' '+' '.join(f'{v:.10f}' for v in t['points'].ravel()) for t in targets]
        (args.output_dir/'labels'/label_name).write_text('\n'.join(lines)+'\n')
        manifest.append(dict(filename=filename,difficulty=row['difficulty'],source_dataset='vietnam_traffic',source_split=row['source_split'],
            original_image_path=row['image'],original_label_path=row['label'],image_path='images/'+filename,label_path='labels/'+label_name,
            sha256=image_digest(image),vehicle_instances=len(targets),**{n:row[n] for n in CLASSES},review_notes=row['review_notes']))
    write_csv(args.output_dir/'manifest.csv',manifest)
    (args.output_dir/'dataset.yaml').write_text(yaml.safe_dump(dict(path=str(args.output_dir.resolve()),val='images',nc=4,names=CLASSES)))
    summary=validate(args.output_dir)
    summary.update(source=str(args.source.resolve()),source_url=yaml.safe_load((args.source/'data.yaml').read_text()).get('roboflow',{}).get('url'),
                   selection='70 visually reviewed little-occlusion easy; remaining 30 reviewer discretion; no model predictions used',
                   leakage='Exact duplicates excluded within source. Friend fine-tuning dataset unavailable: train/eval leakage unverified.')
    (args.output_dir/'selection.json').write_text(json.dumps(summary,indent=2))
    sheet_rows=[dict(r,image=str(args.output_dir/r['image_path']),label=str(args.output_dir/r['label_path'])) for r in manifest]
    sheets(sheet_rows,dict(enumerate(CLASSES)),args.output_dir/'contact_sheets')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
