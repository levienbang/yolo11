#!/usr/bin/env python3
"""Evaluate reviewed Vietnam polygon GT with the existing YOLO orthogonal hulls."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vehicle_pipeline.device import resolve_device
from vehicle_pipeline.hull_common import compute_hull
from vehicle_pipeline.vehicle_benchmark import CLASSES, confusion, summarize_confusion
from prepare_vietnam_segmentation import load_polygons, validate
from evaluation_metrics import region, region_iou, box_iou, matching, summarize, write_csv, full_mask, matrix_image
METHODS=['box_detection','bbox_region','convex','yolo_mask','quickhull','ograham']


def load_gt(path, shape):
    polygons=load_polygons(path,dict(enumerate(CLASSES)),shape)
    targets=[]
    h,w=shape
    for polygon in polygons:
        mask=np.zeros(shape,np.uint8)
        cv2.fillPoly(mask,[polygon['pixels'].reshape(-1,1,2)],1)
        targets.append({'class':polygon['class_id'],'region':region(mask),
                        'box':(polygon['box']*np.array([w,h,w,h])).tolist()})
    return targets,shape


def example(image, gt, predictions, record, conf):
    panels=[]
    for method,title in [(None,'Vietnam polygon GT'),('yolo_mask','YOLO mask'),('quickhull','YOLO + QuickHull'),('ograham','YOLO + O-Graham')]:
        canvas=image.copy()
        if method is None:
            entries=[(g['region'],(255,170,50),CLASSES[g['class']]) for g in gt];missed=[]
        else:
            candidates,matches,missed=matching(record,method,conf)
            entries=[(predictions[i]['regions'][method],(50,210,50) if i in matches else (40,40,230),
                      CLASSES[predictions[i]['class']]+f" {predictions[i]['score']:.2f}") for i in candidates]
        for cropped,color,label in entries:
            mask=full_mask(cropped,image.shape[:2]);selected=mask>0
            canvas[selected]=(canvas[selected]*.8+np.array(color)*.2).astype(np.uint8)
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas,contours,-1,color,2)
            x,y,pixels,_=cropped
            if pixels.size:cv2.putText(canvas,label,(x,max(14,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
        for j in missed:
            contours,_=cv2.findContours(full_mask(gt[j]['region'],image.shape[:2]),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas,contours,-1,(0,165,255),2)
        header=np.full((56,image.shape[1],3),245,np.uint8)
        cv2.putText(header,title,(12,23),cv2.FONT_HERSHEY_SIMPLEX,.65,(20,20,20),1)
        cv2.putText(header,'green=TP red=FP orange=missed | class + IoU >= .50',(12,46),cv2.FONT_HERSHEY_SIMPLEX,.43,(40,40,40),1)
        panels.append(cv2.resize(np.vstack([header,canvas]),(640,696)))
    return np.hstack(panels)


def report_markdown(path, summary, example_paths):
    lines=['# Đánh giá 100 ảnh Việt Nam — YOLO và bao lồi trực giao','',
           f"Ảnh: {summary['images']}; nhãn xe: {summary['gt_instances']}. Chọn 70 dễ (ít che khuất), 20 trung bình, 10 khó bằng xem ảnh.",
           'Không dùng số lượng xe để định nghĩa dễ. Tất cả GT xe là polygon; loại toàn bộ ảnh có nhãn xe chỉ là box.',
           '',f"Model: `{summary['model']}`.",f"Metadata training dataset: `{summary['model_training_data']}`.",
           'Mô hình dùng trực tiếp, không fine-tune trên dữ liệu Việt Nam; chỉ giữ 4 lớp phương tiện khi suy luận.',
           'Không retrain. Đánh giá ảnh trước tracking: chưa đánh giá ID, đếm xe hoặc tốc độ Jetson.',
           '', '## Kết quả', '',
           '| Đầu ra | Precision | Recall | F1 | mAP50 | mAP50–95 | IoU TP | Dice TP |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method,m in summary['overall'].items():
        vals=[m[k] for k in ['precision','recall','f1','map50','map50_95','mean_matched_iou','mean_matched_dice']]
        lines.append('| '+method+' | '+' | '.join(f'{v:.1%}' if v is not None else '—' for v in vals)+' |')
    lines += ['', 'box_detection so box với box GT. bbox_region tô kín box so mask GT, chỉ là baseline vùng pixel.',
              'Mask YOLO/QuickHull/O-Graham đều so cùng polygon GT raster hóa tại kích thước ảnh gốc.',
              'mAP dùng matching đúng lớp, confidence-ordered, IoU 0.50:0.05:0.95, 101 điểm recall; không phải COCOeval chính thức.',
              'IoU/Dice TP chỉ tính cặp đúng lớp có IoU ≥ 0.5: cần đọc cùng recall để thấy xe bỏ sót.',
              '', '## Nhầm lớp', '',
              'Confusion rows=GT, columns=prediction; ghép không xét lớp theo IoU box/mask, ngưỡng 0.5.',
              'Background column=GT bỏ sót; background row=prediction không ghép được.',
              '| Đầu ra | Car → truck | Truck → car | Accuracy trong cặp đã ghép |',
              '|---|---:|---:|---:|']
    for method,m in summary['confusion'].items():
        a=m['confusion']['car_to_truck']['count'];b=m['confusion']['truck_to_car']['count'];acc=m['matched_classification_accuracy']
        lines.append(f"| {method} | {a} | {b} | {acc:.1%} |" if acc is not None else f'| {method} | {a} | {b} | — |')
    lines += ['', '## Bao lồi và mask YOLO', '', '| Thuật toán | ms/hull | IoU với YOLO | Pixel thêm | Pixel bỏ |', '|---|---:|---:|---:|---:|']
    for method,m in summary['hull_fidelity'].items():
        lines.append(f"| {method} | {m['mean_ms']:.3f} | {m['mean_iou_with_yolo']:.1%} | {m['added_pixel_ratio']:.1%} | {m['removed_pixel_ratio']:.1%} |")
    lines += ['', '## Giới hạn và file', '',
              'Bộ chọn có chủ đích, không đại diện ngẫu nhiên cho giao thông Việt Nam. Dễ là nhận xét trực quan tương đối.',
              'Split nguồn và số lớp xem selection.json. Mô hình không được huấn luyện trên bộ nguồn nên ảnh thuộc split train không gây rò rỉ dữ liệu.',
              'Không tự đổi nhãn car/truck của nguồn, không suy đoán pickup/van. Bỏ bicycle ngoài phạm vi.',
              'Ảnh cùng video có tương quan dù không trùng pixel; không coi đây là 100 cảnh độc lập.',
              'AP bị giới hạn bởi confidence floor, NMS và max_det. Một số mask có thể thiếu/không sát vật thể ngay trong nguồn.',
              '`class_metrics.csv`, `difficulty_metrics.csv`, `per_image.csv`, `metrics.json`, `prediction_records.json` chứa số liệu.',
              '`comparisons/` có đủ 100 ảnh so GT / YOLO / QuickHull / O-Graham; `errors/` chứa ảnh có nhầm lớp qua ghép box.',
              '', '## Ví dụ', '']
    for key,value in example_paths.items():
        # Existing convention maps key -> {'path': ..., 'filename': ...}.
        ref=value['path'] if isinstance(value,dict) else value
        lines += [f'**{key}**', '',f'![{key}]({ref})','']
    path.write_text('\n'.join(lines),encoding='utf-8')


def main(default_model=None, default_output=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'datasets/vietnam-seg-review-100')
    parser.add_argument('--model', type=Path, default=default_model or ROOT/'weights/yolo-nano/yolo11n-seg.pt')
    parser.add_argument('--output-dir', type=Path, default=default_output or ROOT/'outputs/evaluation/vietnam-seg-100-nano')
    parser.add_argument('--device', default='auto', help='auto, cpu, mps or a CUDA index')
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--conf', type=float, default=0.20)
    parser.add_argument('--conf-floor', type=float, default=0.001)
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    if not 0 < args.conf_floor <= args.conf <= 1:
        parser.error('Require 0 < conf-floor <= conf <= 1')
    if args.output_dir.exists():
        raise FileExistsError('Output exists; choose another --output-dir: '+str(args.output_dir))
    with (args.dataset/'manifest.csv').open() as f:
        manifest = list(csv.DictReader(f))
    if not manifest or len({r['filename'] for r in manifest}) != len(manifest):
        raise ValueError('Manifest must contain unique images')
    model = YOLO(str(args.model))
    if model.task != 'segment':
        raise ValueError('This comparison requires a segmentation model')
    normalized_names = {i: {'motorbike':'motorcycle','motor':'motorcycle'}.get(name.lower(), name.lower()) for i,name in model.names.items()}
    classes = [i for i, name in normalized_names.items() if name in CLASSES]
    if len(classes) != len(CLASSES):
        raise ValueError('Model must contain car/truck/bus/motorcycle classes')
    # Validate every pair before inference or creating outputs.
    for row in manifest:
        image_path, label_path = args.dataset/row['image_path'], args.dataset/row['label_path']
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(str(image_path)+' / '+str(label_path))
    validation = validate(args.dataset)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'comparisons').mkdir()
    (args.output_dir/'errors').mkdir()
    records, per_image, examples = [], [], {}
    timings = Counter()
    fidelity = {method: Counter() for method in ['quickhull', 'ograham']}
    speeds = Counter()
    start = perf_counter()
    for number, row in enumerate(manifest, 1):
        image = cv2.imread(str(args.dataset/row['image_path']))
        if image is None:
            raise ValueError('Cannot decode '+row['image_path'])
        targets, shape = load_gt(args.dataset/row['label_path'], image.shape[:2])
        if shape != image.shape[:2]:
            raise ValueError('Image and mask dimensions differ')
        if len(targets) != int(row['vehicle_instances']):
            raise ValueError('GT count differs from manifest: '+row['filename'])
        result = model.predict(image, classes=classes, conf=args.conf_floor, iou=.7,
                               imgsz=args.imgsz, max_det=300, device=args.device,
                               retina_masks=True, verbose=False)[0]
        for stage, milliseconds in result.speed.items():
            speeds[stage] += milliseconds
        boxes = result.boxes.data.cpu().numpy()
        if len(boxes) and result.masks is None:
            raise RuntimeError('Detections without masks: '+row['filename'])
        masks = result.masks.data.cpu().numpy() > .5 if len(boxes) else []
        polygons = result.masks.xy if len(boxes) else []
        if len(boxes) and (len(masks) != len(boxes) or masks.shape[1:] != shape):
            raise ValueError('Mask alignment error')
        predictions = []
        for index, (box, mask, points) in enumerate(zip(boxes, masks, polygons)):
            cls = CLASSES.index(normalized_names[int(box[-1])])
            score = float(box[-2])
            x1, y1 = np.floor(box[:2]).astype(int)
            x2, y2 = np.ceil(box[2:4]).astype(int)
            x1, x2 = np.clip([x1, x2], 0, shape[1])
            y1, y2 = np.clip([y1, y2], 0, shape[0])
            rectangle = np.ones((max(0, y2-y1), max(0, x2-x1)), dtype=bool)
            regions = {'yolo_mask': region(mask), 'bbox_region': (x1, y1, rectangle, int(rectangle.sum()))}
            filled = np.zeros(shape, np.uint8)
            if len(points):
                hull = cv2.convexHull(np.asarray(points, np.int32))
                cv2.fillPoly(filled, [hull], 1)
            regions['convex'] = region(filled)
            # Alternate order across objects to avoid always timing one algorithm first.
            order = ['quickhull','ograham'] if (number+index)%2 else ['ograham','quickhull']
            for method in order:
                before = perf_counter()
                hull = compute_hull(points, method)
                elapsed = (perf_counter()-before)*1000
                filled = np.zeros(shape, np.uint8)
                if len(hull) >= 3:
                    cv2.fillPoly(filled, [hull.reshape(-1,1,2)], 1)
                regions[method] = region(filled)
                if score >= args.conf:
                    f = fidelity[method]
                    f['calls'] += 1
                    f['ms'] += elapsed
                    f['iou_sum'] += region_iou(regions['yolo_mask'], regions[method])
                    f['source_pixels'] += int(mask.sum())
                    f['added_pixels'] += int(((filled > 0) & ~mask).sum())
                    f['removed_pixels'] += int((mask & (filled == 0)).sum())
            predictions.append({'class': cls, 'score': score, 'box': box[:4].tolist(), 'regions': regions})
        matrices = {method: np.zeros((len(predictions),len(targets)), np.float32) for method in METHODS}
        for i, pred in enumerate(predictions):
            for j, target in enumerate(targets):
                matrices['box_detection'][i,j] = box_iou(pred['box'], target['box'])
                for method in METHODS[1:]:
                    matrices[method][i,j] = region_iou(pred['regions'][method], target['region'])
        record = {'filename': row['filename'], 'difficulty': row['difficulty'],
                  'scores': [p['score'] for p in predictions], 'pred_classes': [p['class'] for p in predictions],
                  'gt_classes': [g['class'] for g in targets], 'ious': matrices}
        records.append(record)
        record['source_split'] = row['source_split']
        visual = example(image, targets, predictions, record, args.conf)
        cv2.imwrite(str(args.output_dir/'comparisons'/Path(row['filename']).with_suffix('.jpg').name), visual)
        confusion_mat, confusion_matches = confusion(record, args.conf, .5)
        mistakes = [(i,j) for i,j in confusion_matches.items() if record['pred_classes'][i] != record['gt_classes'][j]]
        if mistakes:
            cv2.imwrite(str(args.output_dir/'errors'/Path(row['filename']).with_suffix('.jpg').name), visual)
        for method in METHODS:
            per_image.append({'filename':row['filename'], 'difficulty':row['difficulty'], 'method':method,
                              **summarize([record], method, args.conf, include_ap=False)})
        current = summarize([record], 'yolo_mask', args.conf, include_ap=False)
        quality = (current['f1'], -current['fn'], -current['fp'])
        for kind in ['best','worst']:
            key = row['difficulty']+'_'+kind
            previous = examples.get(key)
            if previous is None or (quality > previous['quality'] if kind=='best' else quality < previous['quality']):
                examples[key] = {'quality':quality, 'filename':row['filename'],
                                 'image':example(image, targets, predictions, record, args.conf)}
        if number == 1 or number % 10 == 0:
            print(f'{number}/{len(manifest)} images; {perf_counter()-start:.1f}s', flush=True)
    overall = {method:summarize(records, method, args.conf) for method in METHODS}
    for method in METHODS:
        overlaps = []
        for record in records:
            _, matches, _ = matching(record, method, args.conf)
            overlaps.extend(float(record['ious'][method][i,j]) for i,j in matches.items())
        overall[method]['mean_matched_dice'] = float(np.mean([2*v/(1+v) for v in overlaps])) if overlaps else None
    confusion_metrics = {}
    for method in ['box_detection','yolo_mask','quickhull','ograham']:
        cm = np.zeros((5,5),dtype=int)
        for record in records:
            local = dict(record, ious={'box_detection':record['ious'][method]})
            cm += confusion(local,args.conf,.5)[0]
        confusion_metrics[method] = summarize_confusion(cm)
        matrix_image(cm,args.output_dir/f'{method}_confusion_matrix.png')
        matrix_image(cm,args.output_dir/f'{method}_confusion_matrix_normalized.png',True)
    by_class = [{'method':method, 'class':name, **summarize(records,method,args.conf,c)}
                for method in METHODS for c,name in enumerate(CLASSES)]
    by_difficulty = [{'method':method, 'difficulty':level,
                      **summarize([r for r in records if r['difficulty']==level],method,args.conf)}
                     for method in METHODS for level in ['easy','medium','hard']]
    thresholds = sorted(set([.05,.10,.20,.30,.50,.70,args.conf]))
    curve = [{'conf':threshold, **summarize(records,'yolo_mask',threshold,include_ap=False)}
             for threshold in thresholds if threshold >= args.conf_floor]
    hull_fidelity = {method:{'calls':int(f['calls']), 'mean_ms':f['ms']/max(f['calls'],1),
                            'mean_iou_with_yolo':f['iou_sum']/max(f['calls'],1),
                            'added_pixel_ratio':f['added_pixels']/max(f['source_pixels'],1),
                            'removed_pixel_ratio':f['removed_pixels']/max(f['source_pixels'],1)}
                     for method,f in fidelity.items()}
    summary = {'images':len(records), 'gt_instances':sum(len(r['gt_classes']) for r in records),
               'model':str(args.model.resolve()), 'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
               'dataset':str(args.dataset.resolve()), 'manifest_sha256':hashlib.sha256((args.dataset/'manifest.csv').read_bytes()).hexdigest(),
               'settings':{'conf':args.conf,'conf_floor':args.conf_floor,'imgsz':args.imgsz,'device':args.device,
                           'nms_iou':.7,'max_det':300,'retina_masks':True,'tracker':None},
               'metric_protocol':'Confidence-ordered class-aware one-to-one matching. 101-point interpolated AP, IoU .50:.05:.95, area all, max_det 300. Not official COCOeval.',
               'validation':validation, 'model_training_data':model.ckpt.get('train_args', {}).get('data') if model.ckpt.get('train_args') else None,
               'confusion':confusion_metrics, 'overall':overall, 'by_class':by_class, 'by_difficulty':by_difficulty,
               'confidence_curve':curve, 'hull_fidelity':hull_fidelity,
               'yolo_stage_mean_ms':{k:v/len(records) for k,v in speeds.items()},
               'images_at_max_det':sum(len(r['scores']) >= 300 for r in records),
               'wall_seconds':perf_counter()-start}
    write_csv(args.output_dir/'class_metrics.csv',by_class)
    write_csv(args.output_dir/'difficulty_metrics.csv',by_difficulty)
    write_csv(args.output_dir/'confidence_curve.csv',curve)
    write_csv(args.output_dir/'per_image.csv',per_image)
    (args.output_dir/'metrics.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    raw_records = [{**{k:v for k,v in r.items() if k!='ious'},'ious':{m:v.tolist() for m,v in r['ious'].items()}} for r in records]
    (args.output_dir/'prediction_records.json').write_text(json.dumps(raw_records))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    rows=[r for r in by_class if r['method']=='yolo_mask']
    x=np.arange(len(rows));width=.36
    ax.bar(x-width/2,[r['precision'] for r in rows],width,label='Precision')
    ax.bar(x+width/2,[r['recall'] for r in rows],width,label='Recall')
    ax.set_xticks(x,CLASSES);ax.set_ylim(0,1.05);ax.set_ylabel('Score')
    ax.set_title(f'YOLO mask | confidence {args.conf:.2f} | match IoU >= 0.50')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(args.output_dir/'class_metrics.png',dpi=160);plt.close(fig)
    (args.output_dir/'examples').mkdir()
    example_paths={}
    for key,value in examples.items():
        filename='examples/'+key+'.jpg'
        cv2.imwrite(str(args.output_dir/filename),value['image'],[cv2.IMWRITE_JPEG_QUALITY,93])
        example_paths[key+' — '+value['filename']]=filename
    report_markdown(args.output_dir/'report.md',summary,example_paths)
    print(json.dumps({'output':str(args.output_dir),'yolo_mask':overall['yolo_mask'],'quickhull':overall['quickhull']},indent=2),flush=True)


if __name__=='__main__':
    main()
