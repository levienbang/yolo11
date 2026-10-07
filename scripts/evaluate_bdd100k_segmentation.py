#!/usr/bin/env python3
"""Evaluate the same YOLO + orthogonal hull pipeline on the 100-image BDD100K sample.

Reuses evaluate_vietnam_segmentation.py unchanged except for GT loading:
BDD GT comes from per-instance bitmasks (some instances have several parts),
so masks are read directly instead of single YOLO polygons.

datasets/bdd100k-seg-100/ is self-contained: 100 images from the BDD100K 10K
instance-segmentation val split, one binary mask PNG per vehicle under masks/,
and one JSON label per image listing {class, mask}. Bicycle, ignored and crowd
instances were excluded when the sample was built.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT/'datasets/bdd100k-seg-100'
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_vietnam_segmentation as ev
from evaluation_metrics import region
from vehicle_pipeline.vehicle_benchmark import CLASSES


def load_gt(path, shape):
    # Mask paths in the label JSON are relative to the dataset root (labels/..).
    dataset = Path(path).resolve().parents[1]
    targets = []
    for item in json.loads(Path(path).read_text()):
        image = cv2.imread(str(dataset/item['mask']), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(dataset/item['mask'])
        mask = (image > 0).astype(np.uint8)
        if mask.shape != shape:
            raise ValueError('Mask/image size mismatch: '+item['mask'])
        ys, xs = np.nonzero(mask)
        targets.append({'class': CLASSES.index(item['class']), 'region': region(mask),
                        'box': [float(xs.min()), float(ys.min()), float(xs.max()+1), float(ys.max()+1)]})
    return targets, shape


def validate(dataset):
    with (dataset/'manifest.csv').open() as f:
        rows = list(csv.DictReader(f))
    counts = Counter()
    for row in rows:
        counts.update({c: int(row[c]) for c in CLASSES})
    return dict(images=len(rows), source_splits=dict(Counter(r['source_split'] for r in rows)),
                class_instances=dict(counts))


def report_markdown(path, summary, example_paths):
    lines = ['# Đánh giá 100 ảnh BDD100K (val) — YOLO và bao lồi trực giao', '',
             f"Ảnh: {summary['images']}; nhãn xe: {summary['gt_instances']} "
             f"({', '.join(f'{k} {v}' for k, v in summary['validation']['class_instances'].items())}).",
             'GT là bitmask từng đối tượng của BDD100K ins_seg; bỏ bicycle. Không chia mức độ khó.',
             f"Model: `{summary['model']}`; dữ liệu train theo metadata: `{summary['model_training_data']}`.", '',
             '| Đầu ra | Precision | Recall | F1 | mAP50 | mAP50–95 | IoU TP |', '|---|---:|---:|---:|---:|---:|---:|']
    for method, m in summary['overall'].items():
        vals = [m[k] for k in ['precision', 'recall', 'f1', 'map50', 'map50_95', 'mean_matched_iou']]
        lines.append('| '+method+' | '+' | '.join(f'{v:.1%}' if v is not None else '—' for v in vals)+' |')
    lines += ['', '| Thuật toán | ms/hull | IoU với YOLO | Pixel thêm | Pixel bỏ |', '|---|---:|---:|---:|---:|']
    for method, m in summary['hull_fidelity'].items():
        lines.append(f"| {method} | {m['mean_ms']:.3f} | {m['mean_iou_with_yolo']:.1%} | "
                     f"{m['added_pixel_ratio']:.1%} | {m['removed_pixel_ratio']:.1%} |")
    lines += ['', 'Motorcycle chỉ có rất ít mẫu: AP lớp này không đáng tin. mAP là trung bình các lớp có GT; bộ tính local, không phải COCOeval.']
    for key, ref in example_paths.items():
        lines += ['', f'**{key}**', '', f'![{key}]({ref})']
    path.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    ev.load_gt, ev.validate, ev.report_markdown = load_gt, validate, report_markdown
    if '--dataset' not in sys.argv:
        sys.argv += ['--dataset', str(DATASET)]
    ev.main(default_output=ROOT/'outputs/evaluation/bdd100k-seg-100')
