#!/usr/bin/env python3
"""Same-vehicle comparison of the YOLO mask and regions derived from it.

For confidence >= 0.20: compactness (contour points, hull vertices), mask topology
(fragments, holes) and IoU of each region with the GT paired to the YOLO mask, so
every region is scored on one fixed set of vehicles. Writes a JSON summary and, for
YOLO11-l, the close-up and scene figures used in the report.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'src'))
import evaluate_vietnam_segmentation as ev
import evaluate_bdd100k_segmentation as bdd
from evaluation_metrics import full_mask, matching, region, region_iou
from vehicle_pipeline.device import resolve_device
from vehicle_pipeline.hull_common import compute_hull
from vehicle_pipeline.vehicle_benchmark import CLASSES

MODELS = {'large': ROOT/'weights/yolo-large/yolo11l-seg.pt', 'nano': ROOT/'weights/yolo-nano/yolo11n-seg.pt'}
DATASETS = {'vietnam': (ROOT/'datasets/vietnam-seg-review-100', ev.load_gt),
            'bdd100k': (ROOT/'datasets/bdd100k-seg-100', bdd.load_gt)}
CONF = 0.20
MIN_CROP_AREA = 1600
COLORS = {'gt': '#2ca02c', 'rect': '#d62728', 'mask': '#7b4fb5', 'convex': '#1f77b4', 'hull': '#ff7f0e'}
TITLES = {'gt': 'Nhãn chuẩn', 'rect': 'Hình chữ nhật', 'mask': 'Mặt nạ YOLO',
          'convex': 'Bao lồi thường', 'hull': 'Bao trực giao'}


def turning_vertices(hull):
    """Vertices where the direction changes; collinear points are not counted."""
    p = np.asarray(hull).reshape(-1, 2)
    if len(p) > 1 and np.array_equal(p[0], p[-1]):
        p = p[:-1]
    if len(p) < 3:
        return len(p)
    before, after = p-np.roll(p, 1, axis=0), np.roll(p, -1, axis=0)-p
    return int(np.count_nonzero(before[:, 0]*after[:, 1]-before[:, 1]*after[:, 0]
                                + (np.sign(before) == -np.sign(after)).all(axis=1)*(before != 0).any(axis=1)))


def topology(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0, 0
    parents = hierarchy[0][:, 3]
    return int(np.sum(parents < 0)), int(np.sum(parents >= 0))


def filled(polygon, shape):
    canvas = np.zeros(shape, np.uint8)
    if len(polygon) >= 3:
        cv2.fillPoly(canvas, [np.asarray(polygon, np.int32).reshape(-1, 1, 2)], 1)
    return canvas


def predict(model, classes, names, image, device, imgsz=960):
    result = model.predict(image, classes=classes, conf=0.001, iou=.7, imgsz=imgsz, max_det=300,
                           device=device, retina_masks=True, verbose=False)[0]
    output = []
    if result.masks is None:
        return output
    data = result.boxes.data.cpu().numpy()
    masks = result.masks.data.cpu().numpy() > .5
    for row, mask, points in zip(data, masks, result.masks.xy):
        if row[-2] < CONF or not len(points):
            continue
        h, w = image.shape[:2]
        x1, y1 = np.clip(np.floor(row[:2]).astype(int), 0, [w, h])
        x2, y2 = np.clip(np.ceil(row[2:4]).astype(int), 0, [w, h])
        rect = np.zeros((h, w), np.uint8)
        rect[y1:y2, x1:x2] = 1
        convex = filled(cv2.convexHull(np.asarray(points, np.int32)), (h, w))
        hull = compute_hull(points, 'ograham')
        output.append(dict(cls=CLASSES.index(names[int(row[-1])]), score=float(row[-2]), mask=mask.astype(np.uint8),
                           rect=rect, convex=convex, hull=filled(hull, (h, w)), points=np.asarray(points),
                           hull_vertices=turning_vertices(hull)))
    return output


def pair(predictions, targets):
    """Class-aware pairs of the YOLO mask at IoU >= 0.5, the protocol used for TP."""
    ious = np.array([[region_iou(region(p['mask']), t['region']) for t in targets] for p in predictions],
                    dtype=np.float32).reshape(len(predictions), len(targets))
    record = dict(scores=[p['score'] for p in predictions], pred_classes=[p['cls'] for p in predictions],
                  gt_classes=[t['class'] for t in targets], ious={'yolo_mask': ious})
    return matching(record, 'yolo_mask', CONF)[1], record


def outline(ax, mask, color, width=1.4):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        c = c.reshape(-1, 2)
        ax.plot(*np.vstack([c, c[:1]]).T, color=color, linewidth=width)


def crop_figure(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = ['gt', 'rect', 'mask', 'convex', 'hull']
    ratios = [r['crop'].shape[0]/r['crop'].shape[1] for _, r in rows]
    fig, axes = plt.subplots(len(rows), 5, figsize=(10, sum(1.9*r+.55 for r in ratios)), squeeze=False,
                             gridspec_kw=dict(height_ratios=[r+.29 for r in ratios]))
    for line, (label, row) in zip(axes, rows):
        for ax, key in zip(line, keys):
            ax.imshow(row['crop'][..., ::-1])
            outline(ax, row['regions'][key], COLORS[key])
            title = TITLES[key]+('' if key == 'gt' else f"\nIoU {row['iou'][key]:.2f}".replace('.', ','))
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        line[0].set_ylabel(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def scene_figure(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def window(image, targets, predictions):
        h, w = image.shape[:2]
        items = [t['region'] for t in targets]+[region(p['mask']) for p in predictions]
        x1 = min(r[0] for r in items); y1 = min(r[1] for r in items)
        x2 = max(r[0]+r[2].shape[1] for r in items); y2 = max(r[1]+r[2].shape[0] for r in items)
        pad = int(.06*max(x2-x1, y2-y1))
        return max(0, x1-pad), min(w, x2+pad), max(0, y1-pad), min(h, y2+pad)

    windows = [window(*row[1:]) for row in rows]
    ratios = [(y2-y1)/(x2-x1) for x1, x2, y1, y2 in windows]
    fig, axes = plt.subplots(len(rows), 3, figsize=(10, sum(3.2*r+.45 for r in ratios)), squeeze=False,
                             gridspec_kw=dict(height_ratios=[r+.14 for r in ratios]))
    for line, (label, image, targets, predictions), (x1, x2, y1, y2) in zip(axes, rows, windows):
        pairs, record = pair(predictions, targets)
        h, w = image.shape[:2]
        missed = [j for j in range(len(targets)) if j not in pairs.values()]
        for ax, key in zip(line, ['gt', 'mask', 'hull']):
            ax.imshow(image[..., ::-1])
            if key == 'gt':
                for t in targets:
                    outline(ax, full_mask(t['region'], (h, w)), '#3aa0ff', .9)
                    ax.text(t['region'][0], t['region'][1]-2, CLASSES[t['class']], color='#3aa0ff', fontsize=5)
            else:
                for i, p in enumerate(predictions):
                    color = '#2bd24b' if i in pairs else '#ff3b30'
                    outline(ax, p[key], color, .9)
                    r = region(p['mask'])
                    ax.text(r[0], r[1]-2, CLASSES[p['cls']], color=color, fontsize=5)
                for j in missed:
                    outline(ax, full_mask(targets[j]['region'], (h, w)), '#ff9f0a', .9)
            ax.set_xlim(x1, x2); ax.set_ylim(y2, y1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(TITLES[key], fontsize=9)
        line[0].set_ylabel(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def selected_scenes(per_image, dataset):
    with open(per_image, newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['method'] == 'yolo_mask']
    key = lambda r: (float(r['f1']), -int(r['fn']), -int(r['fp']))
    if dataset == 'vietnam':
        easy = [r for r in rows if r['difficulty'] == 'easy']
        hard = [r for r in rows if r['difficulty'] == 'hard']
        return [('Dễ, F1 cao nhất', max(easy, key=key)), ('Khó, F1 thấp nhất', min(hard, key=key))]
    ordered = sorted(rows, key=key)
    return [('F1 trung vị', ordered[len(ordered)//2]), ('F1 thấp nhất', ordered[0])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', type=Path, required=True, help='Directory with the four evaluation runs')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=list(DATASETS))
    args = parser.parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, crops, scenes = {}, [], []
    for dataset in args.datasets:
        root, load_gt = DATASETS[dataset]
        with (root/'manifest.csv').open() as f:
            manifest = list(csv.DictReader(f))
        for size, weights in MODELS.items():
            model = YOLO(str(weights))
            names = {i: n.lower() for i, n in model.names.items()}
            classes = [i for i, n in names.items() if n in CLASSES]
            stats = defaultdict(list)
            for row in manifest:
                image = cv2.imread(str(root/row['image_path']))
                targets, _ = load_gt(root/row['label_path'], image.shape[:2])
                predictions = predict(model, classes, names, image, device)
                for p in predictions:
                    fragments, holes = topology(p['mask'])
                    stats['contour_points'].append(len(p['points']))
                    stats['hull_vertices'].append(p['hull_vertices'])
                    stats['fragments'].append(fragments)
                    stats['holes'].append(holes)
                if not predictions or not targets:
                    continue
                pairs, _ = pair(predictions, targets)
                for i, j in pairs.items():
                    p, t = predictions[i], targets[j]
                    iou = {k: region_iou(region(p[k]), t['region']) for k in ['rect', 'mask', 'convex', 'hull']}
                    for key, value in iou.items():
                        stats['iou_'+key].append(value)
                        stats[f"iou_{key}_{CLASSES[t['class']]}"].append(value)
                    if dataset == 'vietnam' and size == 'large' and t['region'][3] >= MIN_CROP_AREA:
                        x, y, m, _ = t['region']
                        pad = int(.12*max(m.shape))
                        h, w = image.shape[:2]
                        x1, x2, y1, y2 = max(0, x-pad), min(w, x+m.shape[1]+pad), max(0, y-pad), min(h, y+m.shape[0]+pad)
                        window = lambda mask: mask[y1:y2, x1:x2].copy()
                        regions = {k: window(p[k]) for k in ['rect', 'mask', 'convex', 'hull']}
                        regions['gt'] = window(full_mask(t['region'], (h, w)))
                        crops.append(dict(cls=t['class'], iou=iou, crop=image[y1:y2, x1:x2].copy(), regions=regions))
            n = len(stats['contour_points'])
            summary[f'{dataset}-{size}'] = dict(
                predictions=n,
                contour_points_median=float(np.median(stats['contour_points'])),
                hull_vertices_median=float(np.median(stats['hull_vertices'])),
                vertex_ratio_median=float(np.median(np.array(stats['hull_vertices'])/np.array(stats['contour_points']))),
                masks_with_fragments=int(np.sum(np.array(stats['fragments']) > 1)),
                masks_with_holes=int(np.sum(np.array(stats['holes']) > 0)),
                pairs=len(stats['iou_mask']),
                mean_iou={k: float(np.mean(stats['iou_'+k])) for k in ['rect', 'mask', 'convex', 'hull']},
                hull_at_least_mask=int(np.sum(np.array(stats['iou_hull']) >= np.array(stats['iou_mask']))),
                by_class={c: dict(pairs=len(stats[f'iou_mask_{c}']),
                                  mask=float(np.mean(stats[f'iou_mask_{c}'])) if stats[f'iou_mask_{c}'] else None,
                                  hull=float(np.mean(stats[f'iou_hull_{c}'])) if stats[f'iou_hull_{c}'] else None)
                          for c in CLASSES})
            print(dataset, size, json.dumps(summary[f'{dataset}-{size}'], ensure_ascii=False), flush=True)
            if size == 'large':
                run = args.evaluation_dir/f"{'vietnam' if dataset == 'vietnam' else 'bdd100k'}-seg-100-large"
                for label, picked in selected_scenes(run/'per_image.csv', dataset):
                    match = next(r for r in manifest if r['filename'] == picked['filename'])
                    image = cv2.imread(str(root/match['image_path']))
                    targets, _ = load_gt(root/match['label_path'], image.shape[:2])
                    scenes.append((('Việt Nam: ' if dataset == 'vietnam' else 'BDD100K: ')+label, image, targets,
                                   predict(model, classes, names, image, device)))
                    print('scene', label, picked['filename'], picked['f1'], flush=True)
    (args.output_dir/'compare_mask_hull.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    rows = []
    for c, name in enumerate(CLASSES):
        group = sorted((x for x in crops if x['cls'] == c), key=lambda x: x['iou']['hull'])
        rows.append((f'{name} (trung vị)', group[len(group)//2]))
    worst = max(crops, key=lambda x: x['iou']['mask']-x['iou']['hull'])
    rows.append((f"{CLASSES[worst['cls']]} (giảm nhiều nhất)", worst))
    crop_figure(rows, args.output_dir/'crop_compare.png')
    for name, group in [('scene_compare.png', scenes[:2]), ('scene_bdd.png', scenes[2:])]:
        if group:
            scene_figure([(f'Hàng {i}', *row[1:]) for i, row in enumerate(group, 1)], args.output_dir/name)
    print('crops', len(crops), 'saved', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
