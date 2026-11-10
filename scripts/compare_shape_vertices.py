#!/usr/bin/env python3
"""Compare IoU and turning-vertex counts on the same ground-truth vehicle masks.

No inference is needed. This reproduces the report's shape-representation table
and adds vertex counts for the rectangle, convex hull and orthogonal hull.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prepare_vietnam_segmentation import load_polygons, names_from_yaml
from vehicle_pipeline.hull_common import compute_hull
from vehicle_pipeline.vehicle_benchmark import CLASSES

METHODS = ('rectangle', 'convex', 'orthogonal')


def vertex_count(polygon):
    """Count turns, excluding closure, repeated points and straight continuations.

    A reversal is retained: an orthogonal boundary can contain a thin spur.
    Coordinates are promoted to int64 before cross/dot products.
    """
    p = np.asarray(polygon, dtype=np.int64).reshape(-1, 2)
    if not len(p):
        return 0
    p = p[np.r_[True, np.any(np.diff(p, axis=0), axis=1)]]
    if len(p) > 1 and np.array_equal(p[0], p[-1]):
        p = p[:-1]
    if len(p) < 3:
        return len(p)
    incoming = p - np.roll(p, 1, axis=0)
    outgoing = np.roll(p, -1, axis=0) - p
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    dot = np.sum(incoming * outgoing, axis=1)
    return int(np.count_nonzero((cross != 0) | (dot <= 0)))


def measure(dataset):
    names = names_from_yaml(dataset / 'dataset.yaml')
    with (dataset / 'manifest.csv').open() as stream:
        manifest = list(csv.DictReader(stream))
    rows = []
    for item in manifest:
        image = cv2.imread(str(dataset / item['image_path']))
        if image is None:
            raise ValueError(f"Cannot read {item['image_path']}")
        h, w = image.shape[:2]
        targets = load_polygons(dataset / item['label_path'], names, (h, w))
        for index, target in enumerate(targets):
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [target['pixels'].reshape(-1, 1, 2)], 1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            points = np.vstack([c.reshape(-1, 2) for c in contours])
            x, y, width, height = cv2.boundingRect(points)
            polygons = {
                'rectangle': np.array([[x, y], [x + width - 1, y],
                                       [x + width - 1, y + height - 1], [x, y + height - 1]]),
                'convex': cv2.convexHull(points).reshape(-1, 2),
                'orthogonal': compute_hull(points, 'ograham'),
            }
            record = dict(filename=item['filename'], object_index=index,
                          class_name=CLASSES[target['class_id']])
            for method, polygon in polygons.items():
                filled = np.zeros_like(mask)
                cv2.fillPoly(filled, [np.asarray(polygon, np.int32).reshape(-1, 1, 2)], 1)
                record[method + '_iou'] = float(np.count_nonzero(mask & filled)
                                               / np.count_nonzero(mask | filled))
                record[method + '_vertices'] = vertex_count(polygon)
            rows.append(record)
    return rows


def summarize(rows):
    summary = {}
    for name in CLASSES + ['all']:
        group = [row for row in rows if name == 'all' or row['class_name'] == name]
        summary[name] = dict(objects=len(group), methods={
            method: dict(mean_iou=float(np.mean([r[method + '_iou'] for r in group])),
                         mean_vertices=float(np.mean([r[method + '_vertices'] for r in group])))
            for method in METHODS})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'datasets/vietnam-seg-review-100')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/shape')
    args = parser.parse_args()
    rows = measure(args.dataset)
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'per_object.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = dict(protocol='Ground-truth masks; external pixel contours; filled polygons; '
                           'turning vertices excluding closure and straight continuations; '
                           'arithmetic means over vehicles, not means of class means.',
                  summary=summary)
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
