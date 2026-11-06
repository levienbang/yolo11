#!/usr/bin/env python3
"""Check on real vehicle point sets that the connected orthogonal hull is unique.

A grid point is in the classical orthogonal hull when each of its four closed
quadrants contains a point of P. The connected hull is unique when that region is
connected, and the built hull must then match it pixel for pixel.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from vehicle_pipeline.hull_common import compute_hull


def classical_region(points):
    """Return the classical hull as a boolean grid, with the unique points and grid origin."""
    p = np.unique(np.asarray(points, np.int32).reshape(-1, 2), axis=0)
    origin = p.min(axis=0)
    size = p.max(axis=0)-origin+1
    grid = np.zeros((size[1], size[0]), bool)
    grid[p[:, 1]-origin[1], p[:, 0]-origin[0]] = True

    def reach(step_y, step_x):
        flipped = grid[::step_y, ::step_x]
        return np.logical_or.accumulate(np.logical_or.accumulate(flipped, axis=0), axis=1)[::step_y, ::step_x]

    return reach(1, 1) & reach(1, -1) & reach(-1, 1) & reach(-1, -1), p, origin


def hull_region(hull, origin, shape):
    """Rasterize a hull so that its zero-width edges are included."""
    region = np.zeros(shape, np.uint8)
    shifted = (np.asarray(hull)-origin).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(region, [shifted], 1)
    cv2.polylines(region, [shifted], True, 1, 1)
    return region.astype(bool)


def inspect(points):
    region, unique, origin = classical_region(points)
    components = cv2.connectedComponents(region.astype(np.uint8), connectivity=4)[0]-1
    matches = all(np.array_equal(hull_region(compute_hull(unique, a), origin, region.shape), region)
                  for a in ('quickhull', 'ograham'))
    return dict(unique_hull=components == 1, matches=bool(matches))


def summarize(rows):
    return dict(objects=len(rows), unique_hull=sum(r['unique_hull'] for r in rows),
                matches_region=sum(r['matches'] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    from ultralytics import YOLO
    from benchmark_hulls import real_contours
    import compare_mask_hull as compare
    from vehicle_pipeline.device import resolve_device
    root = compare.DATASETS['vietnam'][0]
    summary = {'label_masks': summarize([inspect(c) for c in real_contours(root)])}
    device = resolve_device(args.device)
    with (root/'manifest.csv').open() as f:
        manifest = list(csv.DictReader(f))
    for size, weights in compare.MODELS.items():
        model = YOLO(str(weights))
        names = {i: n.lower() for i, n in model.names.items()}
        classes = [i for i, n in names.items() if n in compare.CLASSES]
        rows = []
        for row in manifest:
            image = cv2.imread(str(root/row['image_path']))
            rows += [inspect(p['points']) for p in compare.predict(model, classes, names, image, device)
                     if len(np.unique(np.asarray(p['points'], np.int32), axis=0)) >= 2]
        summary[f'predicted_masks_{size}'] = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
