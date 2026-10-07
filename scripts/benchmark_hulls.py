#!/usr/bin/env python3
"""Controlled speed benchmark: QuickHull vs O-Graham orthogonal hull.

Two inputs: synthetic point sets of growing n (uniform square, circle boundary)
and real vehicle contours from the 100 Vietnam GT polygons. Warm-up runs are
discarded; the median of repeated runs is reported. Only hull construction is timed.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from vehicle_pipeline.hull_common import compute_hull
from prepare_vietnam_segmentation import load_polygons, names_from_yaml
ALGORITHMS = ['quickhull', 'ograham']
sys.setrecursionlimit(100000)  # QuickHull recursion depth grows with hull vertices.


def timed(points, algorithm, warmup, repeats):
    for _ in range(warmup):
        compute_hull(points, algorithm)
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        compute_hull(points, algorithm)
        samples.append((perf_counter()-start)*1000)
    return float(np.median(samples)), float(np.percentile(samples, 25)), float(np.percentile(samples, 75))


def synthetic(kind, n, rng, size=4000):
    if kind == 'uniform':
        return rng.integers(0, size, (n, 2))
    angle = rng.uniform(0, 2*np.pi, n)
    return np.round(size/2 + size/2*np.c_[np.cos(angle), np.sin(angle)]).astype(int)


def real_contours(dataset):
    names = names_from_yaml(dataset/'dataset.yaml')
    with (dataset/'manifest.csv').open() as f:
        rows = list(csv.DictReader(f))
    contours = []
    for row in rows:
        h, w = cv2.imread(str(dataset/row['image_path'])).shape[:2]
        for target in load_polygons(dataset/row['label_path'], names, (h, w)):
            mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(mask, [target['pixels'].reshape(-1, 1, 2)], 1)
            found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            points = np.vstack([c.reshape(-1, 2) for c in found])
            contours.append(points)
    return contours


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'datasets/vietnam-seg-review-100')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/benchmark/hull-speed')
    parser.add_argument('--sizes', type=int, nargs='+', default=[100, 500, 1000, 2000, 5000, 10000])
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    for kind in ['uniform', 'circle']:
        for n in args.sizes:
            points = synthetic(kind, n, rng)
            hulls = [compute_hull(points, a) for a in ALGORITHMS]
            same = np.array_equal(*[np.unique(h, axis=0) for h in hulls])
            for algorithm, hull in zip(ALGORITHMS, hulls):
                median, q1, q3 = timed(points, algorithm, args.warmup, args.repeats)
                rows.append(dict(input=kind, n=n, unique_points=len(np.unique(points, axis=0)),
                                 algorithm=algorithm, median_ms=median, q1_ms=q1, q3_ms=q3,
                                 hull_vertices=len(hull)-1, same_vertices_as_other=same))
            print(kind, n, {r['algorithm']: round(r['median_ms'], 3) for r in rows[-2:]}, flush=True)

    contours = real_contours(args.dataset)
    real = {a: [] for a in ALGORITHMS}
    for points in contours:
        for algorithm in ALGORITHMS:
            real[algorithm].append(timed(points, algorithm, 1, 5)[0])
    sizes = np.array([len(p) for p in contours])
    real_summary = {a: dict(objects=len(v), contour_points_median=int(np.median(sizes)),
                            contour_points_max=int(sizes.max()), median_ms=float(np.median(v)),
                            mean_ms=float(np.mean(v)), p95_ms=float(np.percentile(v, 95)), total_ms=float(np.sum(v)))
                    for a, v in real.items()}
    real_summary['ograham_faster_ratio_median'] = float(np.median(np.array(real['quickhull'])/np.array(real['ograham'])))

    with (args.output_dir/'synthetic.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir/'summary.json').write_text(json.dumps(
        dict(settings=vars(args) | {'dataset': str(args.dataset), 'output_dir': str(args.output_dir)},
             real_contours=real_summary), indent=2, default=str))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, kind in zip(axes, ['uniform', 'circle']):
        for algorithm in ALGORITHMS:
            data = [r for r in rows if r['input'] == kind and r['algorithm'] == algorithm]
            ax.errorbar([r['n'] for r in data], [r['median_ms'] for r in data],
                        yerr=[[r['median_ms']-r['q1_ms'] for r in data], [r['q3_ms']-r['median_ms'] for r in data]],
                        marker='o', capsize=3, label=algorithm)
        ax.set(xscale='log', yscale='log', xlabel='n (số điểm)', ylabel='ms (median)', title=f'Điểm {kind}')
        ax.grid(alpha=.3, which='both')
        ax.legend()
    axes[2].scatter(sizes, real['quickhull'], s=6, alpha=.5, label='quickhull')
    axes[2].scatter(sizes, real['ograham'], s=6, alpha=.5, label='ograham')
    axes[2].set(xscale='log', yscale='log', xlabel='điểm biên mỗi xe', ylabel='ms (median 5 lần)',
                title=f'Contour thật ({len(contours)} xe Việt Nam)')
    axes[2].grid(alpha=.3, which='both')
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir/'hull_speed.png', dpi=160)
    print(json.dumps(real_summary, indent=2))


if __name__ == '__main__':
    main()
