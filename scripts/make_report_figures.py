#!/usr/bin/env python3
"""Draw the report's mask-versus-hull chart and speed chart from saved results."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METHODS = [('bbox_region', 'Hình chữ nhật'), ('convex', 'Bao lồi thường'), ('ograham', 'Bao trực giao'), ('yolo_mask', 'Mặt nạ YOLO')]
METRICS = [('map50', 'mAP50', '#2a78d6'), ('map50_95', 'mAP50–95', '#eb6834')]
GRID, AXIS, MUTED = '#e4e3df', '#52514e', '#52514e'


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(AXIS)
    ax.tick_params(colors=AXIS, length=3)
    ax.grid(axis='y', color=GRID, linewidth=.8)
    ax.set_axisbelow(True)


def quality(evaluation, path):
    overall = json.loads((evaluation/'vietnam-seg-100-large/metrics.json').read_text())['overall']
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    width = .36
    for m, (key, name, color) in enumerate(METRICS):
        for i, (method, _) in enumerate(METHODS):
            value = overall[method][key]*100
            x = i+(m-.5)*width
            ax.bar(x, value, width-.02, color=color, label=name if i == 0 else None)
            ax.text(x, value+1, f'{value:.1f}'.replace('.', ','), ha='center', fontsize=8.5, color=MUTED)
    ax.set_xticks(range(len(METHODS)), [name for _, name in METHODS])
    ax.set_ylabel('Giá trị (%)')
    ax.set_ylim(0, 90)
    ax.legend(frameon=False, ncol=2, loc='upper left')
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def speed(csv_path, path):
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    methods = [('quickhull', 'Bản đệ quy', '#2a78d6', 'o'), ('ograham', 'O-Graham', '#eb6834', 's')]
    for ax, (kind, title) in zip(axes, [('uniform', 'Phân bố đều'), ('circle', 'Trên đường tròn')]):
        for method, name, color, marker in methods:
            data = [r for r in rows if r['input'] == kind and r['algorithm'] == method]
            ax.plot([int(r['n']) for r in data], [float(r['median_ms']) for r in data], color=color,
                    marker=marker, linewidth=2, markersize=6, markeredgecolor='white', label=name)
        ax.set(xscale='log', yscale='log', title=title, xlabel='Số điểm n')
        style(ax)
    axes[0].set_ylabel('Thời gian (ms, trung vị)')
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', type=Path, default=ROOT/'results/evaluation')
    parser.add_argument('--speed-csv', type=Path, default=ROOT/'results/benchmark/hull-speed/synthetic.csv')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quality(args.evaluation_dir, args.output_dir/'mask_vs_hull_map.png')
    speed(args.speed_csv, args.output_dir/'hull_speed.png')


if __name__ == '__main__':
    main()
