"""Peter's O-Graham sort/scan with endpoint, tie and empty-input fixes.

Independent O-Graham implementation; no QuickHull fallback.
"""
import numpy as np


def o_graham(points):
    pts = sorted(set(map(tuple, points)))
    if len(pts) < 2:
        return np.asarray(pts, dtype=np.int32).reshape(-1, 2)
    min_x, max_x = min(p[0] for p in pts), max(p[0] for p in pts)
    min_y, max_y = min(p[1] for p in pts), max(p[1] for p in pts)
    top = sorted(p for p in pts if p[1] == max_y)
    bottom = sorted((p for p in pts if p[1] == min_y), reverse=True)
    left = sorted((p for p in pts if p[0] == min_x), key=lambda p: p[1])
    right = sorted((p for p in pts if p[0] == max_x), key=lambda p: -p[1])
    specs = [
        (top[0], left[-1], lambda p: (-p[1], p[0]), 1, 0, -1),
        (left[0], bottom[-1], lambda p: (p[0], p[1]), 0, 1, -1),
        (bottom[0], right[-1], lambda p: (p[1], -p[0]), 1, 0, 1),
        (right[0], top[-1], lambda p: (-p[0], -p[1]), 0, 1, 1),
    ]
    hull = []
    for start, end, sort_key, sort_axis, scan_axis, direction in specs:
        region = [p for p in pts
                  if min(start[0], end[0]) <= p[0] <= max(start[0], end[0])
                  and min(start[1], end[1]) <= p[1] <= max(start[1], end[1])]
        chain, previous_sort = [start], None
        for p in sorted(region, key=sort_key):
            if p[sort_axis] == previous_sort:
                continue
            previous_sort = p[sort_axis]
            if direction * (p[scan_axis] - chain[-1][scan_axis]) > 0:
                chain.append(p)
        if chain[-1] != end:
            chain.append(end)
        hull.extend(chain)
    vertices = []
    for p, following in zip(hull, hull[1:] + hull[:1]):
        vertices.append(p)
        dx, dy = following[0] - p[0], following[1] - p[1]
        if dx and dy:
            vertices.append((p[0], following[1]) if dx * dy > 0
                            else (following[0], p[1]))
    return np.asarray(vertices + vertices[:1], dtype=np.int32)
