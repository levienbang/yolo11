"""Identical normalization and geometry features for both hull algorithms."""
import numpy as np
from .quickhull import findOrthogonalConvexHull
from .OGraham import o_graham


def normalize_hull(points):
    p = np.asarray(points, dtype=np.int32).reshape(-1, 2)
    if not len(p):
        return p
    p = p[np.r_[True, np.any(p[1:] != p[:-1], axis=1)]]
    if len(p) > 1 and not np.array_equal(p[0], p[-1]):
        p = np.vstack([p, p[0]])
    return p


def compute_hull(points, algorithm):
    p = np.unique(np.asarray(points, dtype=np.int32).reshape(-1, 2), axis=0)
    if len(p) < 4:
        return normalize_hull(p)
    if algorithm == 'quickhull':
        return normalize_hull(findOrthogonalConvexHull(list(map(tuple, p.tolist()))))
    if algorithm == 'ograham':
        return normalize_hull(o_graham(p))
    raise ValueError(f'Unknown algorithm: {algorithm}')


def extract_vector_features(hull, bbox):
    p = np.asarray(hull, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3:
        return 0.0, 0.0, 0.0, len(p)
    if np.array_equal(p[0], p[-1]):
        p = p[:-1]
    x, y = p.T
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    perimeter = np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1).sum()
    bbox_area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return float(area), float(perimeter), float(area / bbox_area), len(p)
