from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import random
import unittest
import cv2
import numpy as np
from check_hull_uniqueness import classical_region, hull_region
from vehicle_pipeline.hull_common import compute_hull, extract_vector_features


def maximal_points(points):
    """Points having a closed orthant free of other points, by brute force."""
    result = set()
    for p in points:
        delta = np.delete(points, np.flatnonzero((points == p).all(axis=1)), axis=0) - p
        for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            if not np.any((sx*delta[:, 0] >= 0) & (sy*delta[:, 1] >= 0)):
                result.add(tuple(p.tolist()))
    return result


class HullTests(unittest.TestCase):
    def check_independent(self, points, hull):
        """Check the hull against its definition, not against the other method."""
        steps = np.diff(hull, axis=0)
        self.assertTrue(np.all((steps[:, 0] == 0) | (steps[:, 1] == 0)))
        origin = points.min(axis=0)
        size = points.max(axis=0) - origin + 1
        grid = np.zeros((size[1], size[0]), np.uint8)
        shifted = (hull - origin).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(grid, [shifted], 1)
        cv2.polylines(grid, [shifted], True, 1, 1)
        # Contains P.
        self.assertTrue(np.all(grid[points[:, 1]-origin[1], points[:, 0]-origin[0]] == 1))
        # Connected, using edge adjacency so that a staircase cannot jump a corner.
        self.assertEqual(cv2.connectedComponents(grid, connectivity=4)[0], 2)
        # Orthogonally convex: every row and column meets the region in one run.
        for line in [*grid, *grid.T]:
            filled = np.flatnonzero(line)
            if len(filled):
                self.assertEqual(filled[-1]-filled[0]+1, len(filled))
        # Every maximal point of P must be a vertex (Lemma 5 of the O-Quickhull paper).
        self.assertLessEqual(maximal_points(points), set(map(tuple, hull.tolist())))
        # Where the classical hull is connected, the connected hull is unique and equal to it.
        region, _, region_origin = classical_region(points)
        if cv2.connectedComponents(region.astype(np.uint8), connectivity=4)[0] == 2:
            np.testing.assert_array_equal(hull_region(hull, region_origin, region.shape), region)

    def check_geometry(self, points):
        hulls = [compute_hull(points, a) for a in ('quickhull', 'ograham')]
        for h in hulls:
            if len(h) > 1:
                np.testing.assert_array_equal(h[0], h[-1])
                self.assertTrue(np.all(np.any(h[1:] != h[:-1], axis=1)))
        features = [extract_vector_features(h, [0, 0, 40, 40]) for h in hulls]
        np.testing.assert_allclose(features[0][:3], features[1][:3])
        unique = np.unique(np.asarray(points, dtype=np.int32).reshape(-1, 2), axis=0)
        if len(unique) < 2:
            return
        for h in hulls:
            self.check_independent(unique, h)
        boundaries = []
        for h in hulls:
            np.testing.assert_array_equal(h.min(axis=0), np.min(points, axis=0))
            np.testing.assert_array_equal(h.max(axis=0), np.max(points, axis=0))
            steps = np.diff(h, axis=0)
            self.assertTrue(np.all((steps[:, 0] == 0) | (steps[:, 1] == 0)))
            edges = set()
            for p, q in zip(h, h[1:]):
                direction = np.sign(q - p)
                for i in range(int(np.abs(q - p).sum())):
                    edges.add(tuple(sorted((tuple(p + direction*i), tuple(p + direction*(i+1))))))
            boundaries.append(edges)
        self.assertEqual(boundaries[0], boundaries[1])

    def test_rectangle(self):
        points = [(0, 0), (10, 0), (10, 20), (0, 20), (3, 5), (8, 9)]
        self.check_geometry(points)
        for a in ('quickhull', 'ograham'):
            area, perimeter, _, count = extract_vector_features(compute_hull(points, a), [0, 0, 10, 20])
            self.assertEqual((area, perimeter, count), (200, 60, 4))

    def test_staircase(self):
        self.check_geometry([(0, 4), (1, 8), (3, 10), (8, 9), (10, 6),
                             (9, 3), (7, 0), (3, 1), (1, 2), (5, 5)])

    def test_diagonal_clusters_are_not_unique(self):
        region, _, _ = classical_region([(0, 0), (2, 0), (0, 2), (2, 2), (6, 6), (8, 6), (6, 8), (8, 8)])
        self.assertEqual(cv2.connectedComponents(region.astype(np.uint8), connectivity=4)[0]-1, 2)

    def test_fewer_than_four_points_stay_axis_parallel(self):
        for p in [[(0, 0), (5, 3)], [(0, 0), (1, 1), (2, 0)]]:
            self.check_geometry(p)

    def test_degenerate(self):
        for p in [[], [(2, 2)], [(1, 1)]*6, [(0, 0), (1, 0), (2, 0), (3, 0)],
                  [(0, 0), (0, 1), (0, 2), (0, 3)], [(0, 0), (1, 1), (2, 2), (3, 3)]]:
            self.check_geometry(p)

    def test_random_ties_and_shuffled_points(self):
        r = random.Random(2026)
        for _ in range(1000):
            p = [(r.randrange(20), r.randrange(20)) for _ in range(r.randrange(4, 150))]
            r.shuffle(p)
            self.check_geometry(p)


if __name__ == '__main__':
    unittest.main()
