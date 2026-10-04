from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import random
import unittest
import numpy as np
from vehicle_pipeline.hull_common import compute_hull, extract_vector_features


class HullTests(unittest.TestCase):
    def check_geometry(self, points):
        hulls = [compute_hull(points, a) for a in ('quickhull', 'ograham')]
        for h in hulls:
            if len(h) > 1:
                np.testing.assert_array_equal(h[0], h[-1])
                self.assertTrue(np.all(np.any(h[1:] != h[:-1], axis=1)))
        features = [extract_vector_features(h, [0, 0, 40, 40]) for h in hulls]
        np.testing.assert_allclose(features[0][:3], features[1][:3])
        if len(np.unique(points, axis=0)) < 4:
            return
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
