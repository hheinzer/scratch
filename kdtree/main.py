import numpy as np
from scipy.spatial import KDTree as ScipyKDTree

from kdtree import KDTree


def test_nearest(tree, ref, queries, num):
    print(f"{"nearest":10s}", end=" ")
    for q in queries:
        idx, dist = tree.nearest(q, num)
        dist_ref, idx_ref = ref.query(q, num)
        idx_ref = idx_ref.astype(np.intc)
        assert np.array_equal(idx, idx_ref), "index mismatch"
        assert np.allclose(dist, dist_ref), "distance mismatch"
    print("ok")


def test_radius(tree, ref, queries, radius):
    print(f"{"radius":10s}", end=" ")
    for q in queries:
        idx, dist = tree.radius(q, radius, sorted=True)
        idx_ref = ref.query_ball_point(q, radius)
        assert set(idx.tolist()) == set(idx_ref), "index mismatch"
        assert np.all(dist[:-1] <= dist[1:]), "not sorted"
    print("ok")


def test_pairs(tree, ref, radius):
    print(f"{"pairs":10s}", end=" ")
    pair = tree.pairs(radius)
    pair_ref = ref.query_pairs(radius)
    assert pair == pair_ref, "pair mismatch"
    print("ok")


def main():
    rng = np.random.default_rng(42)

    num_points = 10000
    num_queries = 100
    dim = 3
    leaf_size = 16
    num = 8
    radius = 0.2

    points = rng.uniform(-1, 1, (num_points, dim))
    queries = rng.uniform(-1, 1, (num_queries, dim))

    tree = KDTree(points, leaf_size)
    ref = ScipyKDTree(points, leaf_size)

    test_nearest(tree, ref, queries, num)
    test_radius(tree, ref, queries, radius)
    test_pairs(tree, ref, radius)


if __name__ == "__main__":
    main()
