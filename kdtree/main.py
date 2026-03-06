import numpy as np
from scipy.spatial import KDTree as ScipyKDTree

from kdtree import KDTree


def test_nearest(tree, tree_ref, queries, num):
    print(f"{"nearest":10s}", end=" ")
    for q in queries:
        idx, dist = tree.nearest(q, num, sorted=True)
        dist_ref, idx_ref = tree_ref.query(q, num)
        idx_ref = idx_ref.astype(np.intc)
        assert np.array_equal(idx, idx_ref), "index mismatch"
        assert np.allclose(dist, dist_ref), "distance mismatch"
    print("ok")


def test_radius(tree, tree_ref, queries, radius):
    print(f"{"radius":10s}", end=" ")
    for q in queries:
        idx, dist = tree.radius(q, radius, sorted=True)
        idx_ref = tree_ref.query_ball_point(q, radius)
        assert set(idx.tolist()) == set(idx_ref), "index mismatch"
        assert np.all(dist[:-1] <= dist[1:]), "not sorted"
    print("ok")


def test_pairs(tree, tree_ref, radius):
    print(f"{"pairs":10s}", end=" ")
    pair = tree.pairs(radius)
    pair_ref = tree_ref.query_pairs(radius)
    assert pair == pair_ref, "pair mismatch"
    print("ok")


def test_cross(tree, tree_ref, other, other_ref, radius):
    print(f"{"cross":10s}", end=" ")
    pair = tree.cross(other, radius)
    list_ref = tree_ref.query_ball_tree(other_ref, radius)
    pair_ref = {(i, j) for i, js in enumerate(list_ref) for j in js}
    assert pair == pair_ref, "pair mismatch"
    print("ok")


def test_counts(tree, tree_ref, radii):
    print(f"{"counts":10s}", end=" ")
    # scipy counts N self-pairs (distance=0) and both orderings of each pair,
    # so scipy_total = N + 2 * unique_pairs; recover unique_pairs accordingly.
    scipy_total = tree_ref.count_neighbors(tree_ref, radii)
    counts_ref = (scipy_total - tree_ref.n) // 2
    assert np.array_equal(tree.counts(radii), counts_ref)
    assert np.array_equal(tree.counts(radii, cumulative=False), np.diff(counts_ref, prepend=0))
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
    tree_ref = ScipyKDTree(points, leaf_size)

    other = KDTree(queries, leaf_size)
    other_ref = ScipyKDTree(queries, leaf_size)

    test_nearest(tree, tree_ref, queries, num)
    test_radius(tree, tree_ref, queries, radius)
    test_pairs(tree, tree_ref, radius)
    test_counts(tree, tree_ref, np.linspace(0.05, 0.3, 6))
    test_cross(tree, tree_ref, other, other_ref, radius)


if __name__ == "__main__":
    main()
