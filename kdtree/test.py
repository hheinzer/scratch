import numpy as np
from scipy.spatial import KDTree as ScipyKDTree

from kdtree import KDTree


def test_nearest(tree, tree_ref, queries, num):
    print(f"{"nearest":10s}", end=" ")
    idx, dist = tree.nearest(queries, num, sorted=True)
    dist_ref, idx_ref = tree_ref.query(queries, num)
    assert np.array_equal(idx, idx_ref), "index mismatch"
    assert np.allclose(dist, dist_ref), "distance mismatch"
    print("passed")


def test_radius(tree, tree_ref, queries, radius):
    print(f"{"radius":10s}", end=" ")
    results = tree.radius(queries, radius, sorted=True)
    results_ref = tree_ref.query_ball_point(queries, radius)
    for (idx_i, dist_i), idx_ref in zip(results, results_ref):
        assert set(idx_i.tolist()) == set(idx_ref), "index mismatch"
        assert np.all(dist_i[:-1] <= dist_i[1:]), "not sorted"
    print("passed")


def test_pairs(tree, tree_ref, other, other_ref, radius):
    print(f"{"pairs":10s}", end=" ")
    pairs = tree.pairs(radius)
    pairs_ref = tree_ref.query_pairs(radius)
    assert pairs == pairs_ref, "self mismatch"
    pairs = tree.pairs(radius, other)
    list_ref = tree_ref.query_ball_tree(other_ref, radius)
    pairs_ref = {(i, j) for i, js in enumerate(list_ref) for j in js}
    assert pairs == pairs_ref, "cross mismatch"
    print("passed")


def test_counts(tree, tree_ref, other, other_ref, radii):
    print(f"{"counts":10s}", end=" ")
    counts = tree.counts(radii)
    counts_ref = tree_ref.count_neighbors(tree_ref, radii)
    # scipy counts N self-pairs (distance=0) and both orderings of each pair,
    # so scipy_total = N + 2 * unique_pairs; recover unique_pairs accordingly
    counts_ref = (counts_ref - tree_ref.n) // 2
    assert np.array_equal(counts, counts_ref), "self mismatch"
    counts = tree.counts(radii, cumulative=False)
    assert np.array_equal(counts, np.diff(counts_ref, prepend=0)), "per-shell mismatch"
    counts = tree.counts(radii, other)
    counts_ref = tree_ref.count_neighbors(other_ref, radii)
    assert np.array_equal(counts, counts_ref), "cross mismatch"
    print("passed")


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
    test_pairs(tree, tree_ref, other, other_ref, radius)
    test_counts(tree, tree_ref, other, other_ref, np.linspace(0.05, 0.3, 6))


if __name__ == "__main__":
    main()
