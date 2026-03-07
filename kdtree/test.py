import numpy as np
from scipy.spatial import KDTree as ScipyKDTree

from kdtree import KDTree


def check(label):
    print(f"{label:23s}", end=" ")


def test_nearest(tree, tree_ref, queries, num):
    check("nearest")
    idx, dist = tree.nearest(queries, num, sorted=True)
    dist_ref, idx_ref = tree_ref.query(queries, num)
    assert np.array_equal(idx, idx_ref), "index mismatch"
    assert np.allclose(dist, dist_ref), "distance mismatch"
    print("passed")


def test_radius(tree, tree_ref, queries, radius, sorted):
    check("radius sorted" if sorted else "radius unsorted")
    results = tree.radius(queries, radius, sorted=sorted)
    results_ref = tree_ref.query_ball_point(queries, radius, return_sorted=sorted)
    for (idx_i, dist_i), idx_ref in zip(results, results_ref):
        assert set(idx_i.tolist()) == set(idx_ref), "index mismatch"
        if sorted:
            assert np.all(dist_i[:-1] <= dist_i[1:]), "not sorted"
    print("passed")


def test_pairs(tree, tree_ref, radius, output_type, other=None, other_ref=None):
    if other is None:
        check("pairs set" if output_type == "set" else "pairs ndarray")
        result = tree.pairs(radius, output_type=output_type)
        pairs_ref = tree_ref.query_pairs(radius)
    else:
        check("cross-pairs set" if output_type == "set" else "cross-pairs ndarray")
        result = tree.pairs(radius, other=other, output_type=output_type)
        list_ref = tree_ref.query_ball_tree(other_ref, radius)
        pairs_ref = {(i, j) for i, js in enumerate(list_ref) for j in js}
    pairs = result if output_type == "set" else set(map(tuple, result.tolist()))
    assert pairs == pairs_ref, "pairs mismatch"
    print("passed")


def test_counts(tree, tree_ref, radii, cumulative, other=None, other_ref=None):
    if other is None:
        check("counts cumulative" if cumulative else "counts per shell")
        counts = tree.counts(radii, cumulative=cumulative)
        counts_ref = tree_ref.count_neighbors(tree_ref, radii, cumulative=True)
        counts_ref = (counts_ref - tree_ref.n) // 2
        if not cumulative:
            counts_ref = np.diff(counts_ref, prepend=0)
    else:
        check("cross-counts cumulative" if cumulative else "cross-counts per shell")
        counts = tree.counts(radii, other=other, cumulative=cumulative)
        counts_ref = tree_ref.count_neighbors(other_ref, radii, cumulative=cumulative)
    assert np.array_equal(counts, counts_ref), "counts mismatch"
    print("passed")


def main():
    rng = np.random.default_rng(42)

    num_points = 10000
    num_queries = 100
    dim = 3
    leaf_size = 16
    cap = 8
    radius = 0.2
    radii = np.linspace(0.05, 0.3, 6)

    points = rng.uniform(-1, 1, (num_points, dim))
    queries = rng.uniform(-1, 1, (num_queries, dim))

    tree = KDTree(points, leaf_size)
    tree_ref = ScipyKDTree(points, leaf_size)

    other = KDTree(queries, leaf_size)
    other_ref = ScipyKDTree(queries, leaf_size)

    test_nearest(tree, tree_ref, queries, cap)
    test_radius(tree, tree_ref, queries, radius, sorted=True)
    test_radius(tree, tree_ref, queries, radius, sorted=False)
    test_pairs(tree, tree_ref, radius, output_type="set")
    test_pairs(tree, tree_ref, radius, output_type="ndarray")
    test_pairs(tree, tree_ref, radius, output_type="set", other=other, other_ref=other_ref)
    test_pairs(tree, tree_ref, radius, output_type="ndarray", other=other, other_ref=other_ref)
    test_counts(tree, tree_ref, radii, cumulative=True)
    test_counts(tree, tree_ref, radii, cumulative=False)
    test_counts(tree, tree_ref, radii, cumulative=True, other=other, other_ref=other_ref)
    test_counts(tree, tree_ref, radii, cumulative=False, other=other, other_ref=other_ref)


if __name__ == "__main__":
    main()
