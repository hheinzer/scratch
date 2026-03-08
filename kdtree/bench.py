import time
import numpy as np
from scipy.spatial import KDTree as ScipyKDTree

from kdtree import KDTree


def bench(fn):
    t0 = time.perf_counter()
    result = fn()
    return time.perf_counter() - t0, result


def report(label, wtime, wtime_ref):
    print(f"{label:23s}  {wtime:6.3f}s  {wtime_ref:8.3f}s  {wtime_ref / wtime:6.1f}x")


def bench_init(points, leaf_size):
    wtime, tree = bench(lambda: KDTree(points, leaf_size))
    wtime_ref, tree_ref = bench(lambda: ScipyKDTree(points, leaf_size))
    report("init", wtime, wtime_ref)
    return tree, tree_ref


def bench_nearest(tree, tree_ref, queries, k):
    wtime, _ = bench(lambda: tree.nearest(queries, k))
    wtime_ref, _ = bench(lambda: tree_ref.query(queries, k))
    report("nearest", wtime, wtime_ref)


def bench_radius(tree, tree_ref, queries, radius, sorted):
    label = "radius sorted" if sorted else "radius unsorted"
    wtime, _ = bench(lambda: tree.radius(queries, radius, sorted=sorted))
    wtime_ref, _ = bench(lambda: tree_ref.query_ball_point(queries, radius, return_sorted=sorted))
    report(label, wtime, wtime_ref)


def bench_pairs(tree, tree_ref, radius, output_type, other=None, other_ref=None):
    if other is None:
        label = "pairs set" if output_type == "set" else "pairs ndarray"
        wtime, _ = bench(lambda: tree.pairs(radius, output_type=output_type))
        wtime_ref, _ = bench(lambda: tree_ref.query_pairs(radius, output_type=output_type))
    else:
        label = "cross-pairs set" if output_type == "set" else "cross-pairs ndarray"
        wtime, _ = bench(lambda: tree.pairs(radius, other=other, output_type=output_type))
        wtime_ref, _ = bench(lambda: tree_ref.query_ball_tree(other_ref, radius))
    report(label, wtime, wtime_ref)


def bench_counts(tree, tree_ref, radii, cumulative, other=None, other_ref=None):
    if other is None:
        label = "counts cumulative" if cumulative else "counts per shell"
        wtime, _ = bench(lambda: tree.counts(radii, cumulative=cumulative))
        wtime_ref, _ = bench(
            lambda: tree_ref.count_neighbors(tree_ref, radii, cumulative=cumulative)
        )
    else:
        label = "cross-counts cumulative" if cumulative else "cross-counts shell"
        wtime, _ = bench(lambda: tree.counts(radii, other=other, cumulative=cumulative))
        wtime_ref, _ = bench(
            lambda: tree_ref.count_neighbors(other_ref, radii, cumulative=cumulative)
        )
    report(label, wtime, wtime_ref)


def bench_weighted(
    tree, tree_ref, radii, weight_self, other=None, other_ref=None, weight_other=None
):
    if other is None:
        label = "counts weighted"
        wtime, _ = bench(lambda: tree.counts_weighted(radii, weight_self))
        wtime_ref, _ = bench(
            lambda: tree_ref.count_neighbors(tree_ref, radii, weights=(weight_self, weight_self))
        )
    else:
        label = "cross-counts weighted"
        wtime, _ = bench(
            lambda: tree.counts_weighted(radii, (weight_self, weight_other), other=other)
        )
        wtime_ref, _ = bench(
            lambda: tree_ref.count_neighbors(other_ref, radii, weights=(weight_self, weight_other))
        )
    report(label, wtime, wtime_ref)


def main():
    rng = np.random.default_rng(0)

    num_points = 10000000
    num_queries = 1000000
    dim = 3
    leaf_size = 16
    cap = 16
    radius = 0.016
    radii_few = np.linspace(0, radius / 2, 10)
    radii_many = np.linspace(0, radius / 2, 100)

    points = rng.uniform(-1, 1, (num_points, dim))
    queries = rng.uniform(-1, 1, (num_queries, dim))

    other = KDTree(queries, leaf_size)
    other_ref = ScipyKDTree(queries, leaf_size)

    weight_self = rng.uniform(size=num_points)
    weight_other = rng.uniform(size=num_queries)

    print(f"{'':23s}  {'kdtree':>7s}  {'scipy':>9s}  speedup")
    tree, tree_ref = bench_init(points, leaf_size)
    bench_nearest(tree, tree_ref, queries, cap)
    bench_radius(tree, tree_ref, queries, radius, True)
    bench_radius(tree, tree_ref, queries, radius, False)
    bench_pairs(tree, tree_ref, radii_few[-1], "set")
    bench_pairs(tree, tree_ref, radii_few[-1], "ndarray")
    bench_pairs(tree, tree_ref, radii_few[-1], "set", other, other_ref)
    bench_pairs(tree, tree_ref, radii_few[-1], "ndarray", other, other_ref)
    bench_counts(tree, tree_ref, radii_few, True)
    bench_counts(tree, tree_ref, radii_many, False)
    bench_counts(tree, tree_ref, radii_few, True, other, other_ref)
    bench_counts(tree, tree_ref, radii_many, False, other, other_ref)
    bench_weighted(tree, tree_ref, radii_few, weight_self)
    bench_weighted(tree, tree_ref, radii_few, weight_self, other, other_ref, weight_other)


if __name__ == "__main__":
    main()
