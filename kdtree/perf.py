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


def perf_init(points, leaf_size):
    wtime, tree = bench(lambda: KDTree(points, leaf_size))
    wtime_ref, tree_ref = bench(lambda: ScipyKDTree(points, leaf_size))
    report("init", wtime, wtime_ref)
    return tree, tree_ref


def perf_nearest(tree, tree_ref, queries, k):
    wtime, _ = bench(lambda: tree.nearest(queries, k))
    wtime_ref, _ = bench(lambda: tree_ref.query(queries, k))
    report("nearest", wtime, wtime_ref)


def perf_radius(tree, tree_ref, queries, radius, sorted):
    label = "radius sorted" if sorted else "radius unsorted"
    wtime, _ = bench(lambda: tree.radius(queries, radius, sorted=sorted))
    wtime_ref, _ = bench(lambda: tree_ref.query_ball_point(queries, radius, return_sorted=sorted))
    report(label, wtime, wtime_ref)


def perf_pairs(tree, tree_ref, radius, output_type, other=None, other_ref=None):
    if other is None:
        label = "pairs set" if output_type == "set" else "pairs ndarray"
        wtime, _ = bench(lambda: tree.pairs(radius, output_type=output_type))
        wtime_ref, _ = bench(lambda: tree_ref.query_pairs(radius, output_type=output_type))
    else:
        label = "cross-pairs set" if output_type == "set" else "cross-pairs ndarray"
        wtime, _ = bench(lambda: tree.pairs(radius, other=other, output_type=output_type))
        wtime_ref, _ = bench(lambda: tree_ref.query_ball_tree(other_ref, radius))
    report(label, wtime, wtime_ref)


def perf_counts(tree, tree_ref, radii, cumulative, other=None, other_ref=None):
    if other is None:
        label = "counts cumulative" if cumulative else "counts per shell"
        wtime, _ = bench(lambda: tree.counts(radii, cumulative=cumulative))
        wtime_ref, _ = bench(lambda: tree_ref.count_neighbors(tree_ref, radii, cumulative=cumulative))
    else:
        label = "cross-counts cumulative" if cumulative else "cross-counts shell"
        wtime, _ = bench(lambda: tree.counts(radii, other=other, cumulative=cumulative))
        wtime_ref, _ = bench(lambda: tree_ref.count_neighbors(other_ref, radii, cumulative=cumulative))
    report(label, wtime, wtime_ref)


def main():
    rng = np.random.default_rng(0)

    num_points = 10000000
    num_queries = 1000000
    dim = 3
    leaf_size = 16
    cap = 16
    radius = 0.016
    radii = np.linspace(0, radius / 2, 10)

    points = rng.uniform(-1, 1, (num_points, dim))
    queries = rng.uniform(-1, 1, (num_queries, dim))

    other = KDTree(queries, leaf_size)
    other_ref = ScipyKDTree(queries, leaf_size)

    print(f"{'':23s}  {'kdtree':>7s}  {'scipy':>9s}  speedup")
    tree, tree_ref = perf_init(points, leaf_size)
    perf_nearest(tree, tree_ref, queries, cap)
    perf_radius(tree, tree_ref, queries, radius, sorted=True)
    perf_radius(tree, tree_ref, queries, radius, sorted=False)
    perf_pairs(tree, tree_ref, radii[-1], output_type="set")
    perf_pairs(tree, tree_ref, radii[-1], output_type="ndarray")
    perf_pairs(tree, tree_ref, radii[-1], output_type="set", other=other, other_ref=other_ref)
    perf_pairs(tree, tree_ref, radii[-1], output_type="ndarray", other=other, other_ref=other_ref)
    perf_counts(tree, tree_ref, radii, cumulative=True)
    perf_counts(tree, tree_ref, radii, cumulative=False)
    perf_counts(tree, tree_ref, radii, cumulative=True, other=other, other_ref=other_ref)
    perf_counts(tree, tree_ref, radii, cumulative=False, other=other, other_ref=other_ref)


if __name__ == "__main__":
    main()
