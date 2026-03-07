import time
import numpy as np
from scipy.spatial import KDTree as ScipyKDTree

from kdtree import KDTree


def bench(fn):
    t0 = time.perf_counter()
    result = fn()
    return time.perf_counter() - t0, result


def report(label, wtime, wtime_ref):
    print(f"{label:10s}  {wtime:6.3f}s  {wtime_ref:8.3f}s  {wtime_ref / wtime:6.1f}x")


def perf_init(points, leaf_size):
    wtime, tree = bench(lambda: KDTree(points, leaf_size))
    wtime_ref, tree_ref = bench(lambda: ScipyKDTree(points, leaf_size))
    report("init", wtime, wtime_ref)
    return tree, tree_ref


def perf_nearest(tree, tree_ref, queries, k):
    wtime, _ = bench(lambda: tree.nearest(queries, k))
    wtime_ref, _ = bench(lambda: tree_ref.query(queries, k))
    report("nearest", wtime, wtime_ref)


def perf_radius(tree, tree_ref, queries, radius):
    wtime, _ = bench(lambda: tree.radius(queries, radius))
    wtime_ref, _ = bench(lambda: tree_ref.query_ball_point(queries, radius))
    report("radius", wtime, wtime_ref)


def perf_pairs(tree, tree_ref, radius):
    wtime, _ = bench(lambda: tree.pairs(radius))
    wtime_ref, _ = bench(lambda: tree_ref.query_pairs(radius))
    report("pairs", wtime, wtime_ref)


def perf_counts(tree, tree_ref, radii):
    wtime, _ = bench(lambda: tree.counts(radii))
    wtime_ref, _ = bench(lambda: tree_ref.count_neighbors(tree_ref, radii))
    report("counts", wtime, wtime_ref)


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

    print(f"{'':10s}  {'kdtree':>7s}  {'scipy':>9s}  speedup")
    tree, tree_ref = perf_init(points, leaf_size)
    perf_nearest(tree, tree_ref, queries, cap)
    perf_radius(tree, tree_ref, queries, radius)
    perf_pairs(tree, tree_ref, radii[-1])
    perf_counts(tree, tree_ref, radii)


if __name__ == "__main__":
    main()
