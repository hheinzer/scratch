import numpy as np
from scipy.spatial import KDTree as ScipyKDTree

from kdtree import KDTree

rng = np.random.default_rng(42)

dim = 3
num_points = 100000
leaf_size = 16
num_queries = 1000
num_neighbors = 8
radius = 0.2

points = rng.uniform(-1, 1, (num_points, dim))
queries = rng.uniform(-1, 1, (num_queries, dim))

tree = KDTree(points, leaf_size)
ref = ScipyKDTree(points, leaf_size)

print(f"query ({num_queries} queries, {num_neighbors} neighbors)", end="")
for q in queries:
    idx, dist = tree.query(q, k=num_neighbors)
    dist_ref, idx_ref = ref.query(q, k=num_neighbors)
    idx_ref = idx_ref.astype(np.intc)

    order = np.argsort(idx)
    order_ref = np.argsort(idx_ref)

    assert np.array_equal(idx[order], idx_ref[order_ref]), "index mismatch"
    assert np.allclose(dist[order], dist_ref[order_ref]), "distance mismatch"
print("\t ok")

print(f"query_radius ({num_queries} queries, radius={radius})", end="")
for q in queries:
    idx, dist = tree.query_radius(q, radius, sorted=True)
    idx_ref = ref.query_ball_point(q, radius)

    assert set(idx.tolist()) == set(idx_ref), "index mismatch"
    assert np.all(dist[:-1] <= dist[1:]), "not sorted"
print("\t ok")
