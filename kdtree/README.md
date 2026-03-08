# kdtree

A kd-tree written in C99. Supports nearest neighbor search, radius search, and dual-tree pair
enumeration. A Python wrapper is provided in `kdtree.py`.

## Usage

Compile `kdtree.c` alongside your project and include `kdtree.h`. See `Makefile` for recommended
compilation flags.

Example usage is shown in `main.c`, `test.py` contains tests against SciPy's kd-tree, `bench.py` is
a benchmark against SciPy's kd-tree, and `plot.py` visualizes the tree structure and search
operations.

## Methods

**`kdtree_init`** Build a tree from a set of points stored in row-major order. The leaf size
controls how many points are stored in leaf nodes (0 uses the default of 16). Only Euclidean
distance is supported. Optionally accepts per-dimension box lengths to enable periodic boundary
conditions (minimum image convention); points do not need to be pre-wrapped. The tree holds
pointers to the input data and does not copy it; all inputs must remain valid for the lifetime of
the tree.

**`kdtree_deinit`** Free all memory associated with the tree.

**`kdtree_nearest`** Find the k nearest neighbors of a query point. Writes indices and distances to
caller-allocated arrays of size k. Results can optionally be sorted in ascending distance order.
Returns the number of results found.

**`kdtree_radius`** Find all points within a given radius of a set of query points. Allocates and
writes results in CSR format; caller must free. Results can optionally be sorted in ascending
distance order. Returns the total number of results.

**`kdtree_pairs`** Find all pairs within a given radius using dual-tree traversal. Finds unique
self-pairs, or cross-pairs between two trees of the same dimension if a second tree is provided.
Allocates and writes results to a caller-owned pointer; caller must free. Returns the pair count.

**`kdtree_counts`** Count pairs within a series of radii (must be sorted ascending). For each
radius, writes the number of pairs whose distance does not exceed it. Counts can be cumulative or
per shell (between consecutive radii). Supports self-pairs and cross-pairs.

**`kdtree_weighted`** Like `kdtree_counts`, but each pair contributes the product of its two weights
instead of 1. For cross-pairs, separate weight arrays are supplied for each tree.

## Performance

Benchmarked against `scipy.spatial.KDTree` on 10M points in 3D with a leaf size of 16. Nearest
neighbor search and radius search use 1M query points. Pair operations use a radius of 0.008;
cumulative counts use 10 radii and per-shell counts use 100 radii up to 0.008.

| Operation               |  kdtree |   scipy | speedup |
|:------------------------|--------:|--------:|--------:|
| init                    |  9.915s |  9.663s |    1.0x |
| nearest                 | 10.382s | 10.985s |    1.1x |
| radius sorted           | 10.935s | 13.377s |    1.2x |
| radius unsorted         | 12.060s | 12.631s |    1.0x |
| pairs set               | 11.925s | 15.915s |    1.3x |
| pairs ndarray           |  7.472s | 10.638s |    1.4x |
| cross-pairs set         |  6.508s |  9.173s |    1.4x |
| cross-pairs ndarray     |  5.826s |  9.230s |    1.6x |
| counts cumulative       | 12.528s | 44.641s |    3.6x |
| counts per shell        | 17.435s | 44.921s |    2.6x |
| cross-counts cumulative |  8.414s | 13.791s |    1.6x |
| cross-counts per shell  | 11.834s | 14.823s |    1.3x |
| counts weighted         | 13.078s | 43.649s |    3.3x |
| cross-counts weighted   |  8.546s | 13.862s |    1.6x |

Run `make bench` to reproduce. Pair and count operations benefit most from dual-tree pruning.

## Implementation notes

- Split axis is chosen as the dimension with the largest coordinate spread
- Median split is found with quickselect, giving O(n log n) build time
- Nearest neighbor search uses a max-heap for large counts and an insertion-sorted array otherwise
- Pair search and counting use dual-tree traversal, pruning node pairs whose bounding boxes are
  farther apart than the search radius
