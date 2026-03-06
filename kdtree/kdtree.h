#pragma once

typedef struct kdtree Kdtree;

// Build a k-d tree from `num` points of dimension `dim`. `leaf_size` controls the number of points
// stored in leaf nodes (0 uses a default).
Kdtree *kdtree_init(const double *point, int num, int dim, int leaf_size);

// Free all memory associated with the tree.
void kdtree_deinit(Kdtree *self);

// Find the nearest neighbors of `point`, writing up to `cap` results to `index` and `distance`.
// If `sorted`, results are in ascending order. Returns the number of results.
int kdtree_nearest(const Kdtree *self, const double *point, int *index, double *distance, int cap,
                   int sorted);

// Find all points within `radius` of `point`, writing up to `cap` results to `index` and
// `distance`. If `sorted`, results are in ascending order. Returns the total number of points found
// (can be larger than `cap`).
int kdtree_radius(const Kdtree *self, const double *point, double radius, int *index,
                  double *distance, int cap, int sorted);

// Find all pairs of points within `radius` of each other using a dual-tree traversal. Unique pairs
// are written to `*pair`. Caller must free. Returns total pair count.
int kdtree_pairs(const Kdtree *self, double radius, int (**pair)[2]);

// Find all pairs (self, other) within `radius` using a dual-tree traversal. Both trees must have
// the same dimension. Pairs are written to `*pair`. Caller must free. Returns total pair count.
int kdtree_cross(const Kdtree *self, const Kdtree *other, double radius, int (**pair)[2]);

// For each `radius[k]` (must be sorted in ascending order), `count[k]` is the number of unique
// pairs with distance at most `radius[k]`. If not `cumulative`, counts are per-shell.
void kdtree_counts(const Kdtree *self, const double *radius, long *count, int num, int cumulative);

// Dump the tree structure to disk.
void kdtree_dump(const Kdtree *self, const char *fname);
