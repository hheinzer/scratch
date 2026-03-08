#pragma once

typedef struct kdtree Kdtree;

// Build a k-d tree from `num` points of dimension `dim`. `leaf_size` controls the number of points
// stored in leaf nodes (0 uses a default). `periodic[i]` is the periodic box length in dimension i
// (pass 0 for non-periodic).
Kdtree *kdtree_init(const double *point, int num, int dim, int leaf_size, const double *periodic);

// Free all memory associated with the tree.
void kdtree_deinit(Kdtree *self);

// Find the nearest neighbors of `num` points, writing up to `cap` results per point to `index` and
// `distance` in row-major order. If `sorted`, results are in ascending order per point. Returns the
// number of results per point.
int kdtree_nearest(const Kdtree *self, const double *point, int *index, double *distance, int num,
                   int cap, int sorted);

// Find all points within `radius` of `num` points. Allocates and writes results in CSR format:
// `(*offset)[i]` to `(*offset)[i+1]` indexes into `*index` and `*distance` for query `i`. If
// `sorted`, results per point are in ascending order. Caller must free `*offset`, `*index`, and
// `*distance`. Returns the total number of results.
int kdtree_radius(const Kdtree *self, const double *point, double radius, int **offset, int **index,
                  double **distance, int num, int sorted);

// Find all pairs within `radius` using a dual-tree traversal. If `!other`, finds unique self-pairs;
// otherwise finds pairs between `self` and `other` (must have same dimension). Pairs are written to
// `*pair`. Caller must free. Returns total pair count.
int kdtree_pairs(const Kdtree *self, const Kdtree *other, double radius, int (**pair)[2]);

// For each `radius[k]` (must be sorted in ascending order), `count[k]` is the number of unique
// pairs with distance at most `radius[k]`. If `!other`, counts self-pairs; otherwise counts pairs
// between `self` and `other` (must have same dimension). If not `cumulative`, counts are per-shell.
void kdtree_counts(const Kdtree *self, const Kdtree *other, const double *radius, long *count,
                   int num, int cumulative);

// Like `kdtree_counts`, but each pair (i, j) contributes `weight[i] * weight[j]` instead of 1. For
// cross-pairs, `weight_self` applies to `self` and `weight_other` to `other`; for self-pairs,
// `weight_other` is unused.
void kdtree_weighted(const Kdtree *self, const Kdtree *other, const double *weight_self,
                     const double *weight_other, const double *radius, double *count, int num,
                     int cumulative);

// Dump the tree structure to disk.
void kdtree_dump(const Kdtree *self, const char *fname);
