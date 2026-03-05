#pragma once

typedef struct kdtree Kdtree;

// Build a k-d tree from `num` points of dimension `dim`. `leaf_size` controls the number of points
// stored in leaf nodes (0 uses a default).
Kdtree *kdtree_init(const double *point, int dim, int num, int leaf_size);

// Free all memory associated with the tree.
void kdtree_deinit(Kdtree *self);

// Find the nearest neighbors of `point`, writing up to `cap` results to `index` and `distance`.
// Returns the number of results.
int kdtree_query(const Kdtree *self, const double *point, int *index, double *distance, int cap);

// Find all points within `radius` of `point`, writing up to `cap` results to `index` and
// `distance`. Returns the total number of points found (can be larger than `cap`).
int kdtree_query_radius(const Kdtree *self, const double *point, double radius, int *index,
                        double *distance, int cap);
