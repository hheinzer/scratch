#pragma once

typedef struct kdtree Kdtree;

// Build a k-d tree from `num` points of dimension `dim`. `leaf_size` controls the number of points
// stored in leaf nodes (0 uses a default).
Kdtree *kdtree_init(const double *point, int num, int dim, int leaf_size);

// Free all memory associated with the tree.
void kdtree_deinit(Kdtree *self);

// Find the nearest neighbors of `point`, writing up to `cap` results to `index` and `distance`
// in ascending order. Returns the number of results.
int kdtree_nearest(const Kdtree *self, const double *point, int *index, double *distance, int cap);

// Find all points within `radius` of `point`, writing up to `cap` results to `index` and
// `distance`. Returns the total number of points found (can be larger than `cap`).
int kdtree_radius(const Kdtree *self, const double *point, double radius, int *index,
                  double *distance, int cap, int sorted);

// Find all pairs of points within `radius` of each other using a dual-tree traversal. Unique pairs
// are written to `*pair` as int[2] elements. Returns the total number of pairs.
int kdtree_pairs(const Kdtree *self, double radius, int (**pair)[2]);
