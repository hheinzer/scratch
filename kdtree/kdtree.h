#pragma once

typedef struct kdtree Kdtree;

Kdtree *kdtree_init(const double *point, int dim, int num, int leaf_size);

void kdtree_deinit(Kdtree *self);

void kdtree_query(const Kdtree *self, const double *point, int *index, double *distance, int num);
