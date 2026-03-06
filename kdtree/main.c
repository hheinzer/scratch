#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "kdtree.h"

int main(int argc, char **argv)
{
    srand((unsigned)time(0));

    int num_points = (argc > 1) ? (int)strtol(argv[1], 0, 10) : 1 << 19;
    int num_queries = (argc > 2) ? (int)strtol(argv[2], 0, 10) : 1 << 16;
    int dim = (argc > 3) ? (int)strtol(argv[3], 0, 10) : 3;
    int leaf_size = (argc > 4) ? (int)strtol(argv[4], 0, 10) : 16;
    int cap = (argc > 5) ? (int)strtol(argv[5], 0, 10) : 8;

    double (*point)[dim] = malloc((size_t)num_points * sizeof(*point));
    assert(point);
    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < dim; j++) {
            point[i][j] = (2 * (rand() / (double)RAND_MAX)) - 1;
        }
    }

    int *query_index = malloc((size_t)num_queries * sizeof(*query_index));
    assert(query_index);
    for (int i = 0; i < num_queries; i++) {
        query_index[i] = rand() % num_points;
    }

    double (*query)[dim] = malloc((size_t)num_queries * sizeof(*query));
    assert(query);
    for (int i = 0; i < num_queries; i++) {
        for (int j = 0; j < dim; j++) {
            query[i][j] = point[query_index[i]][j];
        }
    }

    int *index = malloc((size_t)num_queries * (size_t)cap * sizeof(*index));
    assert(index);
    double *distance = malloc((size_t)num_queries * (size_t)cap * sizeof(*distance));
    assert(distance);

    clock_t beg_init = clock();
    Kdtree *tree = kdtree_init(*point, num_points, dim, leaf_size);
    clock_t end_init = clock();

    clock_t beg_query = clock();
    int found = kdtree_nearest(tree, *query, index, distance, num_queries, cap, 1);
    clock_t end_query = clock();

    assert(found == cap);
    for (int i = 0; i < num_queries; i++) {
        assert(index[(long)i * cap] == query_index[i]);
    }

    double time_init = (double)(end_init - beg_init) / CLOCKS_PER_SEC;
    double time_query = (double)(end_query - beg_query) / CLOCKS_PER_SEC;
    printf(
        "points: %d, "
        "queries: %d, "
        "dim: %d, "
        "leaf_size: %d, "
        "cap: %d, "
        "init: %g, "
        "query: %g\n",
        num_points, num_queries, dim, leaf_size, cap, time_init, time_query);

    kdtree_deinit(tree);

    free(point);
    free(query_index);
    free(query);
    free(index);
    free(distance);
}
