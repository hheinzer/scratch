#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "alloc.h"
#include "kdtree.h"

int main(int argc, char **argv)
{
    srand((unsigned)time(0));

    int dim = (argc > 1) ? (int)strtol(argv[1], 0, 10) : 3;
    int num_points = (argc > 2) ? (int)strtol(argv[2], 0, 10) : 100000;
    int leaf_size = (argc > 3) ? (int)strtol(argv[3], 0, 10) : 10;
    int num_queries = (argc > 4) ? (int)strtol(argv[4], 0, 10) : 1000;
    int num_neighbors = (argc > 5) ? (int)strtol(argv[5], 0, 10) : 1;

    double (*point)[dim] = alloc(num_points, sizeof(*point));
    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < dim; j++) {
            point[i][j] = (2 * (rand() / (double)RAND_MAX)) - 1;
        }
    }

    int *query = alloc(num_queries, sizeof(*query));
    for (int i = 0; i < num_queries; i++) {
        query[i] = rand() % num_points;
    }

    clock_t beg_init = clock();
    Kdtree *tree = kdtree_init(*point, dim, num_points, leaf_size);
    clock_t end_init = clock();

    clock_t beg_query = clock();
    for (int i = 0; i < num_queries; i++) {
        int index[num_neighbors];
        double distance[num_neighbors];
        int num = kdtree_query(tree, point[query[i]], index, distance, num_neighbors);
        assert(num == num_neighbors && index[0] == query[i] && distance[0] <= DBL_EPSILON);
    }
    clock_t end_query = clock();

    double time_init = (double)(end_init - beg_init) / CLOCKS_PER_SEC;
    double time_query = (double)(end_query - beg_query) / CLOCKS_PER_SEC;
    printf(
        "dim: %d, "
        "points: %d, "
        "leaf_size: %d, "
        "queries: %d, "
        "neighbors: %d, "
        "init: %g, "
        "query: %g\n",
        dim, num_points, leaf_size, num_queries, num_neighbors, time_init, time_query);

    kdtree_deinit(tree);

    free(point);
    free(query);
}
