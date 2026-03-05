#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "kdtree.h"

static int contains(const int *index, int idx, int num)
{
    for (int i = 0; i < num; i++) {
        if (index[i] == idx) {
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    srand((unsigned)time(0));

    int dim = (argc > 1) ? (int)strtol(argv[1], 0, 10) : 3;
    int num_points = (argc > 2) ? (int)strtol(argv[2], 0, 10) : 1 << 19;
    int leaf_size = (argc > 3) ? (int)strtol(argv[3], 0, 10) : 16;
    int num_queries = (argc > 4) ? (int)strtol(argv[4], 0, 10) : 1 << 16;
    int num_neighbors = (argc > 5) ? (int)strtol(argv[5], 0, 10) : 8;

    double (*point)[dim] = malloc((size_t)num_points * sizeof(*point));
    assert(point);
    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < dim; j++) {
            point[i][j] = (2 * (rand() / (double)RAND_MAX)) - 1;
        }
    }

    int *query = malloc((size_t)num_queries * sizeof(*query));
    assert(query);
    for (int i = 0; i < num_queries; i++) {
        query[i] = rand() % num_points;
    }

    clock_t beg_init = clock();
    Kdtree *tree = kdtree_init(*point, num_points, dim, leaf_size);
    clock_t end_init = clock();

    clock_t beg_query = clock();
    for (int i = 0; i < num_queries; i++) {
        int index[num_neighbors];
        double distance[num_neighbors];
        int num = kdtree_query(tree, point[query[i]], index, distance, num_neighbors);
        assert(num == num_neighbors && contains(index, query[i], num_neighbors));
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
