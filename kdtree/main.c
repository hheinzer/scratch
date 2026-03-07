#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "kdtree.h"

static double get_time(void)
{
    return (double)clock() / CLOCKS_PER_SEC;
}

static void report(const char *label, double wtime)
{
    printf("%-23s  %6.3fs\n", label, wtime);
}

static double random_uniform(void)
{
    return (2 * ((double)rand() / RAND_MAX)) - 1;
}

static Kdtree *bench_init(const double *point, int num, int dim, int leaf_size)
{
    double beg = get_time();
    Kdtree *tree = kdtree_init(point, num, dim, leaf_size);
    report("init", get_time() - beg);
    return tree;
}

static void bench_nearest(const Kdtree *tree, const double *query, int num, int cap)
{
    int (*index)[cap] = malloc(num * sizeof(*index));
    assert(index);
    double (*distance)[cap] = malloc(num * sizeof(*distance));
    assert(distance);

    double beg = get_time();
    kdtree_nearest(tree, query, *index, *distance, num, cap, 1);
    report("nearest", get_time() - beg);

    free(index);
    free(distance);
}

static void bench_radius(const Kdtree *tree, const double *query, int num, double radius,
                         int sorted)
{
    int *offset;
    int *index;
    double *distance;

    double beg = get_time();
    kdtree_radius(tree, query, radius, &offset, &index, &distance, num, sorted);
    report(sorted ? "radius sorted" : "radius unsorted", get_time() - beg);

    free(offset);
    free(index);
    free(distance);
}

static void bench_pairs(const Kdtree *tree, const Kdtree *other, double radius)
{
    int (*pair)[2] = 0;

    double beg = get_time();
    kdtree_pairs(tree, other, radius, &pair);
    report(other ? "cross-pairs" : "pairs", get_time() - beg);

    free(pair);
}

static void bench_counts(const Kdtree *tree, const Kdtree *other, const double *radii, int num,
                         int cumulative)
{
    const char *label;
    if (!other && cumulative) {
        label = "counts cumulative";
    }
    else if (!other) {
        label = "counts per shell";
    }
    else if (cumulative) {
        label = "cross-counts cumulative";
    }
    else {
        label = "cross-counts per shell";
    }

    long *count = malloc(num * sizeof(*count));
    assert(count);

    double beg = get_time();
    kdtree_counts(tree, other, radii, count, num, cumulative);
    report(label, get_time() - beg);

    free(count);
}

int main(int argc, char **argv)
{
    srand(0);

    int num_points = (argc > 1) ? (int)strtol(argv[1], 0, 10) : 10000000;
    int num_queries = (argc > 2) ? (int)strtol(argv[2], 0, 10) : 1000000;
    int dim = (argc > 3) ? (int)strtol(argv[3], 0, 10) : 3;
    int leaf_size = (argc > 4) ? (int)strtol(argv[4], 0, 10) : 16;
    int cap = (argc > 5) ? (int)strtol(argv[5], 0, 10) : 16;
    double radius = (argc > 6) ? strtod(argv[6], 0) : 0.016;
    int num_radii = (argc > 7) ? (int)strtol(argv[7], 0, 10) : 10;

    double *radii = malloc(num_radii * sizeof(*radii));
    assert(radii);
    for (int i = 0; i < num_radii; i++) {
        radii[i] = i * (radius / 2) / (num_radii - 1);
    }

    double (*point)[dim] = malloc(num_points * sizeof(*point));
    assert(point);
    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < dim; j++) {
            point[i][j] = random_uniform();
        }
    }

    double (*query)[dim] = malloc(num_queries * sizeof(*query));
    assert(query);
    for (int i = 0; i < num_queries; i++) {
        for (int j = 0; j < dim; j++) {
            query[i][j] = random_uniform();
        }
    }

    Kdtree *other = kdtree_init(*query, num_queries, dim, leaf_size);

    Kdtree *tree = bench_init(*point, num_points, dim, leaf_size);
    bench_nearest(tree, *query, num_queries, cap);
    bench_radius(tree, *query, num_queries, radius, 1);
    bench_radius(tree, *query, num_queries, radius, 0);
    bench_pairs(tree, 0, radii[num_radii - 1]);
    bench_pairs(tree, other, radii[num_radii - 1]);
    bench_counts(tree, 0, radii, num_radii, 1);
    bench_counts(tree, 0, radii, num_radii, 0);
    bench_counts(tree, other, radii, num_radii, 1);
    bench_counts(tree, other, radii, num_radii, 0);

    kdtree_deinit(tree);
    kdtree_deinit(other);

    free(radii);
    free(point);
    free(query);
}
