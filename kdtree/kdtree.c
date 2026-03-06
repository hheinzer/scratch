#include "kdtree.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { LEAF_SIZE = 16, HEAP_THRESHOLD = 32 };

typedef struct {
    int num;       // number of points in subtree
    int axis;      // split axis; -1 => leaf, otherwise node
    double value;  // split value
    union {
        struct {
            int beg, end;  // slice of index array
        } leaf;
        struct {
            int left, right;  // child node indices
        } node;
    } child;
} Node;

typedef struct {
    double min, max;
} Rect;

struct kdtree {
    int num_points;
    int num_nodes;
    int dim;
    int leaf_size;
    const double *point;
    int *index;
    Node *node;
    Rect *bbox;
};

static int compute_size(const Kdtree *self, int num)
{
    if (num <= self->leaf_size) {
        return 1;
    }
    int left = num / 2;
    int right = num - left;
    return 1 + compute_size(self, left) + compute_size(self, right);
}

static Rect *get_bbox(const Kdtree *self, int idx)
{
    return &self->bbox[(long)idx * self->dim];
}

static double get_value(const Kdtree *self, int pos, int axis)
{
    return self->point[((long)self->index[pos] * self->dim) + axis];
}

static int split_axis(const Kdtree *self, int beg, int end, Rect *bbox)
{
    int best_axis = 0;
    double best_spread = -1;
    for (int i = 0; i < self->dim; i++) {
        double min = get_value(self, beg, i);
        double max = min;
        for (int j = beg + 1; j < end; j++) {
            double value = get_value(self, j, i);
            if (value < min) {
                min = value;
            }
            else if (value > max) {
                max = value;
            }
        }
        bbox[i].min = min;
        bbox[i].max = max;
        if (max - min > best_spread) {
            best_spread = max - min;
            best_axis = i;
        }
    }
    return best_axis;
}

static void swap_int(int *lhs, int *rhs)
{
    int swap = *lhs;
    *lhs = *rhs;
    *rhs = swap;
}

static void quickselect(Kdtree *self, int beg, int end, int mid, int axis)
{
    int min = beg;
    int max = end - 1;
    while (min < max) {
        double pivot = get_value(self, (min + max) / 2, axis);
        int left = min;
        int right = max;
        while (left <= right) {
            while (get_value(self, left, axis) < pivot) {
                left += 1;
            }
            while (get_value(self, right, axis) > pivot) {
                right -= 1;
            }
            if (left <= right) {
                swap_int(&self->index[left], &self->index[right]);
                left += 1;
                right -= 1;
            }
        }
        if (right < mid) {
            min = left;
        }
        if (left > mid) {
            max = right;
        }
        if (right < mid && left > mid) {
            break;
        }
    }
}

static void build(Kdtree *self, int *next, int beg, int end)
{
    int idx = (*next)++;
    int num = end - beg;

    self->node[idx].num = num;
    Rect *bbox = get_bbox(self, idx);

    if (num <= self->leaf_size) {
        split_axis(self, beg, end, bbox);
        self->node[idx].axis = -1;
        self->node[idx].child.leaf.beg = beg;
        self->node[idx].child.leaf.end = end;
        return;
    }

    int mid = (beg + end) / 2;
    int axis = split_axis(self, beg, end, bbox);
    quickselect(self, beg, end, mid, axis);

    self->node[idx].axis = axis;
    self->node[idx].value = get_value(self, mid, axis);

    self->node[idx].child.node.left = *next;
    build(self, next, beg, mid);

    self->node[idx].child.node.right = *next;
    build(self, next, mid, end);
}

Kdtree *kdtree_init(const double *point, int num, int dim, int leaf_size)
{
    assert(point && num > 0 && dim > 0 && leaf_size >= 0);

    Kdtree *self = malloc(sizeof(*self));
    assert(self);

    self->num_points = num;
    self->dim = dim;
    self->leaf_size = leaf_size ? leaf_size : LEAF_SIZE;
    self->point = point;

    self->index = malloc((size_t)num * sizeof(*self->index));
    assert(self->index);
    for (int i = 0; i < num; i++) {
        self->index[i] = i;
    }

    self->num_nodes = compute_size(self, num);
    self->node = malloc((size_t)self->num_nodes * sizeof(*self->node));
    assert(self->node);

    self->bbox = malloc((size_t)self->num_nodes * (size_t)dim * sizeof(*self->bbox));
    assert(self->bbox);

    int next = 0;
    build(self, &next, 0, num);
    assert(next == self->num_nodes);

    return self;
}

void kdtree_deinit(Kdtree *self)
{
    assert(self);
    free(self->index);
    free(self->node);
    free(self->bbox);
    free(self);
}

static void sorted_push(int *index, double *distance2, int *num, int cap, int idx, double dist2)
{
    if (*num == cap && dist2 >= distance2[0]) {
        return;
    }
    if (*num < cap) {
        int pos = *num;
        while (pos > 0 && distance2[pos - 1] < dist2) {
            index[pos] = index[pos - 1];
            distance2[pos] = distance2[pos - 1];
            pos -= 1;
        }
        index[pos] = idx;
        distance2[pos] = dist2;
        *num += 1;
    }
    else {
        int pos = 0;
        while (pos < cap - 1 && distance2[pos + 1] >= dist2) {
            index[pos] = index[pos + 1];
            distance2[pos] = distance2[pos + 1];
            pos += 1;
        }
        index[pos] = idx;
        distance2[pos] = dist2;
    }
}

static void swap_double(double *lhs, double *rhs)
{
    double swap = *lhs;
    *lhs = *rhs;
    *rhs = swap;
}

static void sift_up(int *index, double *distance2, int pos)
{
    while (pos > 0) {
        int parent = (pos - 1) / 2;
        if (distance2[parent] >= distance2[pos]) {
            break;
        }
        swap_int(&index[parent], &index[pos]);
        swap_double(&distance2[parent], &distance2[pos]);
        pos = parent;
    }
}

static void sift_down(int *index, double *distance2, int pos, int num)
{
    while (1) {
        int left = (2 * pos) + 1;
        int right = (2 * pos) + 2;
        int largest = pos;
        if (left < num && distance2[left] > distance2[largest]) {
            largest = left;
        }
        if (right < num && distance2[right] > distance2[largest]) {
            largest = right;
        }
        if (largest == pos) {
            break;
        }
        swap_int(&index[largest], &index[pos]);
        swap_double(&distance2[largest], &distance2[pos]);
        pos = largest;
    }
}

static void heap_push(int *index, double *distance2, int *num, int cap, int idx, double dist2)
{
    if (*num == cap && dist2 >= distance2[0]) {
        return;
    }
    if (*num < cap) {
        index[*num] = idx;
        distance2[*num] = dist2;
        sift_up(index, distance2, *num);
        *num += 1;
    }
    else {
        index[0] = idx;
        distance2[0] = dist2;
        sift_down(index, distance2, 0, cap);
    }
}

static double bbox_dist2(const Kdtree *self, const double *point, const Rect *bbox)
{
    double dist2 = 0;
    for (int i = 0; i < self->dim; i++) {
        if (point[i] < bbox[i].min) {
            double diff = bbox[i].min - point[i];
            dist2 += diff * diff;
        }
        else if (point[i] > bbox[i].max) {
            double diff = point[i] - bbox[i].max;
            dist2 += diff * diff;
        }
    }
    return dist2;
}

static void search(const Kdtree *self, int idx, const double *point, int *index, double *distance2,
                   int *num, int cap)
{
    const Node *node = &self->node[idx];

    if (node->axis == -1) {
        for (int i = node->child.leaf.beg; i < node->child.leaf.end; i++) {
            double dist2 = 0;
            for (int j = 0; j < self->dim; j++) {
                double diff = point[j] - get_value(self, i, j);
                dist2 += diff * diff;
            }
            if (cap <= HEAP_THRESHOLD) {
                sorted_push(index, distance2, num, cap, self->index[i], dist2);
            }
            else {
                heap_push(index, distance2, num, cap, self->index[i], dist2);
            }
        }
        return;
    }

    int near;
    int far;
    if (point[node->axis] <= node->value) {
        near = node->child.node.left;
        far = node->child.node.right;
    }
    else {
        near = node->child.node.right;
        far = node->child.node.left;
    }

    search(self, near, point, index, distance2, num, cap);

    if (*num < cap || bbox_dist2(self, point, get_bbox(self, far)) <= distance2[0]) {
        search(self, far, point, index, distance2, num, cap);
    }
}

static void sort_results(int *index, double *distance, int num)
{
    while (num > 1) {
        num -= 1;
        swap_int(&index[0], &index[num]);
        swap_double(&distance[0], &distance[num]);
        sift_down(index, distance, 0, num);
    }
}

int kdtree_nearest(const Kdtree *self, const double *point, int *index, double *distance, int num,
                   int cap, int sorted)
{
    assert(self && point && index && distance && num > 0 && cap > 0);

    int found = 0;
    for (long i = 0; i < num; i++) {
        found = 0;
        search(self, 0, &point[i * self->dim], &index[i * cap], &distance[i * cap], &found, cap);

        for (int j = 0; j < found; j++) {
            distance[(i * cap) + j] = sqrt(distance[(i * cap) + j]);
        }

        if (sorted) {
            sort_results(&index[i * cap], &distance[i * cap], found);
        }
    }

    return found;
}

static void search_radius(const Kdtree *self, int idx, const double *point, double radius2,
                          int *index, double *distance, int *num, int cap)
{
    if (bbox_dist2(self, point, get_bbox(self, idx)) > radius2) {
        return;
    }

    const Node *node = &self->node[idx];

    if (node->axis == -1) {
        for (int i = node->child.leaf.beg; i < node->child.leaf.end; i++) {
            double dist2 = 0;
            for (int j = 0; j < self->dim; j++) {
                double diff = point[j] - get_value(self, i, j);
                dist2 += diff * diff;
            }
            if (dist2 <= radius2) {
                if (*num < cap) {
                    index[*num] = self->index[i];
                    distance[*num] = sqrt(dist2);
                }
                *num += 1;
            }
        }
        return;
    }

    search_radius(self, node->child.node.left, point, radius2, index, distance, num, cap);
    search_radius(self, node->child.node.right, point, radius2, index, distance, num, cap);
}

int kdtree_radius(const Kdtree *self, const double *point, double radius, int **offset, int **index,
                  double **distance, int num, int sorted)
{
    assert(self && point && radius >= 0 && num > 0 && offset && index && distance);

    *offset = malloc((size_t)(num + 1) * sizeof(**offset));
    assert(*offset);

    int cap = num * self->leaf_size;
    *index = malloc((size_t)cap * sizeof(**index));
    assert(*index);

    *distance = malloc((size_t)cap * sizeof(**distance));
    assert(*distance);

    int total = 0;
    for (long i = 0; i < num; i++) {
        (*offset)[i] = total;

        int found;
        int remaining = cap - total;
        while (1) {
            found = 0;
            search_radius(self, 0, &point[i * self->dim], radius * radius, &(*index)[total],
                          &(*distance)[total], &found, remaining);
            if (found <= remaining) {
                break;
            }
            cap = total + (2 * found);
            *index = realloc(*index, (size_t)cap * sizeof(**index));
            assert(*index);
            *distance = realloc(*distance, (size_t)cap * sizeof(**distance));
            assert(*distance);
            remaining = cap - total;
        }

        if (sorted) {
            for (int j = (found / 2) - 1; j >= 0; j--) {
                sift_down(*index + total, *distance + total, j, found);
            }
            sort_results(*index + total, *distance + total, found);
        }

        total += found;
    }

    (*offset)[num] = total;
    return total;
}

static double node_dist2(const Kdtree *self, int lhs, int rhs)
{
    const Rect *bbox_lhs = get_bbox(self, lhs);
    const Rect *bbox_rhs = get_bbox(self, rhs);
    double dist2 = 0;
    for (int i = 0; i < self->dim; i++) {
        if (bbox_lhs[i].max < bbox_rhs[i].min) {
            double diff = bbox_rhs[i].min - bbox_lhs[i].max;
            dist2 += diff * diff;
        }
        else if (bbox_rhs[i].max < bbox_lhs[i].min) {
            double diff = bbox_lhs[i].min - bbox_rhs[i].max;
            dist2 += diff * diff;
        }
    }
    return dist2;
}

typedef struct {
    int num;
    int cap;
    int (*pair)[2];
} Pairs;

static void pair_push(Pairs *pairs, int lhs, int rhs)
{
    if (pairs->num == pairs->cap) {
        int new_cap = pairs->cap > 0 ? pairs->cap * 2 : 1;
        int (*tmp)[2] = realloc(pairs->pair, (size_t)new_cap * sizeof(*tmp));
        assert(tmp);
        pairs->pair = tmp;
        pairs->cap = new_cap;
    }
    pairs->pair[pairs->num][0] = lhs;
    pairs->pair[pairs->num][1] = rhs;
    pairs->num += 1;
}

static void search_pairs(const Kdtree *self, int lhs, int rhs, double radius2, Pairs *pairs)
{
    if (lhs > rhs) {
        swap_int(&lhs, &rhs);
    }

    if (node_dist2(self, lhs, rhs) > radius2) {
        return;
    }

    const Node *node_lhs = &self->node[lhs];

    if (lhs == rhs) {
        if (node_lhs->axis == -1) {
            for (int i = node_lhs->child.leaf.beg; i < node_lhs->child.leaf.end; i++) {
                for (int j = i + 1; j < node_lhs->child.leaf.end; j++) {
                    double dist2 = 0;
                    for (int k = 0; k < self->dim; k++) {
                        double diff = get_value(self, i, k) - get_value(self, j, k);
                        dist2 += diff * diff;
                    }
                    if (dist2 <= radius2) {
                        int idx_i = self->index[i];
                        int idx_j = self->index[j];
                        if (idx_i > idx_j) {
                            swap_int(&idx_i, &idx_j);
                        }
                        pair_push(pairs, idx_i, idx_j);
                    }
                }
            }
            return;
        }
        int left = node_lhs->child.node.left;
        int right = node_lhs->child.node.right;
        search_pairs(self, left, left, radius2, pairs);
        search_pairs(self, left, right, radius2, pairs);
        search_pairs(self, right, right, radius2, pairs);
        return;
    }

    const Node *node_rhs = &self->node[rhs];

    if (node_lhs->axis == -1 && node_rhs->axis == -1) {
        for (int i = node_lhs->child.leaf.beg; i < node_lhs->child.leaf.end; i++) {
            for (int j = node_rhs->child.leaf.beg; j < node_rhs->child.leaf.end; j++) {
                double dist2 = 0;
                for (int k = 0; k < self->dim; k++) {
                    double diff = get_value(self, i, k) - get_value(self, j, k);
                    dist2 += diff * diff;
                }
                if (dist2 <= radius2) {
                    int idx_i = self->index[i];
                    int idx_j = self->index[j];
                    if (idx_i > idx_j) {
                        swap_int(&idx_i, &idx_j);
                    }
                    pair_push(pairs, idx_i, idx_j);
                }
            }
        }
        return;
    }

    if (node_lhs->axis == -1 || (node_rhs->axis != -1 && node_rhs->num >= node_lhs->num)) {
        search_pairs(self, lhs, node_rhs->child.node.left, radius2, pairs);
        search_pairs(self, lhs, node_rhs->child.node.right, radius2, pairs);
    }
    else {
        search_pairs(self, node_lhs->child.node.left, rhs, radius2, pairs);
        search_pairs(self, node_lhs->child.node.right, rhs, radius2, pairs);
    }
}

static double other_dist2(const Kdtree *self, const Kdtree *other, int idx_self, int idx_other)
{
    const Rect *bbox_self = get_bbox(self, idx_self);
    const Rect *bbox_other = get_bbox(other, idx_other);
    double dist2 = 0;
    for (int i = 0; i < self->dim; i++) {
        if (bbox_self[i].max < bbox_other[i].min) {
            double diff = bbox_other[i].min - bbox_self[i].max;
            dist2 += diff * diff;
        }
        else if (bbox_other[i].max < bbox_self[i].min) {
            double diff = bbox_self[i].min - bbox_other[i].max;
            dist2 += diff * diff;
        }
    }
    return dist2;
}

static void search_pairs_other(const Kdtree *self, const Kdtree *other, int idx_self, int idx_other,
                               double radius2, Pairs *pairs)
{
    if (other_dist2(self, other, idx_self, idx_other) > radius2) {
        return;
    }

    const Node *node_self = &self->node[idx_self];
    const Node *node_other = &other->node[idx_other];

    if (node_self->axis == -1 && node_other->axis == -1) {
        for (int i = node_self->child.leaf.beg; i < node_self->child.leaf.end; i++) {
            for (int j = node_other->child.leaf.beg; j < node_other->child.leaf.end; j++) {
                double dist2 = 0;
                for (int k = 0; k < self->dim; k++) {
                    double diff = get_value(self, i, k) - get_value(other, j, k);
                    dist2 += diff * diff;
                }
                if (dist2 <= radius2) {
                    pair_push(pairs, self->index[i], other->index[j]);
                }
            }
        }
        return;
    }

    if (node_self->axis == -1 || (node_other->axis != -1 && node_other->num >= node_self->num)) {
        search_pairs_other(self, other, idx_self, node_other->child.node.left, radius2, pairs);
        search_pairs_other(self, other, idx_self, node_other->child.node.right, radius2, pairs);
    }
    else {
        search_pairs_other(self, other, node_self->child.node.left, idx_other, radius2, pairs);
        search_pairs_other(self, other, node_self->child.node.right, idx_other, radius2, pairs);
    }
}

int kdtree_pairs(const Kdtree *self, const Kdtree *other, double radius, int (**pair)[2])
{
    assert(self && radius >= 0 && pair);

    Pairs pairs = {0};
    if (!other) {
        search_pairs(self, 0, 0, radius * radius, &pairs);
    }
    else {
        assert(self->dim == other->dim);
        search_pairs_other(self, other, 0, 0, radius * radius, &pairs);
    }

    *pair = pairs.pair;
    return pairs.num;
}

static int lower_bound(const double *radius, int num, double dist2)
{
    int beg = 0;
    int end = num;
    while (beg < end) {
        int mid = (beg + end) / 2;
        if (radius[mid] * radius[mid] < dist2) {
            beg = mid + 1;
        }
        else {
            end = mid;
        }
    }
    return beg;
}

static void search_counts(const Kdtree *self, int lhs, int rhs, const double *radius, long *count,
                          int num)
{
    if (lhs > rhs) {
        swap_int(&lhs, &rhs);
    }

    if (node_dist2(self, lhs, rhs) > radius[num - 1] * radius[num - 1]) {
        return;
    }

    const Node *node_lhs = &self->node[lhs];

    if (lhs == rhs) {
        if (node_lhs->axis == -1) {
            for (int i = node_lhs->child.leaf.beg; i < node_lhs->child.leaf.end; i++) {
                for (int j = i + 1; j < node_lhs->child.leaf.end; j++) {
                    double dist2 = 0;
                    for (int k = 0; k < self->dim; k++) {
                        double diff = get_value(self, i, k) - get_value(self, j, k);
                        dist2 += diff * diff;
                    }
                    int bin = lower_bound(radius, num, dist2);
                    if (bin < num) {
                        count[bin] += 1;
                    }
                }
            }
            return;
        }
        int left = node_lhs->child.node.left;
        int right = node_lhs->child.node.right;
        search_counts(self, left, left, radius, count, num);
        search_counts(self, left, right, radius, count, num);
        search_counts(self, right, right, radius, count, num);
        return;
    }

    const Node *node_rhs = &self->node[rhs];

    if (node_lhs->axis == -1 && node_rhs->axis == -1) {
        for (int i = node_lhs->child.leaf.beg; i < node_lhs->child.leaf.end; i++) {
            for (int j = node_rhs->child.leaf.beg; j < node_rhs->child.leaf.end; j++) {
                double dist2 = 0;
                for (int k = 0; k < self->dim; k++) {
                    double diff = get_value(self, i, k) - get_value(self, j, k);
                    dist2 += diff * diff;
                }
                int bin = lower_bound(radius, num, dist2);
                if (bin < num) {
                    count[bin] += 1;
                }
            }
        }
        return;
    }

    if (node_lhs->axis == -1 || (node_rhs->axis != -1 && node_rhs->num >= node_lhs->num)) {
        search_counts(self, lhs, node_rhs->child.node.left, radius, count, num);
        search_counts(self, lhs, node_rhs->child.node.right, radius, count, num);
    }
    else {
        search_counts(self, node_lhs->child.node.left, rhs, radius, count, num);
        search_counts(self, node_lhs->child.node.right, rhs, radius, count, num);
    }
}

static void search_counts_other(const Kdtree *self, const Kdtree *other, int idx_self,
                                int idx_other, const double *radius, long *count, int num)
{
    if (other_dist2(self, other, idx_self, idx_other) > radius[num - 1] * radius[num - 1]) {
        return;
    }

    const Node *node_self = &self->node[idx_self];
    const Node *node_other = &other->node[idx_other];

    if (node_self->axis == -1 && node_other->axis == -1) {
        for (int i = node_self->child.leaf.beg; i < node_self->child.leaf.end; i++) {
            for (int j = node_other->child.leaf.beg; j < node_other->child.leaf.end; j++) {
                double dist2 = 0;
                for (int k = 0; k < self->dim; k++) {
                    double diff = get_value(self, i, k) - get_value(other, j, k);
                    dist2 += diff * diff;
                }
                int bin = lower_bound(radius, num, dist2);
                if (bin < num) {
                    count[bin] += 1;
                }
            }
        }
        return;
    }

    if (node_self->axis == -1 || (node_other->axis != -1 && node_other->num >= node_self->num)) {
        search_counts_other(self, other, idx_self, node_other->child.node.left, radius, count, num);
        search_counts_other(self, other, idx_self, node_other->child.node.right, radius, count,
                            num);
    }
    else {
        search_counts_other(self, other, node_self->child.node.left, idx_other, radius, count, num);
        search_counts_other(self, other, node_self->child.node.right, idx_other, radius, count,
                            num);
    }
}

void kdtree_counts(const Kdtree *self, const Kdtree *other, const double *radius, long *count,
                   int num, int cumulative)
{
    assert(self && radius && num >= 1 && count);

    memset(count, 0, (size_t)num * sizeof(*count));
    if (!other) {
        search_counts(self, 0, 0, radius, count, num);
    }
    else {
        assert(self->dim == other->dim);
        search_counts_other(self, other, 0, 0, radius, count, num);
    }

    if (cumulative) {
        for (int i = 1; i < num; i++) {
            count[i] += count[i - 1];
        }
    }
}

void kdtree_dump(const Kdtree *self, const char *fname)
{
    assert(self && fname);

    FILE *file = fopen(fname, "w");
    assert(file);

    int num_nodes = self->num_nodes;
    fprintf(file, "# kdtree dim=%d nodes=%d\n", self->dim, num_nodes);
    fprintf(file, "# idx num axis value left right beg end depth bbox_min bbox_max\n");

    int *depth = calloc((size_t)num_nodes, sizeof(*depth));
    assert(depth);
    for (int i = 0; i < num_nodes; i++) {
        const Node *node = &self->node[i];
        if (node->axis != -1) {
            depth[node->child.node.left] = depth[i] + 1;
            depth[node->child.node.right] = depth[i] + 1;
        }
    }

    for (int i = 0; i < num_nodes; i++) {
        const Node *node = &self->node[i];
        if (node->axis == -1) {
            fprintf(file, "%d %d %d %g %d %d %d %d %d", i, node->num, node->axis, 0.0, -1, -1,
                    node->child.leaf.beg, node->child.leaf.end, depth[i]);
        }
        else {
            fprintf(file, "%d %d %d %g %d %d %d %d %d", i, node->num, node->axis, node->value,
                    node->child.node.left, node->child.node.right, 0, 0, depth[i]);
        }
        const Rect *bbox = get_bbox(self, i);
        for (int j = 0; j < self->dim; j++) {
            fprintf(file, " %g", bbox[j].min);
        }
        for (int j = 0; j < self->dim; j++) {
            fprintf(file, " %g", bbox[j].max);
        }
        fprintf(file, "\n");
    }

    free(depth);
    fclose(file);
}
