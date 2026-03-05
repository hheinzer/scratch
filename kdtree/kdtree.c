#include "kdtree.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>

enum { LEAF_SIZE = 10, HEAP_THRESHOLD = 32 };

typedef struct {
    int num;       // number of points in subtree
    int axis;      // split axis; -1 => leaf
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

    self->dim = dim;
    self->leaf_size = leaf_size ? leaf_size : LEAF_SIZE;
    self->point = point;

    self->index = malloc((size_t)num * sizeof(*self->index));
    assert(self->index);
    for (int i = 0; i < num; i++) {
        self->index[i] = i;
    }

    int size = compute_size(self, num);
    self->node = malloc((size_t)size * sizeof(*self->node));
    assert(self->node);

    self->bbox = malloc((size_t)size * (size_t)dim * sizeof(*self->bbox));
    assert(self->bbox);

    int next = 0;
    build(self, &next, 0, num);
    assert(next == size);

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
        swap_int(&index[0], &index[num - 1]);
        swap_double(&distance[0], &distance[num - 1]);
        num -= 1;
        sift_down(index, distance, 0, num);
    }
}

int kdtree_nearest(const Kdtree *self, const double *point, int *index, double *distance, int cap,
                   int sorted)
{
    assert(self && point && index && distance && cap > 0);

    int num = 0;
    search(self, 0, point, index, distance, &num, cap);

    for (int i = 0; i < num; i++) {
        distance[i] = sqrt(distance[i]);
    }

    if (sorted) {
        sort_results(index, distance, num);
    }

    return num;
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

int kdtree_radius(const Kdtree *self, const double *point, double radius, int *index,
                  double *distance, int cap, int sorted)
{
    assert(self && point && radius >= 0 && index && distance && cap > 0);

    int num = 0;
    search_radius(self, 0, point, radius * radius, index, distance, &num, cap);

    if (sorted) {
        int min = (num < cap) ? num : cap;
        for (int i = (min / 2) - 1; i >= 0; i--) {
            sift_down(index, distance, i, min);
        }
        sort_results(index, distance, min);
    }

    return num;
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

int kdtree_pairs(const Kdtree *self, double radius, int (**pair)[2])
{
    assert(self && radius >= 0 && pair);

    Pairs pairs = {0};
    search_pairs(self, 0, 0, radius * radius, &pairs);

    *pair = pairs.pair;
    return pairs.num;
}

static double cross_dist2(const Kdtree *self, const Kdtree *other, int idx_self, int idx_other)
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

static void search_cross(const Kdtree *self, const Kdtree *other, int idx_self, int idx_other,
                         double radius2, Pairs *pairs)
{
    if (cross_dist2(self, other, idx_self, idx_other) > radius2) {
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
        search_cross(self, other, idx_self, node_other->child.node.left, radius2, pairs);
        search_cross(self, other, idx_self, node_other->child.node.right, radius2, pairs);
    }
    else {
        search_cross(self, other, node_self->child.node.left, idx_other, radius2, pairs);
        search_cross(self, other, node_self->child.node.right, idx_other, radius2, pairs);
    }
}

int kdtree_cross(const Kdtree *self, const Kdtree *other, double radius, int (**pair)[2])
{
    assert(self && other && self->dim == other->dim && radius >= 0 && pair);

    Pairs pairs = {0};
    search_cross(self, other, 0, 0, radius * radius, &pairs);

    *pair = pairs.pair;
    return pairs.num;
}
