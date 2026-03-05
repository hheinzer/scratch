#include "kdtree.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

enum { LEAF_SIZE = 10 };

typedef struct {
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
    Rect *rect;
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

static double get_value(const Kdtree *self, int pos, int axis)
{
    return self->point[((long)self->index[pos] * self->dim) + axis];
}

static int split_axis(const Kdtree *self, int beg, int end)
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

    if (num <= self->leaf_size) {
        self->node[idx].axis = -1;
        self->node[idx].child.leaf.beg = beg;
        self->node[idx].child.leaf.end = end;
        return;
    }

    int mid = (beg + end) / 2;
    int axis = split_axis(self, beg, end);
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

    int next = 0;
    build(self, &next, 0, num);
    assert(next == size);

    self->rect = malloc((size_t)dim * sizeof(*self->rect));
    assert(self->rect);

    return self;
}

void kdtree_deinit(Kdtree *self)
{
    assert(self);
    free(self->index);
    free(self->node);
    free(self->rect);
    free(self);
}

static void sorted_push(int *index, double *distance2, int *num, int cap, int idx, double dist2)
{
    if (*num == cap && dist2 >= distance2[cap - 1]) {
        return;
    }
    int end = (*num < cap) ? *num : cap - 1;
    int pos = end;
    while (pos > 0 && distance2[pos - 1] > dist2) {
        index[pos] = index[pos - 1];
        distance2[pos] = distance2[pos - 1];
        pos -= 1;
    }
    index[pos] = idx;
    distance2[pos] = dist2;
    if (*num < cap) {
        *num += 1;
    }
}

static double rect_dist2(const Kdtree *self, const double *point)
{
    double dist2 = 0;
    for (int i = 0; i < self->dim; i++) {
        if (point[i] < self->rect[i].min) {
            double diff = self->rect[i].min - point[i];
            dist2 += diff * diff;
        }
        else if (point[i] > self->rect[i].max) {
            double diff = point[i] - self->rect[i].max;
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
            sorted_push(index, distance2, num, cap, self->index[i], dist2);
        }
        return;
    }

    int axis = node->axis;
    double split_val = node->value;
    double query_val = point[axis];

    int closer_child;
    int farther_child;
    double *closer_limit;
    double *farther_limit;

    if (query_val <= split_val) {
        closer_child = node->child.node.left;
        farther_child = node->child.node.right;
        closer_limit = &self->rect[axis].max;
        farther_limit = &self->rect[axis].min;
    }
    else {
        closer_child = node->child.node.right;
        farther_child = node->child.node.left;
        closer_limit = &self->rect[axis].min;
        farther_limit = &self->rect[axis].max;
    }

    double closer_prev = *closer_limit;
    *closer_limit = split_val;
    search(self, closer_child, point, index, distance2, num, cap);
    *closer_limit = closer_prev;

    double farther_prev = *farther_limit;
    *farther_limit = split_val;
    if (*num < cap || rect_dist2(self, point) <= distance2[cap - 1]) {
        search(self, farther_child, point, index, distance2, num, cap);
    }
    *farther_limit = farther_prev;
}

int kdtree_query(const Kdtree *self, const double *point, int *index, double *distance, int cap)
{
    assert(self && point && index && distance && cap > 0);

    for (int i = 0; i < self->dim; i++) {
        self->rect[i].min = -DBL_MAX;
        self->rect[i].max = DBL_MAX;
    }

    int num = 0;
    search(self, 0, point, index, distance, &num, cap);

    for (int i = 0; i < num; i++) {
        distance[i] = sqrt(distance[i]);
    }

    return num;
}

static void search_radius(const Kdtree *self, int idx, const double *point, double radius2,
                          int *index, double *distance, int *num, int cap)
{
    if (rect_dist2(self, point) > radius2) {
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

    int axis = node->axis;
    double split_val = node->value;

    double prev_max = self->rect[axis].max;
    self->rect[axis].max = split_val;
    search_radius(self, node->child.node.left, point, radius2, index, distance, num, cap);
    self->rect[axis].max = prev_max;

    double prev_min = self->rect[axis].min;
    self->rect[axis].min = split_val;
    search_radius(self, node->child.node.right, point, radius2, index, distance, num, cap);
    self->rect[axis].min = prev_min;
}

int kdtree_query_radius(const Kdtree *self, const double *point, double radius, int *index,
                        double *distance, int cap, int sorted)
{
    assert(self && point && radius >= 0 && index && distance && cap > 0);

    for (int i = 0; i < self->dim; i++) {
        self->rect[i].min = -DBL_MAX;
        self->rect[i].max = DBL_MAX;
    }

    int num = 0;
    search_radius(self, 0, point, radius * radius, index, distance, &num, cap);

    if (sorted) {
        int min = (num < cap) ? num : cap;
        for (int i = 1; i < min; i++) {
            int idx = index[i];
            double dist = distance[i];
            int pos = i;
            while (pos > 0 && distance[pos - 1] > dist) {
                index[pos] = index[pos - 1];
                distance[pos] = distance[pos - 1];
                pos -= 1;
            }
            index[pos] = idx;
            distance[pos] = dist;
        }
    }

    return num;
}
