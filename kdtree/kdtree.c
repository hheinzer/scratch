#include "kdtree.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "alloc.h"

enum { LEAF_SIZE = 10 };

typedef struct {
    int dim;       // split dimension; -1 => leaf
    double value;  // split value
    union {
        struct {
            int beg, end;  // slice of index array
        } leaf;
        struct {
            int left, right;  // child node indices
        } inner;
    } child;
} Node;

typedef struct {
    double min, max;
} Rect;

struct kdtree {
    int dim;
    int num;
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

static double get_value(const Kdtree *self, int pos, int dim)
{
    return self->point[((long)self->index[pos] * self->dim) + dim];
}

static int split_dim(const Kdtree *self, int beg, int end)
{
    int best_dim = 0;
    double best_spread = -1;
    for (int dim = 0; dim < self->dim; dim++) {
        double min = get_value(self, beg, dim);
        double max = min;
        for (int i = beg + 1; i < end; i++) {
            double value = get_value(self, i, dim);
            if (value < min) {
                min = value;
            }
            else if (value > max) {
                max = value;
            }
        }
        if (max - min > best_spread) {
            best_spread = max - min;
            best_dim = dim;
        }
    }
    return best_dim;
}

static void swap_int(int *lhs, int *rhs)
{
    int swap = *lhs;
    *lhs = *rhs;
    *rhs = swap;
}

static void nth_element(Kdtree *self, int beg, int end, int mid, int dim)
{
    int min = beg;
    int max = end - 1;
    while (min < max) {
        double pivot = get_value(self, (min + max) / 2, dim);
        int left = min;
        int right = max;
        while (left <= right) {
            while (get_value(self, left, dim) < pivot) {
                left += 1;
            }
            while (get_value(self, right, dim) > pivot) {
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
        self->node[idx].dim = -1;
        self->node[idx].child.leaf.beg = beg;
        self->node[idx].child.leaf.end = end;
        return;
    }

    int mid = (beg + end) / 2;
    int dim = split_dim(self, beg, end);
    nth_element(self, beg, end, mid, dim);

    self->node[idx].dim = dim;
    self->node[idx].value = get_value(self, mid, dim);

    self->node[idx].child.inner.left = *next;
    build(self, next, beg, mid);

    self->node[idx].child.inner.right = *next;
    build(self, next, mid, end);
}

Kdtree *kdtree_init(const double *point, int dim, int num, int leaf_size)
{
    assert(point && dim > 0 && num > 0 && leaf_size >= 0);

    Kdtree *self = alloc(1, sizeof(*self));

    self->dim = dim;
    self->num = num;
    self->leaf_size = leaf_size ? leaf_size : LEAF_SIZE;
    self->point = point;

    self->index = alloc(num, sizeof(*self->index));
    for (int i = 0; i < num; i++) {
        self->index[i] = i;
    }

    int size = compute_size(self, num);
    self->node = alloc(size, sizeof(*self->node));

    int next = 0;
    build(self, &next, 0, num);
    assert(next == size);

    self->rect = alloc(dim, sizeof(*self->rect));

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

static void swap_double(double *lhs, double *rhs)
{
    double swap = *lhs;
    *lhs = *rhs;
    *rhs = swap;
}

static void heap_sift_down(int *index, double *distance2, int num, int pos)
{
    while (1) {
        int largest = pos;
        int left = (2 * pos) + 1;
        int right = (2 * pos) + 2;
        if (left < num && distance2[left] > distance2[largest]) {
            largest = left;
        }
        if (right < num && distance2[right] > distance2[largest]) {
            largest = right;
        }
        if (largest == pos) {
            break;
        }
        swap_int(&index[pos], &index[largest]);
        swap_double(&distance2[pos], &distance2[largest]);
        pos = largest;
    }
}

static void heap_sift_up(int *index, double *distance2, int pos)
{
    while (pos > 0) {
        int parent = (pos - 1) / 2;
        if (distance2[pos] <= distance2[parent]) {
            break;
        }
        swap_int(&index[pos], &index[parent]);
        swap_double(&distance2[pos], &distance2[parent]);
        pos = parent;
    }
}

static void heap_push(int *index, double *distance2, int *num, int cap, int idx, double dist2)
{
    if (*num < cap) {
        index[*num] = idx;
        distance2[*num] = dist2;
        *num += 1;
        heap_sift_up(index, distance2, *num - 1);
    }
    else if (dist2 < distance2[0]) {
        index[0] = idx;
        distance2[0] = dist2;
        heap_sift_down(index, distance2, *num, 0);
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

    if (node->dim == -1) {
        for (int i = node->child.leaf.beg; i < node->child.leaf.end; i++) {
            double dist2 = 0;
            for (int j = 0; j < self->dim; j++) {
                double diff = point[j] - get_value(self, i, j);
                dist2 += diff * diff;
            }
            heap_push(index, distance2, num, cap, self->index[i], dist2);
        }
        return;
    }

    int split = node->dim;
    double split_val = node->value;
    double query_val = point[split];

    int closer_child;
    int farther_child;
    double *closer_limit;
    double *farther_limit;

    if (query_val <= split_val) {
        closer_child = node->child.inner.left;
        farther_child = node->child.inner.right;
        closer_limit = &self->rect[split].max;
        farther_limit = &self->rect[split].min;
    }
    else {
        closer_child = node->child.inner.right;
        farther_child = node->child.inner.left;
        closer_limit = &self->rect[split].min;
        farther_limit = &self->rect[split].max;
    }

    double closer_prev = *closer_limit;
    *closer_limit = split_val;
    search(self, closer_child, point, index, distance2, num, cap);
    *closer_limit = closer_prev;

    double farther_prev = *farther_limit;
    *farther_limit = split_val;
    if (*num < cap || rect_dist2(self, point) <= distance2[0]) {
        search(self, farther_child, point, index, distance2, num, cap);
    }
    *farther_limit = farther_prev;
}

void kdtree_query(const Kdtree *self, const double *point, int *index, double *distance, int num_)
{
    assert(self && point && index && distance && num_ > 0);

    for (int i = 0; i < self->dim; i++) {
        self->rect[i].min = -DBL_MAX;
        self->rect[i].max = DBL_MAX;
    }

    int cap = (num_ < self->num) ? num_ : self->num;

    int num = 0;
    search(self, 0, point, index, distance, &num, cap);

    // sort results ascending by distance
    for (int i = 1; i < num; i++) {
        int idx = index[i];
        double dist2 = distance[i];
        int cur = i - 1;
        while (cur >= 0 && distance[cur] > dist2) {
            index[cur + 1] = index[cur];
            distance[cur + 1] = distance[cur];
            cur -= 1;
        }
        index[cur + 1] = idx;
        distance[cur + 1] = dist2;
    }

    for (int i = 0; i < num; i++) {
        distance[i] = sqrt(distance[i]);
    }

    for (int i = num; i < num_; i++) {
        index[i] = -1;
        distance[i] = INFINITY;
    }
}
