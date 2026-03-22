//
// --- HEADER ---
//

typedef struct tensor Tensor;

// creation

Tensor *tensor_empty(const int *shape, int ndim);
Tensor *tensor_zeros(const int *shape, int ndim);
Tensor *tensor_ones(const int *shape, int ndim);
Tensor *tensor_fill(const int *shape, int ndim, float value);
Tensor *tensor_arange(float start, float stop, float step);
Tensor *tensor_range(float start, float stop, float step);
Tensor *tensor_linspace(float start, float stop, int steps);
Tensor *tensor_logspace(float base, float start, float stop, int steps);
Tensor *tensor_eye(int rows, int cols);
Tensor *tensor_from(const int *shape, int ndim, const float *data);
Tensor *tensor_rand(const int *shape, int ndim);
Tensor *tensor_randn(const int *shape, int ndim);

// movement

Tensor *tensor_reshape(const Tensor *src, const int *shape, int ndim);
Tensor *tensor_flatten(const Tensor *src, int beg_dim, int end_dim);
Tensor *tensor_flatten_all(const Tensor *src);
Tensor *tensor_unflatten(const Tensor *src, int dim, const int *size, int num);
Tensor *tensor_squeeze(const Tensor *src, int dim);
Tensor *tensor_squeeze_all(const Tensor *src);
Tensor *tensor_unsqueeze(const Tensor *src, int dim);
Tensor *tensor_permute(const Tensor *src, const int *order);
Tensor *tensor_transpose(const Tensor *src, int dim0, int dim1);
Tensor *tensor_slice(const Tensor *src, int dim, int beg, int end, int step);
Tensor *tensor_select(const Tensor *src, int dim, int index);
Tensor *tensor_expand(const Tensor *src, const int *shape, int ndim);
Tensor *tensor_cat(const Tensor **src, int num, int dim);
Tensor *tensor_stack(const Tensor **src, int num, int dim);

// unary

Tensor *tensor_neg(const Tensor *src);
Tensor *tensor_abs(const Tensor *src);
Tensor *tensor_sign(const Tensor *src);
Tensor *tensor_square(const Tensor *src);
Tensor *tensor_sqrt(const Tensor *src);
Tensor *tensor_rsqrt(const Tensor *src);
Tensor *tensor_exp(const Tensor *src);
Tensor *tensor_log(const Tensor *src);
Tensor *tensor_relu(const Tensor *src);
Tensor *tensor_sigmoid(const Tensor *src);
Tensor *tensor_tanh(const Tensor *src);
Tensor *tensor_logical_not(const Tensor *src);

// binary

Tensor *tensor_add(const Tensor *lhs, const Tensor *rhs);
Tensor *tensor_sub(const Tensor *lhs, const Tensor *rhs);
Tensor *tensor_mul(const Tensor *lhs, const Tensor *rhs);
Tensor *tensor_div(const Tensor *lhs, const Tensor *rhs);
Tensor *tensor_mod(const Tensor *lhs, const Tensor *rhs);
Tensor *tensor_pow(const Tensor *lhs, const Tensor *rhs);
Tensor *tensor_eq(const Tensor *lhs, const Tensor *rhs);
Tensor *tensor_ne(const Tensor *lhs, const Tensor *rhs);
Tensor *tensor_lt(const Tensor *lhs, const Tensor *rhs);
Tensor *tensor_le(const Tensor *lhs, const Tensor *rhs);
Tensor *tensor_gt(const Tensor *lhs, const Tensor *rhs);
Tensor *tensor_ge(const Tensor *lhs, const Tensor *rhs);
Tensor *tensor_logical_and(const Tensor *lhs, const Tensor *rhs);
Tensor *tensor_logical_or(const Tensor *lhs, const Tensor *rhs);
Tensor *tensor_logical_xor(const Tensor *lhs, const Tensor *rhs);
Tensor *tensor_minimum(const Tensor *lhs, const Tensor *rhs);
Tensor *tensor_maximum(const Tensor *lhs, const Tensor *rhs);

// i/o

void tensor_print(const Tensor *self);

//
// --- SOURCE ---
//

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { MAX_NDIM = 8 };

struct tensor {
    int ndim;
    long numel;
    int shape[MAX_NDIM];
    long stride[MAX_NDIM];
    float *data;
};

// memory

enum { MAX_SAVE = 1024 };

typedef struct stack {
    void *ptr;
    struct stack *prev;
} Stack;

static Stack *g_head = 0;

static void *stack_malloc(size_t num, size_t size)
{
    assert(size > 0 && num <= SIZE_MAX / size);
    if (num == 0) {
        return 0;
    }
    Stack *next = malloc(sizeof(*next));
    assert(next);
    next->ptr = malloc(num * size);
    assert(next->ptr);
    next->prev = g_head;
    g_head = next;
    return next->ptr;
}

static void *stack_calloc(size_t num, size_t size)
{
    void *ptr = stack_malloc(num, size);
    return ptr ? memset(ptr, 0, num * size) : 0;
}

static void *stack_memdup(const void *ptr, size_t num, size_t size)
{
    void *dup = stack_malloc(num, size);
    return (dup && ptr) ? memcpy(dup, ptr, num * size) : 0;
}

static int g_index = 0;
static Stack *g_save[MAX_SAVE];

static void *stack_pop(void)
{
    assert(g_head);
    void *ptr = g_head->ptr;
    Stack *prev = g_head->prev;
    free(g_head);
    g_head = prev;
    return ptr;
}

static void stack_clear(void)
{
    while (g_head) {
        free(stack_pop());
    }
    g_index = 0;
}

static void stack_save(void)
{
    assert(g_index < MAX_SAVE);
    g_save[g_index++] = g_head;
}

static void stack_restore(void)
{
    assert(g_index - 1 >= 0);
    Stack *save = g_save[--g_index];
    while (g_head != save) {
        free(stack_pop());
    }
}

// creation

static int valid_shape(const int *shape, int ndim)
{
    if (ndim == 0) {
        return 1;
    }
    if (ndim < 0 || ndim > MAX_NDIM || !shape) {
        return 0;
    }
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            return 0;
        }
    }
    return 1;
}

static long compute_numel(const int *shape, int ndim)
{
    long numel = 1;
    for (int i = 0; i < ndim; i++) {
        numel *= shape[i];
    }
    return numel;
}

static void compute_stride(long *stride, const int *shape, int ndim)
{
    stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
}

Tensor *tensor_empty(const int *shape, int ndim)
{
    assert(valid_shape(shape, ndim));
    Tensor *out = stack_calloc(1, sizeof(*out));
    out->ndim = ndim;
    out->numel = compute_numel(shape, ndim);
    if (ndim > 0) {
        memcpy(out->shape, shape, ndim * sizeof(*shape));
        compute_stride(out->stride, shape, ndim);
    }
    out->data = stack_malloc(out->numel, sizeof(*out->data));
    return out;
}

Tensor *tensor_zeros(const int *shape, int ndim)
{
    Tensor *out = tensor_empty(shape, ndim);
    memset(out->data, 0, out->numel * sizeof(*out->data));
    return out;
}

Tensor *tensor_ones(const int *shape, int ndim)
{
    return tensor_fill(shape, ndim, 1);
}

Tensor *tensor_fill(const int *shape, int ndim, float value)
{
    if (value == 0) {
        return tensor_zeros(shape, ndim);
    }
    Tensor *out = tensor_empty(shape, ndim);
    for (long i = 0; i < out->numel; i++) {
        out->data[i] = value;
    }
    return out;
}

Tensor *tensor_arange(float start, float stop, float step)
{
    assert(step != 0);
    int numel = (int)ceilf((stop - start) / step);
    float last = start + ((float)(numel - 1) * step);
    if (numel > 0 && ((step > 0 && last >= stop) || (step < 0 && last <= stop))) {
        numel -= 1;
    }
    assert(numel > 0);
    Tensor *out = tensor_empty((int[]){numel}, 1);
    for (int i = 0; i < numel; i++) {
        out->data[i] = start + ((float)i * step);
    }
    return out;
}

Tensor *tensor_range(float start, float stop, float step)
{
    assert(step != 0);
    int numel = (int)ceilf((stop - start) / step) + 1;
    float last = start + ((float)(numel - 1) * step);
    if (numel > 1 && ((step > 0 && last > stop) || (step < 0 && last < stop))) {
        numel -= 1;
    }
    assert(numel > 0);
    Tensor *out = tensor_empty((int[]){numel}, 1);
    for (int i = 0; i < numel; i++) {
        out->data[i] = start + ((float)i * step);
    }
    return out;
}

Tensor *tensor_linspace(float start, float stop, int steps)
{
    assert(steps > 0);
    Tensor *out = tensor_empty((int[]){steps}, 1);
    if (steps == 1) {
        out->data[0] = start;
    }
    else {
        float step = (stop - start) / (float)(steps - 1);
        for (int i = 0; i < steps; i++) {
            out->data[i] = start + ((float)i * step);
        }
    }
    return out;
}

Tensor *tensor_logspace(float base, float start, float stop, int steps)
{
    assert(steps > 0 && base > 0);
    Tensor *out = tensor_empty((int[]){steps}, 1);
    if (steps == 1) {
        out->data[0] = powf(base, start);
    }
    else {
        float step = (stop - start) / (float)(steps - 1);
        for (int i = 0; i < steps; i++) {
            out->data[i] = powf(base, start + ((float)i * step));
        }
    }
    return out;
}

Tensor *tensor_eye(int rows, int cols)
{
    assert(rows > 0 && cols > 0);
    Tensor *out = tensor_zeros((int[]){rows, cols}, 2);
    int diag = (rows < cols) ? rows : cols;
    for (long i = 0; i < diag; i++) {
        out->data[i * (cols + 1)] = 1;
    }
    return out;
}

Tensor *tensor_from(const int *shape, int ndim, const float *data)
{
    assert(data);
    Tensor *out = tensor_empty(shape, ndim);
    memcpy(out->data, data, out->numel * sizeof(*data));
    return out;
}

static float random_uniform(void)
{
    return (float)rand() / (float)RAND_MAX;
}

Tensor *tensor_rand(const int *shape, int ndim)
{
    Tensor *out = tensor_empty(shape, ndim);
    for (long i = 0; i < out->numel; i++) {
        out->data[i] = random_uniform();
    }
    return out;
}

static void random_normal(float *rn0, float *rn1)
{
    static const float two_pi = 2 * 3.14159265358979323846F;
    float ru1;
    do {
        ru1 = random_uniform();
    } while (ru1 == 0);
    float ru2 = random_uniform();
    float mag = sqrtf(-2 * logf(ru1));
    *rn0 = mag * cosf(two_pi * ru2);
    *rn1 = mag * sinf(two_pi * ru2);
}

Tensor *tensor_randn(const int *shape, int ndim)
{
    Tensor *out = tensor_empty(shape, ndim);
    for (long i = 0; i + 1 < out->numel; i += 2) {
        random_normal(&out->data[i], &out->data[i + 1]);
    }
    if (out->numel % 2 != 0) {
        float dummy;
        random_normal(&out->data[out->numel - 1], &dummy);
    }
    return out;
}

// movement

static void normalize_shape(int *out_shape, const int *shape, int ndim, long numel)
{
    int index = -1;
    long product = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] == -1) {
            assert(index == -1);
            index = i;
        }
        else {
            assert(shape[i] > 0);
            product *= shape[i];
        }
        out_shape[i] = shape[i];
    }
    if (index != -1) {
        assert(product > 0 && numel % product == 0);
        out_shape[index] = (int)(numel / product);
    }
    else {
        assert(product == numel);
    }
}

static int is_contiguous(const Tensor *src)
{
    long stride = 1;
    for (int i = src->ndim - 1; i >= 0; i--) {
        if (src->shape[i] != 1 && src->stride[i] != stride) {
            return 0;
        }
        stride *= src->shape[i];
    }
    return 1;
}

static void pack_data(Tensor *out, long *off_out, const Tensor *src, long off_src, int dim)
{
    if (dim == src->ndim - 1) {
        if (src->stride[dim] == 1) {
            memcpy(out->data + *off_out, src->data + off_src, src->shape[dim] * sizeof(*src->data));
        }
        else {
            for (int i = 0; i < src->shape[dim]; i++) {
                out->data[*off_out + i] = src->data[off_src + (i * src->stride[dim])];
            }
        }
        *off_out += src->shape[dim];
    }
    else {
        for (int i = 0; i < src->shape[dim]; i++) {
            pack_data(out, off_out, src, off_src + (i * src->stride[dim]), dim + 1);
        }
    }
}

Tensor *tensor_reshape(const Tensor *src, const int *shape, int ndim)
{
    assert(src && ndim >= 0 && ndim <= MAX_NDIM);
    Tensor *out = stack_calloc(1, sizeof(*out));
    out->ndim = ndim;
    out->numel = src->numel;
    if (ndim > 0) {
        normalize_shape(out->shape, shape, ndim, src->numel);
        compute_stride(out->stride, out->shape, ndim);
    }
    else {
        assert(src->numel == 1);
    }
    if (is_contiguous(src)) {
        out->data = src->data;
    }
    else {
        out->data = stack_malloc(out->numel, sizeof(*out->data));
        long off_out = 0;
        pack_data(out, &off_out, src, 0, 0);
    }
    return out;
}

static int normalize_dim(int dim, int ndim)
{
    assert(-ndim <= dim && dim < ndim);
    return (dim < 0) ? (dim + ndim) : dim;
}

Tensor *tensor_flatten(const Tensor *src, int beg_dim, int end_dim)
{
    assert(src);
    beg_dim = normalize_dim(beg_dim, src->ndim);
    end_dim = normalize_dim(end_dim, src->ndim);
    assert(beg_dim <= end_dim);
    int shape[MAX_NDIM];
    int ndim = 0;
    for (int i = 0; i < beg_dim; i++) {
        shape[ndim++] = src->shape[i];
    }
    int flat = 1;
    for (int i = beg_dim; i <= end_dim; i++) {
        flat *= src->shape[i];
    }
    shape[ndim++] = flat;
    for (int i = end_dim + 1; i < src->ndim; i++) {
        shape[ndim++] = src->shape[i];
    }
    return tensor_reshape(src, shape, ndim);
}

Tensor *tensor_flatten_all(const Tensor *src)
{
    return tensor_reshape(src, (int[]){-1}, 1);
}

Tensor *tensor_unflatten(const Tensor *src, int dim, const int *size, int num)
{
    assert(src && size && num > 0 && src->ndim + num - 1 <= MAX_NDIM);
    dim = normalize_dim(dim, src->ndim);
    int shape[MAX_NDIM];
    int ndim = 0;
    for (int i = 0; i < dim; i++) {
        shape[ndim++] = src->shape[i];
    }
    int unflat = 1;
    for (int i = 0; i < num; i++) {
        assert(size[i] > 0);
        shape[ndim++] = size[i];
        unflat *= size[i];
    }
    assert(unflat == src->shape[dim]);
    for (int i = dim + 1; i < src->ndim; i++) {
        shape[ndim++] = src->shape[i];
    }
    return tensor_reshape(src, shape, ndim);
}

Tensor *tensor_squeeze(const Tensor *src, int dim)
{
    assert(src);
    dim = normalize_dim(dim, src->ndim);
    assert(src->shape[dim] == 1);
    Tensor *out = stack_memdup(src, 1, sizeof(*out));
    out->ndim -= 1;
    for (int i = dim; i < src->ndim - 1; i++) {
        out->shape[i] = src->shape[i + 1];
        out->stride[i] = src->stride[i + 1];
    }
    return out;
}

Tensor *tensor_squeeze_all(const Tensor *src)
{
    assert(src);
    Tensor *out = stack_memdup(src, 1, sizeof(*out));
    int ndim = 0;
    for (int i = 0; i < src->ndim; i++) {
        if (src->shape[i] != 1) {
            out->shape[ndim] = src->shape[i];
            out->stride[ndim] = src->stride[i];
            ndim += 1;
        }
    }
    out->ndim = ndim;
    return out;
}

Tensor *tensor_unsqueeze(const Tensor *src, int dim)
{
    assert(src && src->ndim < MAX_NDIM);
    dim = normalize_dim(dim, src->ndim + 1);
    Tensor *out = stack_memdup(src, 1, sizeof(*out));
    out->ndim += 1;
    for (int i = src->ndim; i > dim; i--) {
        out->shape[i] = src->shape[i - 1];
        out->stride[i] = src->stride[i - 1];
    }
    out->shape[dim] = 1;
    out->stride[dim] = (dim < src->ndim) ? src->stride[dim] : 1;
    return out;
}

static int valid_permute(const Tensor *src, const int *order)
{
    int seen[MAX_NDIM] = {0};
    for (int i = 0; i < src->ndim; i++) {
        if (order[i] < 0 || order[i] >= src->ndim || seen[order[i]]) {
            return 0;
        }
        seen[order[i]] = 1;
    }
    return 1;
}

Tensor *tensor_permute(const Tensor *src, const int *order_)
{
    assert(src && order_);
    int order[MAX_NDIM];
    for (int i = 0; i < src->ndim; i++) {
        order[i] = normalize_dim(order_[i], src->ndim);
    }
    assert(valid_permute(src, order));
    Tensor *out = stack_memdup(src, 1, sizeof(*out));
    for (int i = 0; i < src->ndim; i++) {
        out->shape[i] = src->shape[order[i]];
        out->stride[i] = src->stride[order[i]];
    }
    return out;
}

Tensor *tensor_transpose(const Tensor *src, int dim0, int dim1)
{
    assert(src);
    dim0 = normalize_dim(dim0, src->ndim);
    dim1 = normalize_dim(dim1, src->ndim);
    Tensor *out = stack_memdup(src, 1, sizeof(*out));
    out->shape[dim0] = src->shape[dim1];
    out->shape[dim1] = src->shape[dim0];
    out->stride[dim0] = src->stride[dim1];
    out->stride[dim1] = src->stride[dim0];
    return out;
}

Tensor *tensor_slice(const Tensor *src, int dim, int beg, int end, int step)
{
    assert(src && step != 0);
    dim = normalize_dim(dim, src->ndim);
    int size = src->shape[dim];
    if (beg == INT_MIN) {
        beg = (step > 0) ? 0 : (size - 1);
    }
    if (end == INT_MAX) {
        end = (step > 0) ? size : -1;
    }
    if (beg < 0) {
        beg += size;
    }
    if (end < 0 && end != -1) {
        end += size;
    }
    assert(0 <= beg && beg < size);
    if (step > 0) {
        assert(beg <= end && end <= size);
    }
    else {
        assert(-1 <= end && end <= beg);
    }
    Tensor *out = stack_memdup(src, 1, sizeof(*out));
    out->data += beg * src->stride[dim];
    if (step > 0) {
        out->shape[dim] = (end - beg + step - 1) / step;
    }
    else {
        out->shape[dim] = (end - beg + step + 1) / step;
    }
    out->stride[dim] *= step;
    out->numel = compute_numel(out->shape, out->ndim);
    return out;
}

Tensor *tensor_select(const Tensor *src, int dim, int index)
{
    dim = normalize_dim(dim, src->ndim);
    index = normalize_dim(index, src->shape[dim]);
    return tensor_squeeze(tensor_slice(src, dim, index, index + 1, 1), dim);
}

static int valid_expand(const Tensor *src, const int *shape, int ndim)
{
    if (ndim < src->ndim) {
        return 0;
    }
    int offset = ndim - src->ndim;
    for (int i = 0; i < src->ndim; i++) {
        if (src->shape[i] != 1 && src->shape[i] != shape[offset + i]) {
            return 0;
        }
    }
    return 1;
}

Tensor *tensor_expand(const Tensor *src, const int *shape_, int ndim)
{
    assert(src && ndim >= 0 && ndim <= MAX_NDIM);
    int shape[MAX_NDIM];
    int offset = ndim - src->ndim;
    for (int i = 0; i < ndim; i++) {
        if (i >= offset && shape_[i] == -1) {
            shape[i] = src->shape[i - offset];
        }
        else {
            shape[i] = shape_[i];
        }
    }
    assert(valid_shape(shape, ndim) && valid_expand(src, shape, ndim));
    Tensor *out = stack_memdup(src, 1, sizeof(*out));
    out->ndim = ndim;
    out->numel = compute_numel(shape, ndim);
    memcpy(out->shape, shape, ndim * sizeof(*shape));
    for (int i = 0; i < ndim; i++) {
        if (i < offset) {
            out->stride[i] = 0;
        }
        else {
            if (src->shape[i - offset] == shape[i]) {
                out->stride[i] = src->stride[i - offset];
            }
            else {
                out->stride[i] = 0;
            }
        }
    }
    return out;
}

static int valid_cat(const Tensor **src, int num, int dim)
{
    int ndim = src[0]->ndim;
    for (int i = 1; i < num; i++) {
        if (!src[i] || src[i]->ndim != ndim) {
            return 0;
        }
        for (int j = 0; j < ndim; j++) {
            if (j != dim && src[i]->shape[j] != src[0]->shape[j]) {
                return 0;
            }
        }
    }
    return 1;
}

static void cat_data(Tensor *out, long off_out, const Tensor *src, long off_src, int dim)
{
    if (dim == src->ndim - 1) {
        if (src->stride[dim] == 1 && out->stride[dim] == 1) {
            memcpy(out->data + off_out, src->data + off_src, src->shape[dim] * sizeof(*src->data));
        }
        else {
            for (int i = 0; i < src->shape[dim]; i++) {
                out->data[off_out + (i * out->stride[dim])] =
                    src->data[off_src + (i * src->stride[dim])];
            }
        }
    }
    else {
        for (int i = 0; i < src->shape[dim]; i++) {
            cat_data(out, off_out + (i * out->stride[dim]), src, off_src + (i * src->stride[dim]),
                     dim + 1);
        }
    }
}

Tensor *tensor_cat(const Tensor **src, int num, int dim)
{
    assert(src && src[0] && src[0]->ndim > 0 && num > 0);
    int ndim = src[0]->ndim;
    dim = normalize_dim(dim, ndim);
    assert(valid_cat(src, num, dim));
    int shape[MAX_NDIM];
    memcpy(shape, src[0]->shape, ndim * sizeof(*src[0]->shape));
    for (int i = 1; i < num; i++) {
        shape[dim] += src[i]->shape[dim];
    }
    Tensor *out = tensor_empty(shape, ndim);
    long off_out = 0;
    for (int i = 0; i < num; i++) {
        cat_data(out, off_out * out->stride[dim], src[i], 0, 0);
        off_out += src[i]->shape[dim];
    }
    return out;
}

Tensor *tensor_stack(const Tensor **src, int num, int dim)
{
    assert(src && src[0] && num > 0);
    const Tensor *tmp[num];
    for (int i = 0; i < num; i++) {
        tmp[i] = tensor_unsqueeze(src[i], dim);
    }
    return tensor_cat(tmp, num, dim);
}

// unary

typedef void Unary(float *, const float *, long, int);

static void unary_neg(float *out, const float *src, long str_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = -src[i * str_src];
    }
}

static void unary_abs(float *out, const float *src, long str_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = fabsf(src[i * str_src]);
    }
}

static void unary_sign(float *out, const float *src, long str_src, int num)
{
    for (int i = 0; i < num; i++) {
        float val = src[i * str_src];
        out[i] = (float)((val > 0) - (val < 0));
    }
}

static void unary_square(float *out, const float *src, long str_src, int num)
{
    for (int i = 0; i < num; i++) {
        float val = src[i * str_src];
        out[i] = val * val;
    }
}

static void unary_sqrt(float *out, const float *src, long str_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = sqrtf(src[i * str_src]);
    }
}

static void unary_rsqrt(float *out, const float *src, long str_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = 1 / sqrtf(src[i * str_src]);
    }
}

static void unary_exp(float *out, const float *src, long str_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = expf(src[i * str_src]);
    }
}

static void unary_log(float *out, const float *src, long str_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = logf(src[i * str_src]);
    }
}

static void unary_relu(float *out, const float *src, long str_src, int num)
{
    for (int i = 0; i < num; i++) {
        float val = src[i * str_src];
        out[i] = (val > 0) ? val : 0;
    }
}

static void unary_sigmoid(float *out, const float *src, long str_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = 1 / (1 + expf(-src[i * str_src]));
    }
}

static void unary_tanh(float *out, const float *src, long str_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = tanhf(src[i * str_src]);
    }
}

static void unary_logical_not(float *out, const float *src, long str_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = (src[i * str_src] == 0) ? 1 : 0;
    }
}

static void apply_unary(Tensor *out, long off_out, const Tensor *src, long off_src, int dim,
                        Unary *func)
{
    if (dim == out->ndim - 1) {
        func(out->data + off_out, src->data + off_src, src->stride[dim], out->shape[dim]);
    }
    else {
        for (int i = 0; i < out->shape[dim]; i++) {
            apply_unary(out, off_out + (i * out->stride[dim]), src,
                        off_src + (i * src->stride[dim]), dim + 1, func);
        }
    }
}

static Tensor *unary(const Tensor *src, Unary *func)
{
    assert(src && func);
    Tensor *out = tensor_empty(src->shape, src->ndim);
    if (src->ndim == 0) {
        func(out->data, src->data, 0, 1);
    }
    else {
        apply_unary(out, 0, src, 0, 0, func);
    }
    return out;
}

Tensor *tensor_neg(const Tensor *src)
{
    return unary(src, unary_neg);
}

Tensor *tensor_abs(const Tensor *src)
{
    return unary(src, unary_abs);
}

Tensor *tensor_sign(const Tensor *src)
{
    return unary(src, unary_sign);
}

Tensor *tensor_square(const Tensor *src)
{
    return unary(src, unary_square);
}

Tensor *tensor_sqrt(const Tensor *src)
{
    return unary(src, unary_sqrt);
}

Tensor *tensor_rsqrt(const Tensor *src)
{
    return unary(src, unary_rsqrt);
}

Tensor *tensor_exp(const Tensor *src)
{
    return unary(src, unary_exp);
}

Tensor *tensor_log(const Tensor *src)
{
    return unary(src, unary_log);
}

Tensor *tensor_relu(const Tensor *src)
{
    return unary(src, unary_relu);
}

Tensor *tensor_sigmoid(const Tensor *src)
{
    return unary(src, unary_sigmoid);
}

Tensor *tensor_tanh(const Tensor *src)
{
    return unary(src, unary_tanh);
}

Tensor *tensor_logical_not(const Tensor *src)
{
    return unary(src, unary_logical_not);
}

// binary

typedef void Binary(float *, const float *, long, const float *, long, int);

static void binary_add(float *out, const float *lhs, long str_lhs, const float *rhs, long str_rhs,
                       int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = lhs[i * str_lhs] + rhs[i * str_rhs];
    }
}

static void binary_sub(float *out, const float *lhs, long str_lhs, const float *rhs, long str_rhs,
                       int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = lhs[i * str_lhs] - rhs[i * str_rhs];
    }
}

static void binary_mul(float *out, const float *lhs, long str_lhs, const float *rhs, long str_rhs,
                       int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = lhs[i * str_lhs] * rhs[i * str_rhs];
    }
}

static void binary_div(float *out, const float *lhs, long str_lhs, const float *rhs, long str_rhs,
                       int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = lhs[i * str_lhs] / rhs[i * str_rhs];
    }
}

static void binary_mod(float *out, const float *lhs, long str_lhs, const float *rhs, long str_rhs,
                       int num)
{
    for (int i = 0; i < num; i++) {
        float val_rhs = rhs[i * str_rhs];
        float mod = fmodf(lhs[i * str_lhs], val_rhs);
        out[i] = (mod != 0 && (mod < 0) != (val_rhs < 0)) ? (mod + val_rhs) : mod;
    }
}

static void binary_pow(float *out, const float *lhs, long str_lhs, const float *rhs, long str_rhs,
                       int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = powf(lhs[i * str_lhs], rhs[i * str_rhs]);
    }
}

static void binary_eq(float *out, const float *lhs, long str_lhs, const float *rhs, long str_rhs,
                      int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = (lhs[i * str_lhs] == rhs[i * str_rhs]) ? 1 : 0;
    }
}

static void binary_ne(float *out, const float *lhs, long str_lhs, const float *rhs, long str_rhs,
                      int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = (lhs[i * str_lhs] != rhs[i * str_rhs]) ? 1 : 0;
    }
}

static void binary_lt(float *out, const float *lhs, long str_lhs, const float *rhs, long str_rhs,
                      int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = (lhs[i * str_lhs] < rhs[i * str_rhs]) ? 1 : 0;
    }
}

static void binary_le(float *out, const float *lhs, long str_lhs, const float *rhs, long str_rhs,
                      int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = (lhs[i * str_lhs] <= rhs[i * str_rhs]) ? 1 : 0;
    }
}

static void binary_gt(float *out, const float *lhs, long str_lhs, const float *rhs, long str_rhs,
                      int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = (lhs[i * str_lhs] > rhs[i * str_rhs]) ? 1 : 0;
    }
}

static void binary_ge(float *out, const float *lhs, long str_lhs, const float *rhs, long str_rhs,
                      int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = (lhs[i * str_lhs] >= rhs[i * str_rhs]) ? 1 : 0;
    }
}

static void binary_logical_and(float *out, const float *lhs, long str_lhs, const float *rhs,
                               long str_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = ((lhs[i * str_lhs] != 0) && (rhs[i * str_rhs] != 0)) ? 1 : 0;
    }
}

static void binary_logical_or(float *out, const float *lhs, long str_lhs, const float *rhs,
                              long str_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = ((lhs[i * str_lhs] != 0) || (rhs[i * str_rhs] != 0)) ? 1 : 0;
    }
}

static void binary_logical_xor(float *out, const float *lhs, long str_lhs, const float *rhs,
                               long str_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = ((lhs[i * str_lhs] != 0) != (rhs[i * str_rhs] != 0)) ? 1 : 0;
    }
}

static void binary_minimum(float *out, const float *lhs, long str_lhs, const float *rhs,
                           long str_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = fminf(lhs[i * str_lhs], rhs[i * str_rhs]);
    }
}

static void binary_maximum(float *out, const float *lhs, long str_lhs, const float *rhs,
                           long str_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = fmaxf(lhs[i * str_lhs], rhs[i * str_rhs]);
    }
}

static int broadcast_shape(int *shape, const Tensor *lhs, const Tensor *rhs)
{
    int ndim = (lhs->ndim > rhs->ndim) ? lhs->ndim : rhs->ndim;
    int off_lhs = ndim - lhs->ndim;
    int off_rhs = ndim - rhs->ndim;
    for (int i = 0; i < ndim; i++) {
        int dim_lhs = (i >= off_lhs) ? lhs->shape[i - off_lhs] : 1;
        int dim_rhs = (i >= off_rhs) ? rhs->shape[i - off_rhs] : 1;
        assert(dim_lhs == dim_rhs || dim_lhs == 1 || dim_rhs == 1);
        shape[i] = (dim_lhs > dim_rhs) ? dim_lhs : dim_rhs;
    }
    return ndim;
}

static void apply_binary(Tensor *out, long off_out, const Tensor *lhs, long off_lhs,
                         const Tensor *rhs, long off_rhs, int dim, Binary *func)
{
    if (dim == out->ndim - 1) {
        func(out->data + off_out, lhs->data + off_lhs, lhs->stride[dim], rhs->data + off_rhs,
             rhs->stride[dim], out->shape[dim]);
    }
    else {
        for (int i = 0; i < out->shape[dim]; i++) {
            apply_binary(out, off_out + (i * out->stride[dim]), lhs,
                         off_lhs + (i * lhs->stride[dim]), rhs, off_rhs + (i * rhs->stride[dim]),
                         dim + 1, func);
        }
    }
}

static Tensor *binary(const Tensor *lhs_, const Tensor *rhs_, Binary *func)
{
    assert(lhs_ && rhs_ && func);
    int shape[MAX_NDIM];
    int ndim = broadcast_shape(shape, lhs_, rhs_);
    Tensor *lhs = tensor_expand(lhs_, shape, ndim);
    Tensor *rhs = tensor_expand(rhs_, shape, ndim);
    Tensor *out = tensor_empty(shape, ndim);
    if (ndim == 0) {
        func(out->data, lhs->data, 0, rhs->data, 0, 1);
    }
    else {
        apply_binary(out, 0, lhs, 0, rhs, 0, 0, func);
    }
    return out;
}

Tensor *tensor_add(const Tensor *lhs, const Tensor *rhs)
{
    return binary(lhs, rhs, binary_add);
}

Tensor *tensor_sub(const Tensor *lhs, const Tensor *rhs)
{
    return binary(lhs, rhs, binary_sub);
}

Tensor *tensor_mul(const Tensor *lhs, const Tensor *rhs)
{
    return binary(lhs, rhs, binary_mul);
}

Tensor *tensor_div(const Tensor *lhs, const Tensor *rhs)
{
    return binary(lhs, rhs, binary_div);
}

Tensor *tensor_mod(const Tensor *lhs, const Tensor *rhs)
{
    return binary(lhs, rhs, binary_mod);
}

Tensor *tensor_pow(const Tensor *lhs, const Tensor *rhs)
{
    return binary(lhs, rhs, binary_pow);
}

Tensor *tensor_eq(const Tensor *lhs, const Tensor *rhs)
{
    return binary(lhs, rhs, binary_eq);
}

Tensor *tensor_ne(const Tensor *lhs, const Tensor *rhs)
{
    return binary(lhs, rhs, binary_ne);
}

Tensor *tensor_lt(const Tensor *lhs, const Tensor *rhs)
{
    return binary(lhs, rhs, binary_lt);
}

Tensor *tensor_le(const Tensor *lhs, const Tensor *rhs)
{
    return binary(lhs, rhs, binary_le);
}

Tensor *tensor_gt(const Tensor *lhs, const Tensor *rhs)
{
    return binary(lhs, rhs, binary_gt);
}

Tensor *tensor_ge(const Tensor *lhs, const Tensor *rhs)
{
    return binary(lhs, rhs, binary_ge);
}

Tensor *tensor_logical_and(const Tensor *lhs, const Tensor *rhs)
{
    return binary(lhs, rhs, binary_logical_and);
}

Tensor *tensor_logical_or(const Tensor *lhs, const Tensor *rhs)
{
    return binary(lhs, rhs, binary_logical_or);
}

Tensor *tensor_logical_xor(const Tensor *lhs, const Tensor *rhs)
{
    return binary(lhs, rhs, binary_logical_xor);
}

Tensor *tensor_minimum(const Tensor *lhs, const Tensor *rhs)
{
    return binary(lhs, rhs, binary_minimum);
}

Tensor *tensor_maximum(const Tensor *lhs, const Tensor *rhs)
{
    return binary(lhs, rhs, binary_maximum);
}

// i/o

static void print_data(const Tensor *self, long off, int dim)
{
    printf("[");
    if (dim == self->ndim - 1) {
        for (int i = 0; i < self->shape[dim]; i++) {
            printf("%g", self->data[off + (i * self->stride[dim])]);
            if (i < self->shape[dim] - 1) {
                printf(" ");
            }
        }
    }
    else {
        for (int i = 0; i < self->shape[dim]; i++) {
            if (i > 0) {
                for (int j = 0; j < self->ndim - dim - 1; j++) {
                    printf("\n");
                }
                for (int j = 0; j <= dim; j++) {
                    printf(" ");
                }
            }
            print_data(self, off + (i * self->stride[dim]), dim + 1);
        }
    }
    printf("]");
}

void tensor_print(const Tensor *self)
{
    assert(self);
    printf("Tensor(ndim=%d, numel=%ld, shape=[", self->ndim, self->numel);
    for (int i = 0; i < self->ndim; i++) {
        printf("%d%s", self->shape[i], (i < self->ndim - 1) ? ", " : "");
    }
    printf("], stride=[");
    for (int i = 0; i < self->ndim; i++) {
        printf("%ld%s", self->stride[i], (i < self->ndim - 1) ? ", " : "");
    }
    printf("])\n");
    if (self->ndim == 0) {
        printf("%g", self->data[0]);
    }
    else {
        print_data(self, 0, 0);
    }
    printf("\n\n");
}

// test

#define ensure(cond)                                                                         \
    do {                                                                                     \
        if (!(cond)) {                                                                       \
            fprintf(stderr, "%s:%d %s: `%s` failed\n", __FILE__, __LINE__, __func__, #cond); \
            abort();                                                                         \
        }                                                                                    \
    } while (0)

#define isclose(a, b) (fabsf((a) - (b)) < FLT_EPSILON)

// NOLINTBEGIN(readability-identifier-length)

static void test_creation(void)
{
    stack_save();

    // tensor_empty
    Tensor *t = tensor_empty((int[]){2, 3}, 2);
    ensure(t->ndim == 2 && t->numel == 6);
    ensure(t->shape[0] == 2 && t->shape[1] == 3);
    ensure(t->stride[0] == 3 && t->stride[1] == 1);

    // tensor_empty 0-dim
    t = tensor_empty(0, 0);
    ensure(t->ndim == 0 && t->numel == 1);

    // tensor_zeros
    t = tensor_zeros((int[]){3}, 1);
    ensure(t->data[0] == 0 && t->data[1] == 0 && t->data[2] == 0);

    // tensor_ones
    t = tensor_ones((int[]){3}, 1);
    ensure(t->data[0] == 1 && t->data[1] == 1 && t->data[2] == 1);

    // tensor_fill
    t = tensor_fill((int[]){3}, 1, 5);
    ensure(t->data[0] == 5 && t->data[1] == 5 && t->data[2] == 5);

    // tensor_from
    float data[] = {1, 2, 3, 4};
    t = tensor_from((int[]){4}, 1, data);
    ensure(t->data[0] == 1 && t->data[3] == 4);
    data[0] = 99;
    ensure(t->data[0] == 1);  // copy, not a reference

    // tensor_arange: [1, 4) -> [1, 2, 3]
    t = tensor_arange(1, 4, 1);
    ensure(t->numel == 3);
    ensure(t->data[0] == 1 && t->data[1] == 2 && t->data[2] == 3);

    // tensor_arange: negative step (3, 0, -1) -> [3, 2, 1]
    t = tensor_arange(3, 0, -1);
    ensure(t->numel == 3);
    ensure(t->data[0] == 3 && t->data[1] == 2 && t->data[2] == 1);

    // tensor_range: [1, 3] -> [1, 2, 3]
    t = tensor_range(1, 3, 1);
    ensure(t->numel == 3);
    ensure(t->data[0] == 1 && t->data[1] == 2 && t->data[2] == 3);

    // tensor_linspace: 0 to 1, 3 steps -> [0, 0.5, 1]
    t = tensor_linspace(0, 1, 3);
    ensure(t->numel == 3);
    ensure(t->data[0] == 0 && isclose(t->data[1], 0.5F) && t->data[2] == 1);

    // tensor_logspace: base 10, 0 to 2, 3 steps -> [1, 10, 100]
    t = tensor_logspace(10, 0, 2, 3);
    ensure(t->numel == 3);
    ensure(isclose(t->data[0], 1) && isclose(t->data[1], 10) && isclose(t->data[2], 100));

    // tensor_eye: 3x2, diagonal is 1
    t = tensor_eye(3, 2);
    ensure(t->shape[0] == 3 && t->shape[1] == 2);
    ensure(t->data[0] == 1 && t->data[1] == 0);  // row 0
    ensure(t->data[2] == 0 && t->data[3] == 1);  // row 1
    ensure(t->data[4] == 0 && t->data[5] == 0);  // row 2

    // tensor_rand: values in [0, 1)
    t = tensor_rand((int[]){100}, 1);
    ensure(t->numel == 100);
    for (long i = 0; i < t->numel; i++) {
        ensure(t->data[i] >= 0 && t->data[i] < 1);
    }

    // tensor_randn: correct shape
    t = tensor_randn((int[]){100}, 1);
    ensure(t->numel == 100);

    stack_restore();
}

static void test_movement(void)
{
    stack_save();

    // tensor_reshape: [6] -> [2, 3]
    Tensor *t = tensor_arange(1, 7, 1);
    t = tensor_reshape(t, (int[]){2, 3}, 2);
    ensure(t->ndim == 2 && t->shape[0] == 2 && t->shape[1] == 3);
    ensure(t->data[0] == 1 && t->data[5] == 6);

    // tensor_reshape: infer dim with -1
    t = tensor_arange(1, 7, 1);
    t = tensor_reshape(t, (int[]){-1, 3}, 2);
    ensure(t->shape[0] == 2 && t->shape[1] == 3);

    // tensor_flatten: dims 0..1 of [2, 3] -> [6]
    t = tensor_arange(1, 7, 1);
    t = tensor_reshape(t, (int[]){2, 3}, 2);
    t = tensor_flatten(t, 0, 1);
    ensure(t->ndim == 1 && t->numel == 6);

    // tensor_flatten_all
    t = tensor_arange(1, 7, 1);
    t = tensor_reshape(t, (int[]){2, 3}, 2);
    t = tensor_flatten_all(t);
    ensure(t->ndim == 1 && t->numel == 6);

    // tensor_unflatten: [6] -> [2, 3]
    t = tensor_arange(1, 7, 1);
    t = tensor_unflatten(t, 0, (int[]){2, 3}, 2);
    ensure(t->ndim == 2 && t->shape[0] == 2 && t->shape[1] == 3);

    // tensor_squeeze: [1, 3, 1] remove dim 0 -> [3, 1]
    t = tensor_zeros((int[]){1, 3, 1}, 3);
    t = tensor_squeeze(t, 0);
    ensure(t->ndim == 2 && t->shape[0] == 3 && t->shape[1] == 1);

    // tensor_squeeze: negative dim
    t = tensor_zeros((int[]){1, 3, 1}, 3);
    t = tensor_squeeze(t, -1);
    ensure(t->ndim == 2 && t->shape[0] == 1 && t->shape[1] == 3);

    // tensor_squeeze_all: [1, 3, 1] -> [3]
    t = tensor_zeros((int[]){1, 3, 1}, 3);
    t = tensor_squeeze_all(t);
    ensure(t->ndim == 1 && t->shape[0] == 3);

    // tensor_unsqueeze: [3] insert at 0 -> [1, 3]
    t = tensor_zeros((int[]){3}, 1);
    t = tensor_unsqueeze(t, 0);
    ensure(t->ndim == 2 && t->shape[0] == 1 && t->shape[1] == 3);

    // tensor_permute: [2, 3] -> [3, 2], check strides swapped
    t = tensor_arange(1, 7, 1);
    t = tensor_reshape(t, (int[]){2, 3}, 2);
    t = tensor_permute(t, (int[]){1, 0});
    ensure(t->shape[0] == 3 && t->shape[1] == 2);
    ensure(t->stride[0] == 1 && t->stride[1] == 3);

    // tensor_transpose: same as permute for 2D
    t = tensor_arange(1, 7, 1);
    t = tensor_reshape(t, (int[]){2, 3}, 2);
    t = tensor_transpose(t, 0, 1);
    ensure(t->shape[0] == 3 && t->shape[1] == 2);
    ensure(t->stride[0] == 1 && t->stride[1] == 3);
    // element [1][0]: data[1*1 + 0*3] = data[1] = 2
    ensure(t->data[(t->stride[0] * 1) + (t->stride[1] * 0)] == 2);

    // tensor_slice: [0,1,2,3,4,5] slice [1:4] -> [1,2,3]
    t = tensor_arange(0, 6, 1);
    t = tensor_slice(t, 0, 1, 4, 1);
    ensure(t->numel == 3);
    ensure(t->data[0] == 1 && t->data[1] == 2 && t->data[2] == 3);

    // tensor_slice: reverse step, stride is -1, access via stride
    t = tensor_arange(0, 6, 1);
    t = tensor_slice(t, 0, INT_MIN, INT_MAX, -1);
    ensure(t->numel == 6 && t->stride[0] == -1);
    ensure(t->data[0] == 5 && t->data[5 * t->stride[0]] == 0);

    // tensor_slice: negative start index
    t = tensor_arange(0, 6, 1);
    t = tensor_slice(t, 0, -4, -2, 1);  // beg=-4->2, end=-2->4 -> [2, 3]
    ensure(t->numel == 2 && t->data[0] == 2 && t->data[1] == 3);

    // tensor_select: row 1 of [2, 3] -> [4, 5, 6]
    t = tensor_arange(1, 7, 1);
    t = tensor_reshape(t, (int[]){2, 3}, 2);
    t = tensor_select(t, 0, 1);
    ensure(t->ndim == 1 && t->numel == 3);
    ensure(t->data[0] == 4 && t->data[1] == 5 && t->data[2] == 6);

    // tensor_select: negative index
    t = tensor_arange(1, 7, 1);
    t = tensor_reshape(t, (int[]){2, 3}, 2);
    t = tensor_select(t, 0, -1);
    ensure(t->data[0] == 4 && t->data[2] == 6);

    // tensor_expand: [1, 3] -> [2, 3], broadcast dim gets stride 0
    t = tensor_from((int[]){1, 3}, 2, (float[]){1, 2, 3});
    t = tensor_expand(t, (int[]){2, 3}, 2);
    ensure(t->shape[0] == 2 && t->shape[1] == 3 && t->stride[0] == 0);
    ensure(t->data[t->stride[1] * 0] == 1 && t->data[t->stride[1] * 2] == 3);

    // tensor_expand: -1 preserves dim
    t = tensor_from((int[]){1, 3}, 2, (float[]){1, 2, 3});
    t = tensor_expand(t, (int[]){2, -1}, 2);
    ensure(t->shape[0] == 2 && t->shape[1] == 3);

    // tensor_cat: [[1,2,3], [4,5,6]] along dim 0 -> [1..6]
    Tensor *a = tensor_arange(1, 4, 1);
    Tensor *b = tensor_arange(4, 7, 1);
    t = tensor_cat((const Tensor *[]){a, b}, 2, 0);
    ensure(t->ndim == 1 && t->numel == 6);
    ensure(t->data[0] == 1 && t->data[5] == 6);

    // tensor_cat: along dim 1
    a = tensor_from((int[]){2, 2}, 2, (float[]){1, 2, 3, 4});
    b = tensor_from((int[]){2, 1}, 2, (float[]){5, 6});
    t = tensor_cat((const Tensor *[]){a, b}, 2, 1);
    ensure(t->shape[0] == 2 && t->shape[1] == 3);
    ensure(t->data[2] == 5);  // first element of appended column

    // tensor_stack: stack [1,2,3] and [4,5,6] along dim 0 -> [2, 3]
    a = tensor_arange(1, 4, 1);
    b = tensor_arange(4, 7, 1);
    t = tensor_stack((const Tensor *[]){a, b}, 2, 0);
    ensure(t->ndim == 2 && t->shape[0] == 2 && t->shape[1] == 3);
    ensure(t->data[0] == 1 && t->data[3] == 4);

    stack_restore();
}

static void test_unary(void)
{
    stack_save();

    // tensor_neg
    Tensor *t = tensor_from((int[]){3}, 1, (float[]){-1, 0, 2});
    Tensor *out = tensor_neg(t);
    ensure(out->data[0] == 1 && out->data[1] == 0 && out->data[2] == -2);

    // tensor_abs
    t = tensor_from((int[]){3}, 1, (float[]){-2, 0, 3});
    out = tensor_abs(t);
    ensure(out->data[0] == 2 && out->data[1] == 0 && out->data[2] == 3);

    // tensor_sign
    t = tensor_from((int[]){3}, 1, (float[]){-5, 0, 3});
    out = tensor_sign(t);
    ensure(out->data[0] == -1 && out->data[1] == 0 && out->data[2] == 1);

    // tensor_square
    t = tensor_from((int[]){3}, 1, (float[]){-2, 3, 4});
    out = tensor_square(t);
    ensure(out->data[0] == 4 && out->data[1] == 9 && out->data[2] == 16);

    // tensor_sqrt
    t = tensor_from((int[]){3}, 1, (float[]){0, 1, 4});
    out = tensor_sqrt(t);
    ensure(out->data[0] == 0 && out->data[1] == 1 && isclose(out->data[2], 2));

    // tensor_rsqrt
    t = tensor_from((int[]){2}, 1, (float[]){1, 4});
    out = tensor_rsqrt(t);
    ensure(isclose(out->data[0], 1) && isclose(out->data[1], 0.5F));

    // tensor_exp: exp(0)=1, exp(1)~=e
    static const float e = 2.7182818284590452354F;
    t = tensor_from((int[]){2}, 1, (float[]){0, 1});
    out = tensor_exp(t);
    ensure(out->data[0] == expf(0) && isclose(out->data[1], e));

    // tensor_log: log(1)=0, log(e)~=1
    t = tensor_from((int[]){2}, 1, (float[]){1, e});
    out = tensor_log(t);
    ensure(out->data[0] == logf(1) && isclose(out->data[1], 1));

    // tensor_relu
    t = tensor_from((int[]){3}, 1, (float[]){-1, 0, 2});
    out = tensor_relu(t);
    ensure(out->data[0] == 0 && out->data[1] == 0 && out->data[2] == 2);

    // tensor_sigmoid: sigmoid(0)=0.5, sigmoid(large)~=1
    t = tensor_from((int[]){2}, 1, (float[]){0, 1e6F});
    out = tensor_sigmoid(t);
    ensure(isclose(out->data[0], 0.5F) && isclose(out->data[1], 1));

    // tensor_tanh: tanh(0)=0, tanh(large)~=1
    t = tensor_from((int[]){2}, 1, (float[]){0, 1e6F});
    out = tensor_tanh(t);
    ensure(isclose(out->data[0], 0) && isclose(out->data[1], 1));

    // tensor_logical_not: 0->1, nonzero->0
    t = tensor_from((int[]){3}, 1, (float[]){0, 1, -5});
    out = tensor_logical_not(t);
    ensure(out->data[0] == 1 && out->data[1] == 0 && out->data[2] == 0);

    // non-contiguous input (transposed)
    t = tensor_from((int[]){2, 2}, 2, (float[]){1, 2, 3, 4});
    t = tensor_transpose(t, 0, 1);
    out = tensor_neg(t);
    ensure(out->data[0] == -1 && out->data[1] == -3 && out->data[2] == -2 && out->data[3] == -4);

    stack_restore();
}

static void test_binary(void)
{
    stack_save();

    Tensor *a = tensor_from((int[]){3}, 1, (float[]){1, 2, 3});
    Tensor *b = tensor_from((int[]){3}, 1, (float[]){4, 5, 6});

    // tensor_add
    Tensor *out = tensor_add(a, b);
    ensure(out->data[0] == 5 && out->data[1] == 7 && out->data[2] == 9);

    // tensor_sub
    out = tensor_sub(b, a);
    ensure(out->data[0] == 3 && out->data[1] == 3 && out->data[2] == 3);

    // tensor_mul
    out = tensor_mul(a, b);
    ensure(out->data[0] == 4 && out->data[1] == 10 && out->data[2] == 18);

    // tensor_div
    out = tensor_div(b, a);
    ensure(isclose(out->data[0], 4) && isclose(out->data[1], 2.5F) && isclose(out->data[2], 2));

    // tensor_mod: Python convention - sign follows divisor
    a = tensor_from((int[]){4}, 1, (float[]){7, -7, 7, -7});
    b = tensor_from((int[]){4}, 1, (float[]){3, 3, -3, -3});
    out = tensor_mod(a, b);
    ensure(isclose(out->data[0], 1) && isclose(out->data[1], 2));    // 7%3=1, -7%3=2
    ensure(isclose(out->data[2], -2) && isclose(out->data[3], -1));  // 7%-3=-2, -7%-3=-1

    // tensor_pow
    a = tensor_from((int[]){3}, 1, (float[]){2, 3, 4});
    b = tensor_from((int[]){3}, 1, (float[]){2, 2, 2});
    out = tensor_pow(a, b);
    ensure(out->data[0] == 4 && out->data[1] == 9 && out->data[2] == 16);

    // tensor_eq / tensor_ne
    a = tensor_from((int[]){3}, 1, (float[]){1, 2, 3});
    b = tensor_from((int[]){3}, 1, (float[]){1, 0, 3});
    out = tensor_eq(a, b);
    ensure(out->data[0] == 1 && out->data[1] == 0 && out->data[2] == 1);
    out = tensor_ne(a, b);
    ensure(out->data[0] == 0 && out->data[1] == 1 && out->data[2] == 0);

    // tensor_lt / tensor_le / tensor_gt / tensor_ge
    out = tensor_lt(a, b);
    ensure(out->data[0] == 0 && out->data[1] == 0 && out->data[2] == 0);
    out = tensor_le(a, b);
    ensure(out->data[0] == 1 && out->data[1] == 0 && out->data[2] == 1);
    out = tensor_gt(a, b);
    ensure(out->data[0] == 0 && out->data[1] == 1 && out->data[2] == 0);
    out = tensor_ge(a, b);
    ensure(out->data[0] == 1 && out->data[1] == 1 && out->data[2] == 1);

    // tensor_logical_and / tensor_logical_or / tensor_logical_xor
    a = tensor_from((int[]){3}, 1, (float[]){1, 0, 1});
    b = tensor_from((int[]){3}, 1, (float[]){1, 1, 0});
    out = tensor_logical_and(a, b);
    ensure(out->data[0] == 1 && out->data[1] == 0 && out->data[2] == 0);
    out = tensor_logical_or(a, b);
    ensure(out->data[0] == 1 && out->data[1] == 1 && out->data[2] == 1);
    out = tensor_logical_xor(a, b);
    ensure(out->data[0] == 0 && out->data[1] == 1 && out->data[2] == 1);

    // tensor_minimum / tensor_maximum
    a = tensor_from((int[]){3}, 1, (float[]){1, 5, 3});
    b = tensor_from((int[]){3}, 1, (float[]){4, 2, 3});
    out = tensor_minimum(a, b);
    ensure(out->data[0] == 1 && out->data[1] == 2 && out->data[2] == 3);
    out = tensor_maximum(a, b);
    ensure(out->data[0] == 4 && out->data[1] == 5 && out->data[2] == 3);

    // broadcasting: [3] + [1] -> [3]
    a = tensor_from((int[]){3}, 1, (float[]){1, 2, 3});
    b = tensor_from((int[]){1}, 1, (float[]){10});
    out = tensor_add(a, b);
    ensure(out->data[0] == 11 && out->data[1] == 12 && out->data[2] == 13);

    // broadcasting: [2, 3] + [3] -> [2, 3]
    a = tensor_from((int[]){2, 3}, 2, (float[]){1, 2, 3, 4, 5, 6});
    b = tensor_from((int[]){3}, 1, (float[]){10, 20, 30});
    out = tensor_add(a, b);
    ensure(out->data[0] == 11 && out->data[3] == 14 && out->data[5] == 36);

    stack_restore();
}

// NOLINTEND(readability-identifier-length)

int main(void)
{
    test_creation();
    test_movement();
    test_unary();
    test_binary();
    stack_clear();
}
