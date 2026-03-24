#include "tensor.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void *stack_pop(void)
{
    assert(g_head);
    void *ptr = g_head->ptr;
    Stack *prev = g_head->prev;
    free(g_head);
    g_head = prev;
    return ptr;
}

static int g_index = 0;
static Stack *g_save[MAX_SAVE];

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

// tensor

enum { MAX_NDIM = 8 };

typedef struct {
    int num_inputs;
    Tensor *input[3];
    Tensor *saved[3];
    void (*backward)(Tensor *);
} Autograd;

struct tensor {
    int requires_grad;
    int ndim;
    long numel;
    int shape[MAX_NDIM];
    long stride[MAX_NDIM];
    float *data;
    float *grad;
    Autograd *ctx;
};

// context

void tensor_frame_begin(void)
{
    stack_save();
}

void tensor_frame_end(void)
{
    stack_restore();
}

static int g_grad_enabled = 1;

void tensor_no_grad_begin(void)
{
    g_grad_enabled = 0;
}

void tensor_no_grad_end(void)
{
    g_grad_enabled = 1;
}

// access

int tensor_ndim(const Tensor *self)
{
    assert(self);
    return self->ndim;
}

long tensor_numel(const Tensor *self)
{
    assert(self);
    return self->numel;
}

const int *tensor_shape(const Tensor *self)
{
    assert(self);
    return self->shape;
}

const long *tensor_stride(const Tensor *self)
{
    assert(self);
    return self->stride;
}

float *tensor_data(const Tensor *self)
{
    assert(self);
    return self->data;
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

Tensor *tensor_wrap(const int *shape, int ndim, float *data)
{
    assert(data);
    Tensor *out = stack_calloc(1, sizeof(*out));
    out->ndim = ndim;
    out->numel = compute_numel(shape, ndim);
    if (ndim > 0) {
        memcpy(out->shape, shape, ndim * sizeof(*shape));
        compute_stride(out->stride, out->shape, ndim);
    }
    out->data = data;
    return out;
}

Tensor *tensor_scalar(float value)
{
    return tensor_from(0, 0, &value);
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

static void pack_data(Tensor *out, long *offset_out, const Tensor *src, long offset_src, int dim)
{
    if (dim == src->ndim - 1) {
        if (src->stride[dim] == 1) {
            memcpy(out->data + *offset_out, src->data + offset_src,
                   src->shape[dim] * sizeof(*src->data));
        }
        else {
            for (int i = 0; i < src->shape[dim]; i++) {
                out->data[*offset_out + i] = src->data[offset_src + (i * src->stride[dim])];
            }
        }
        *offset_out += src->shape[dim];
    }
    else {
        for (int i = 0; i < src->shape[dim]; i++) {
            pack_data(out, offset_out, src, offset_src + (i * src->stride[dim]), dim + 1);
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
        long offset = 0;
        pack_data(out, &offset, src, 0, 0);
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
    beg_dim = (beg_dim == INT_MIN) ? 0 : normalize_dim(beg_dim, src->ndim);
    end_dim = (end_dim == INT_MAX) ? (src->ndim - 1) : normalize_dim(end_dim, src->ndim);
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
    Tensor *out = stack_memdup(src, 1, sizeof(*out));
    if (dim == INT_MAX) {
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
    dim = normalize_dim(dim, src->ndim);
    assert(src->shape[dim] == 1);
    out->ndim -= 1;
    for (int i = dim; i < src->ndim - 1; i++) {
        out->shape[i] = src->shape[i + 1];
        out->stride[i] = src->stride[i + 1];
    }
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

Tensor *tensor_flip(const Tensor *src, int dim)
{
    assert(src);
    dim = normalize_dim(dim, src->ndim);
    Tensor *out = stack_memdup(src, 1, sizeof(*out));
    out->data += (src->shape[dim] - 1) * src->stride[dim];
    out->stride[dim] = -src->stride[dim];
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

static void cat_data(Tensor *out, long offset_out, const Tensor *src, long offset_src, int dim)
{
    if (dim == src->ndim - 1) {
        if (src->stride[dim] == 1 && out->stride[dim] == 1) {
            memcpy(out->data + offset_out, src->data + offset_src,
                   src->shape[dim] * sizeof(*src->data));
        }
        else {
            for (int i = 0; i < src->shape[dim]; i++) {
                out->data[offset_out + (i * out->stride[dim])] =
                    src->data[offset_src + (i * src->stride[dim])];
            }
        }
    }
    else {
        for (int i = 0; i < src->shape[dim]; i++) {
            cat_data(out, offset_out + (i * out->stride[dim]), src,
                     offset_src + (i * src->stride[dim]), dim + 1);
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
    long offset = 0;
    for (int i = 0; i < num; i++) {
        cat_data(out, offset * out->stride[dim], src[i], 0, 0);
        offset += src[i]->shape[dim];
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

Tensor *tensor_contiguous(const Tensor *src)
{
    return tensor_reshape(src, src->shape, src->ndim);
}

Tensor *tensor_clone(const Tensor *src)
{
    assert(src);
    Tensor *out = tensor_empty(src->shape, src->ndim);
    if (is_contiguous(src)) {
        memcpy(out->data, src->data, src->numel * sizeof(*src->data));
    }
    else {
        long offset = 0;
        pack_data(out, &offset, src, 0, 0);
    }
    return out;
}

Tensor *tensor_detach(const Tensor *src)
{
    assert(src);
    Tensor *out = stack_memdup(src, 1, sizeof(*out));
    out->requires_grad = 0;
    out->grad = 0;
    out->ctx = 0;
    return out;
}

// unary

typedef void Unary(float *, const float *, long, int);

static void unary_neg(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = -src[i * stride_src];
    }
}

static void unary_abs(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = fabsf(src[i * stride_src]);
    }
}

static void unary_sign(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        float val = src[i * stride_src];
        out[i] = (float)((val > 0) - (val < 0));
    }
}

static void unary_square(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        float val = src[i * stride_src];
        out[i] = val * val;
    }
}

static void unary_sqrt(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = sqrtf(src[i * stride_src]);
    }
}

static void unary_rsqrt(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = 1 / sqrtf(src[i * stride_src]);
    }
}

static void unary_exp(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = expf(src[i * stride_src]);
    }
}

static void unary_sin(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = sinf(src[i * stride_src]);
    }
}

static void unary_cos(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = cosf(src[i * stride_src]);
    }
}

static void unary_tan(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = tanf(src[i * stride_src]);
    }
}

static void unary_log(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = logf(src[i * stride_src]);
    }
}

static void unary_floor(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = floorf(src[i * stride_src]);
    }
}

static void unary_ceil(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = ceilf(src[i * stride_src]);
    }
}

static void unary_round(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = roundf(src[i * stride_src]);
    }
}

static void unary_relu(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        float val = src[i * stride_src];
        out[i] = (val > 0) ? val : 0;
    }
}

static void unary_sigmoid(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = 1 / (1 + expf(-src[i * stride_src]));
    }
}

static void unary_tanh(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = tanhf(src[i * stride_src]);
    }
}

static void unary_logical_not(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = (src[i * stride_src] == 0) ? 1 : 0;
    }
}

static void apply_unary(Tensor *out, long offset_out, const Tensor *src, long offset_src, int dim,
                        Unary *func)
{
    if (dim == out->ndim - 1) {
        func(out->data + offset_out, src->data + offset_src, src->stride[dim], out->shape[dim]);
    }
    else {
        for (int i = 0; i < out->shape[dim]; i++) {
            apply_unary(out, offset_out + (i * out->stride[dim]), src,
                        offset_src + (i * src->stride[dim]), dim + 1, func);
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

Tensor *tensor_sin(const Tensor *src)
{
    return unary(src, unary_sin);
}

Tensor *tensor_cos(const Tensor *src)
{
    return unary(src, unary_cos);
}

Tensor *tensor_tan(const Tensor *src)
{
    return unary(src, unary_tan);
}

Tensor *tensor_log(const Tensor *src)
{
    return unary(src, unary_log);
}

Tensor *tensor_floor(const Tensor *src)
{
    return unary(src, unary_floor);
}

Tensor *tensor_ceil(const Tensor *src)
{
    return unary(src, unary_ceil);
}

Tensor *tensor_round(const Tensor *src)
{
    return unary(src, unary_round);
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

static void binary_add(float *out, const float *lhs, long stride_lhs, const float *rhs,
                       long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = lhs[i * stride_lhs] + rhs[i * stride_rhs];
    }
}

static void binary_sub(float *out, const float *lhs, long stride_lhs, const float *rhs,
                       long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = lhs[i * stride_lhs] - rhs[i * stride_rhs];
    }
}

static void binary_mul(float *out, const float *lhs, long stride_lhs, const float *rhs,
                       long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = lhs[i * stride_lhs] * rhs[i * stride_rhs];
    }
}

static void binary_div(float *out, const float *lhs, long stride_lhs, const float *rhs,
                       long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = lhs[i * stride_lhs] / rhs[i * stride_rhs];
    }
}

static void binary_mod(float *out, const float *lhs, long stride_lhs, const float *rhs,
                       long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        float val_rhs = rhs[i * stride_rhs];
        float mod = fmodf(lhs[i * stride_lhs], val_rhs);
        out[i] = (mod != 0 && (mod < 0) != (val_rhs < 0)) ? (mod + val_rhs) : mod;
    }
}

static void binary_pow(float *out, const float *lhs, long stride_lhs, const float *rhs,
                       long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = powf(lhs[i * stride_lhs], rhs[i * stride_rhs]);
    }
}

static void binary_eq(float *out, const float *lhs, long stride_lhs, const float *rhs,
                      long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = (lhs[i * stride_lhs] == rhs[i * stride_rhs]) ? 1 : 0;
    }
}

static void binary_ne(float *out, const float *lhs, long stride_lhs, const float *rhs,
                      long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = (lhs[i * stride_lhs] != rhs[i * stride_rhs]) ? 1 : 0;
    }
}

static void binary_lt(float *out, const float *lhs, long stride_lhs, const float *rhs,
                      long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = (lhs[i * stride_lhs] < rhs[i * stride_rhs]) ? 1 : 0;
    }
}

static void binary_le(float *out, const float *lhs, long stride_lhs, const float *rhs,
                      long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = (lhs[i * stride_lhs] <= rhs[i * stride_rhs]) ? 1 : 0;
    }
}

static void binary_gt(float *out, const float *lhs, long stride_lhs, const float *rhs,
                      long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = (lhs[i * stride_lhs] > rhs[i * stride_rhs]) ? 1 : 0;
    }
}

static void binary_ge(float *out, const float *lhs, long stride_lhs, const float *rhs,
                      long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = (lhs[i * stride_lhs] >= rhs[i * stride_rhs]) ? 1 : 0;
    }
}

static void binary_logical_and(float *out, const float *lhs, long stride_lhs, const float *rhs,
                               long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = ((lhs[i * stride_lhs] != 0) && (rhs[i * stride_rhs] != 0)) ? 1 : 0;
    }
}

static void binary_logical_or(float *out, const float *lhs, long stride_lhs, const float *rhs,
                              long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = ((lhs[i * stride_lhs] != 0) || (rhs[i * stride_rhs] != 0)) ? 1 : 0;
    }
}

static void binary_logical_xor(float *out, const float *lhs, long stride_lhs, const float *rhs,
                               long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = ((lhs[i * stride_lhs] != 0) != (rhs[i * stride_rhs] != 0)) ? 1 : 0;
    }
}

static void binary_minimum(float *out, const float *lhs, long stride_lhs, const float *rhs,
                           long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = fminf(lhs[i * stride_lhs], rhs[i * stride_rhs]);
    }
}

static void binary_maximum(float *out, const float *lhs, long stride_lhs, const float *rhs,
                           long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = fmaxf(lhs[i * stride_lhs], rhs[i * stride_rhs]);
    }
}

static int broadcast_binary(int *shape, const Tensor *lhs, const Tensor *rhs)
{
    int ndim = (lhs->ndim > rhs->ndim) ? lhs->ndim : rhs->ndim;
    for (int i = 0; i < ndim; i++) {
        int dim_lhs = i - (ndim - lhs->ndim);
        int dim_rhs = i - (ndim - rhs->ndim);
        int size_lhs = (dim_lhs >= 0) ? lhs->shape[dim_lhs] : 1;
        int size_rhs = (dim_rhs >= 0) ? rhs->shape[dim_rhs] : 1;
        int size = (size_lhs > size_rhs) ? size_lhs : size_rhs;
        assert((size_lhs == size || size_lhs == 1) && (size_rhs == size || size_rhs == 1));
        shape[i] = size;
    }
    return ndim;
}

static void apply_binary(Tensor *out, long offset_out, const Tensor *lhs, long offset_lhs,
                         const Tensor *rhs, long offset_rhs, int dim, Binary *func)
{
    if (dim == out->ndim - 1) {
        func(out->data + offset_out, lhs->data + offset_lhs, lhs->stride[dim],
             rhs->data + offset_rhs, rhs->stride[dim], out->shape[dim]);
    }
    else {
        for (int i = 0; i < out->shape[dim]; i++) {
            apply_binary(out, offset_out + (i * out->stride[dim]), lhs,
                         offset_lhs + (i * lhs->stride[dim]), rhs,
                         offset_rhs + (i * rhs->stride[dim]), dim + 1, func);
        }
    }
}

static Tensor *binary(const Tensor *lhs, const Tensor *rhs, Binary *func)
{
    assert(lhs && rhs && func);
    int shape[MAX_NDIM];
    int ndim = broadcast_binary(shape, lhs, rhs);
    lhs = tensor_expand(lhs, shape, ndim);
    rhs = tensor_expand(rhs, shape, ndim);
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

// ternary

typedef void Ternary(float *, const float *, long, const float *, long, const float *, long, int);

static void ternary_where(float *out, const float *lhs, long stride_lhs, const float *mid,
                          long stride_mid, const float *rhs, long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        float cond = lhs[i * stride_lhs];
        float if_true = mid[i * stride_mid];
        float if_false = rhs[i * stride_rhs];
        out[i] = (cond != 0) ? if_true : if_false;
    }
}

static void ternary_lerp(float *out, const float *lhs, long stride_lhs, const float *mid,
                         long stride_mid, const float *rhs, long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        float start = lhs[i * stride_lhs];
        float stop = mid[i * stride_mid];
        float weight = rhs[i * stride_rhs];
        out[i] = start + (weight * (stop - start));
    }
}

static void ternary_clamp(float *out, const float *lhs, long stride_lhs, const float *mid,
                          long stride_mid, const float *rhs, long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        float src = lhs[i * stride_lhs];
        float min = mid[i * stride_mid];
        float max = rhs[i * stride_rhs];
        out[i] = fmaxf(min, fminf(max, src));
    }
}

static int broadcast_ternary(int *shape, const Tensor *lhs, const Tensor *mid, const Tensor *rhs)
{
    int ndim = (lhs->ndim > rhs->ndim) ? lhs->ndim : rhs->ndim;
    if (mid->ndim > ndim) {
        ndim = mid->ndim;
    }
    for (int i = 0; i < ndim; i++) {
        int dim_lhs = i - (ndim - lhs->ndim);
        int dim_mid = i - (ndim - mid->ndim);
        int dim_rhs = i - (ndim - rhs->ndim);
        int size_lhs = (dim_lhs >= 0) ? lhs->shape[dim_lhs] : 1;
        int size_mid = (dim_mid >= 0) ? mid->shape[dim_mid] : 1;
        int size_rhs = (dim_rhs >= 0) ? rhs->shape[dim_rhs] : 1;
        int size = (size_lhs > size_rhs) ? size_lhs : size_rhs;
        if (size_mid > size) {
            size = size_mid;
        }
        assert((size_lhs == size || size_lhs == 1) && (size_mid == size || size_mid == 1) &&
               (size_rhs == size || size_rhs == 1));
        shape[i] = size;
    }
    return ndim;
}

static void apply_ternary(Tensor *out, long offset_out, const Tensor *lhs, long offset_lhs,
                          const Tensor *mid, long offset_mid, const Tensor *rhs, long offset_rhs,
                          int dim, Ternary *func)
{
    if (dim == out->ndim - 1) {
        func(out->data + offset_out, lhs->data + offset_lhs, lhs->stride[dim],
             mid->data + offset_mid, mid->stride[dim], rhs->data + offset_rhs, rhs->stride[dim],
             out->shape[dim]);
    }
    else {
        for (int i = 0; i < out->shape[dim]; i++) {
            apply_ternary(out, offset_out + (i * out->stride[dim]), lhs,
                          offset_lhs + (i * lhs->stride[dim]), mid,
                          offset_mid + (i * mid->stride[dim]), rhs,
                          offset_rhs + (i * rhs->stride[dim]), dim + 1, func);
        }
    }
}

static Tensor *ternary(const Tensor *lhs, const Tensor *mid, const Tensor *rhs, Ternary *func)
{
    assert(lhs && mid && rhs && func);
    int shape[MAX_NDIM];
    int ndim = broadcast_ternary(shape, lhs, mid, rhs);
    lhs = tensor_expand(lhs, shape, ndim);
    mid = tensor_expand(mid, shape, ndim);
    rhs = tensor_expand(rhs, shape, ndim);
    Tensor *out = tensor_empty(shape, ndim);
    if (ndim == 0) {
        func(out->data, lhs->data, 0, mid->data, 0, rhs->data, 0, 1);
    }
    else {
        apply_ternary(out, 0, lhs, 0, mid, 0, rhs, 0, 0, func);
    }
    return out;
}

Tensor *tensor_where(const Tensor *cond, const Tensor *if_true, const Tensor *if_false)
{
    return ternary(cond, if_true, if_false, ternary_where);
}

Tensor *tensor_lerp(const Tensor *start, const Tensor *stop, const Tensor *weight)
{
    return ternary(start, stop, weight, ternary_lerp);
}

Tensor *tensor_clamp(const Tensor *src, const Tensor *min, const Tensor *max)
{
    return ternary(src, min, max, ternary_clamp);
}

// reduction

typedef void Reduce(float *, const float *, long, long);

static void reduce_min(float *out, const float *src, long stride_src, long num)
{
    float acc = src[0];
    for (long i = 1; i < num; i++) {
        float val = src[i * stride_src];
        if (val < acc) {
            acc = val;
        }
    }
    *out = acc;
}

static void reduce_max(float *out, const float *src, long stride_src, long num)
{
    float acc = src[0];
    for (long i = 1; i < num; i++) {
        float val = src[i * stride_src];
        if (val > acc) {
            acc = val;
        }
    }
    *out = acc;
}

static void reduce_sum(float *out, const float *src, long stride_src, long num)
{
    float acc = 0;
    for (long i = 0; i < num; i++) {
        acc += src[i * stride_src];
    }
    *out = acc;
}

static void reduce_prod(float *out, const float *src, long stride_src, long num)
{
    float acc = 1;
    for (long i = 0; i < num; i++) {
        acc *= src[i * stride_src];
    }
    *out = acc;
}

static void reduce_all(float *out, const float *src, long stride_src, long num)
{
    *out = 1;
    for (long i = 0; i < num; i++) {
        if (src[i * stride_src] == 0) {
            *out = 0;
            return;
        }
    }
}

static void reduce_any(float *out, const float *src, long stride_src, long num)
{
    *out = 0;
    for (long i = 0; i < num; i++) {
        if (src[i * stride_src] != 0) {
            *out = 1;
            return;
        }
    }
}

static void apply_reduce(Tensor *out, long offset_out, const Tensor *src, long offset_src, int dim,
                         Reduce *func, int axis)
{
    if (dim == src->ndim) {
        func(out->data + offset_out, src->data + offset_src, src->stride[axis], src->shape[axis]);
        return;
    }
    if (dim == axis) {
        apply_reduce(out, offset_out, src, offset_src, dim + 1, func, axis);
    }
    else {
        int dim_out = (out->ndim == src->ndim || dim < axis) ? dim : (dim - 1);
        for (int i = 0; i < src->shape[dim]; i++) {
            apply_reduce(out, offset_out + (i * out->stride[dim_out]), src,
                         offset_src + (i * src->stride[dim]), dim + 1, func, axis);
        }
    }
}

static Tensor *reduce(const Tensor *src, int axis, int keepdim, Reduce *func)
{
    assert(src && func);
    if (axis == INT_MAX) {
        int shape[MAX_NDIM];
        for (int i = 0; i < src->ndim; i++) {
            shape[i] = 1;
        }
        int ndim = keepdim ? src->ndim : 0;
        Tensor *out = tensor_empty(shape, ndim);
        if (is_contiguous(src)) {
            func(out->data, src->data, 1, src->numel);
        }
        else {
            stack_save();
            src = tensor_contiguous(src);
            func(out->data, src->data, 1, src->numel);
            stack_restore();
        }
        return out;
    }
    axis = normalize_dim(axis, src->ndim);
    int shape[MAX_NDIM];
    int ndim = 0;
    for (int i = 0; i < src->ndim; i++) {
        if (i == axis) {
            if (keepdim) {
                shape[ndim++] = 1;
            }
        }
        else {
            shape[ndim++] = src->shape[i];
        }
    }
    Tensor *out = tensor_empty(shape, ndim);
    apply_reduce(out, 0, src, 0, 0, func, axis);
    return out;
}

Tensor *tensor_min(const Tensor *src, int axis, int keepdim)
{
    return reduce(src, axis, keepdim, reduce_min);
}

Tensor *tensor_max(const Tensor *src, int axis, int keepdim)
{
    return reduce(src, axis, keepdim, reduce_max);
}

Tensor *tensor_sum(const Tensor *src, int axis, int keepdim)
{
    return reduce(src, axis, keepdim, reduce_sum);
}

Tensor *tensor_prod(const Tensor *src, int axis, int keepdim)
{
    return reduce(src, axis, keepdim, reduce_prod);
}

Tensor *tensor_all(const Tensor *src, int axis, int keepdim)
{
    return reduce(src, axis, keepdim, reduce_all);
}

Tensor *tensor_any(const Tensor *src, int axis, int keepdim)
{
    return reduce(src, axis, keepdim, reduce_any);
}

Tensor *tensor_mean(const Tensor *src, int axis, int keepdim)
{
    Tensor *out = tensor_sum(src, axis, keepdim);
    long count = (axis == INT_MAX) ? src->numel : src->shape[normalize_dim(axis, src->ndim)];
    for (long i = 0; i < out->numel; i++) {
        out->data[i] /= (float)count;
    }
    return out;
}

Tensor *tensor_var(const Tensor *src, int axis, int keepdim)
{
    return tensor_mean(tensor_square(tensor_sub(src, tensor_mean(src, axis, 1))), axis, keepdim);
}

Tensor *tensor_std(const Tensor *src, int axis, int keepdim)
{
    return tensor_sqrt(tensor_var(src, axis, keepdim));
}

Tensor *tensor_norm(const Tensor *src, int axis, int keepdim)
{
    return tensor_sqrt(tensor_sum(tensor_square(src), axis, keepdim));
}

// argreduction

typedef void ArgReduce(long *, const float *, long, long);

static void argreduce_min(long *index, const float *src, long stride_src, long num)
{
    float acc = src[0];
    long arg = 0;
    for (long i = 1; i < num; i++) {
        float val = src[i * stride_src];
        if (val < acc) {
            acc = val;
            arg = i;
        }
    }
    *index = arg;
}

static void argreduce_max(long *index, const float *src, long stride_src, long num)
{
    float acc = src[0];
    long arg = 0;
    for (long i = 1; i < num; i++) {
        float val = src[i * stride_src];
        if (val > acc) {
            acc = val;
            arg = i;
        }
    }
    *index = arg;
}

static void apply_argreduce(const long *stride, long *index, long offset_index, const Tensor *src,
                            long offset_src, int dim, ArgReduce *func, int axis)
{
    if (dim == src->ndim) {
        func(index + offset_index, src->data + offset_src, src->stride[axis], src->shape[axis]);
        return;
    }
    if (dim == axis) {
        apply_argreduce(stride, index, offset_index, src, offset_src, dim + 1, func, axis);
    }
    else {
        int dim_out = (dim < axis) ? dim : (dim - 1);
        for (int i = 0; i < src->shape[dim]; i++) {
            apply_argreduce(stride, index, offset_index + (i * stride[dim_out]), src,
                            offset_src + (i * src->stride[dim]), dim + 1, func, axis);
        }
    }
}

static void argreduce(const Tensor *src, long *index, int axis, ArgReduce *func)
{
    assert(src && index && func);
    if (axis == INT_MAX) {
        if (is_contiguous(src)) {
            func(index, src->data, 1, src->numel);
        }
        else {
            stack_save();
            src = tensor_contiguous(src);
            func(index, src->data, 1, src->numel);
            stack_restore();
        }
        return;
    }
    axis = normalize_dim(axis, src->ndim);
    int shape[MAX_NDIM];
    int ndim = 0;
    for (int i = 0; i < src->ndim; i++) {
        if (i != axis) {
            shape[ndim++] = src->shape[i];
        }
    }
    long stride[MAX_NDIM];
    if (ndim > 0) {
        compute_stride(stride, shape, ndim);
    }
    apply_argreduce(stride, index, 0, src, 0, 0, func, axis);
}

void tensor_argmin(const Tensor *src, long *index, int axis)
{
    argreduce(src, index, axis, argreduce_min);
}

void tensor_argmax(const Tensor *src, long *index, int axis)
{
    argreduce(src, index, axis, argreduce_max);
}

// processing

Tensor *tensor_softmax(const Tensor *src, int axis)
{
    Tensor *exp = tensor_exp(tensor_sub(src, tensor_max(src, axis, 1)));
    return tensor_div(exp, tensor_sum(exp, axis, 1));
}

Tensor *tensor_log_softmax(const Tensor *src, int axis)
{
    Tensor *sub = tensor_sub(src, tensor_max(src, axis, 1));
    return tensor_sub(sub, tensor_log(tensor_sum(tensor_exp(sub), axis, 1)));
}

Tensor *tensor_cross_entropy(const Tensor *logit, const Tensor *target)
{
    assert(logit && target && logit->ndim == 2 && target->ndim == 1);
    assert(logit->shape[0] == target->shape[0]);
    int numel = logit->shape[0];
    Tensor *loss = tensor_empty((int[]){numel}, 1);
    stack_save();
    const Tensor *lsm = tensor_log_softmax(logit, 1);
    if (!is_contiguous(target)) {
        target = tensor_contiguous(target);
    }
    for (int i = 0; i < numel; i++) {
        int class = (int)target->data[i];
        assert(class >= 0 && class < logit->shape[1]);
        loss->data[i] = -lsm->data[((long)i * lsm->stride[0]) + class];
    }
    stack_restore();
    return tensor_mean(loss, INT_MAX, 0);
}

Tensor *tensor_dot(const Tensor *lhs, const Tensor *rhs)
{
    assert(lhs && rhs && lhs->ndim == 1 && rhs->ndim == 1);
    assert(lhs->shape[0] == rhs->shape[0]);
    float sum = 0;
    for (int i = 0; i < lhs->shape[0]; i++) {
        sum += lhs->data[i * lhs->stride[0]] * rhs->data[i * rhs->stride[0]];
    }
    return tensor_scalar(sum);
}

static const Tensor *matmul_prepare(const Tensor *src, long *stride, int *trans)
{
    int ndim = src->ndim;
    if (src->stride[ndim - 1] == 1) {
        *stride = src->stride[ndim - 2];
        *trans = 0;
    }
    else if (src->stride[ndim - 2] == 1) {
        *stride = src->stride[ndim - 1];
        *trans = 1;
    }
    else {
        src = tensor_contiguous(src);
        *stride = src->stride[src->ndim - 2];
        *trans = 0;
    }
    return src;
}

static void matmul(float *out, long stride_out, const float *lhs, long stride_lhs, const float *rhs,
                   long stride_rhs, int rows, int cols, int inner, int trans_lhs, int trans_rhs)
{
    for (int i = 0; i < rows; i++) {
        memset(out + (i * stride_out), 0, cols * sizeof(*out));
        for (int k = 0; k < inner; k++) {
            float val_lhs = !trans_lhs ? lhs[(i * stride_lhs) + k] : lhs[(k * stride_lhs) + i];
            for (int j = 0; j < cols; j++) {
                float val_rhs = !trans_rhs ? rhs[(k * stride_rhs) + j] : rhs[(j * stride_rhs) + k];
                out[(i * stride_out) + j] += val_lhs * val_rhs;
            }
        }
    }
}

Tensor *tensor_matmul(const Tensor *lhs, const Tensor *rhs)
{
    assert(lhs && rhs && lhs->ndim >= 2 && rhs->ndim >= 2);

    int rows = lhs->shape[lhs->ndim - 2];
    int cols = rhs->shape[rhs->ndim - 1];
    int inner = lhs->shape[lhs->ndim - 1];
    assert(inner == rhs->shape[rhs->ndim - 2]);

    int ndim = (lhs->ndim > rhs->ndim) ? lhs->ndim : rhs->ndim;
    int shape[MAX_NDIM];
    long batches = 1;
    for (int i = 0; i < ndim - 2; i++) {
        int dim_lhs = i - (ndim - lhs->ndim);
        int dim_rhs = i - (ndim - rhs->ndim);
        int size_lhs = (dim_lhs >= 0) ? lhs->shape[dim_lhs] : 1;
        int size_rhs = (dim_rhs >= 0) ? rhs->shape[dim_rhs] : 1;
        assert(size_lhs == size_rhs || size_lhs == 1 || size_rhs == 1);
        shape[i] = (size_lhs > size_rhs) ? size_lhs : size_rhs;
        batches *= shape[i];
    }
    shape[ndim - 2] = rows;
    shape[ndim - 1] = cols;

    Tensor *out = tensor_empty(shape, ndim);
    long stride_out = out->stride[out->ndim - 2];
    long stride_batch = (out->ndim > 2) ? out->stride[out->ndim - 3] : 0;

    stack_save();

    int trans_lhs;
    long stride_lhs;
    lhs = matmul_prepare(lhs, &stride_lhs, &trans_lhs);

    int trans_rhs;
    long stride_rhs;
    rhs = matmul_prepare(rhs, &stride_rhs, &trans_rhs);

    for (long i = 0; i < batches; i++) {
        long offset_lhs = 0;
        long offset_rhs = 0;
        long remaining = i;
        for (int j = ndim - 3; j >= 0; j--) {
            long idx_batch = remaining % shape[j];
            int dim_lhs = j - (ndim - lhs->ndim);
            if (dim_lhs >= 0) {
                long idx_lhs = (lhs->shape[dim_lhs] == 1) ? 0 : idx_batch;
                offset_lhs += idx_lhs * lhs->stride[dim_lhs];
            }
            int dim_rhs = j - (ndim - rhs->ndim);
            if (dim_rhs >= 0) {
                long idx_rhs = (rhs->shape[dim_rhs] == 1) ? 0 : idx_batch;
                offset_rhs += idx_rhs * rhs->stride[dim_rhs];
            }
            remaining /= shape[j];
        }
        float *out_data = out->data + (i * stride_batch);
        const float *lhs_data = lhs->data + offset_lhs;
        const float *rhs_data = rhs->data + offset_rhs;
        matmul(out_data, stride_out, lhs_data, stride_lhs, rhs_data, stride_rhs, rows, cols, inner,
               trans_lhs, trans_rhs);
    }

    stack_restore();
    return out;
}

// autograd

Tensor *tensor_requires_grad(Tensor *self)
{
    assert(self);
    self->requires_grad = 1;
    return self;
}

Tensor *tensor_grad(const Tensor *self)
{
    assert(self);
    return self->grad ? tensor_wrap(self->shape, self->ndim, self->grad) : 0;
}

enum { MAX_TOPO = 1024 };

static void build_topo(Tensor *self, Tensor **topo, int *count)
{
    if (!self) {
        return;
    }
    for (int i = 0; i < *count; i++) {
        if (topo[i] == self) {
            return;
        }
    }
    if (self->ctx) {
        for (int i = 0; i < self->ctx->num_inputs; i++) {
            build_topo(self->ctx->input[i], topo, count);
        }
    }
    assert(*count < MAX_TOPO);
    topo[(*count)++] = self;
}

void tensor_backward(Tensor *self, const Tensor *grad)
{
    assert(self);

    if (!self->grad) {
        self->grad = stack_calloc(self->numel, sizeof(*self->grad));
    }

    if (grad) {
        assert(grad->numel == self->numel);
        for (long i = 0; i < self->numel; i++) {
            self->grad[i] += grad->data[i];
        }
    }
    else {
        for (long i = 0; i < self->numel; i++) {
            self->grad[i] += 1;
        }
    }

    Tensor *topo[MAX_TOPO];
    int count = 0;
    build_topo(self, topo, &count);

    for (int i = count - 1; i >= 0; i--) {
        if (topo[i]->ctx && topo[i]->ctx->backward) {
            topo[i]->ctx->backward(topo[i]);
        }
    }
}

void tensor_zero_grad(Tensor *self)
{
    assert(self);
    if (self->grad) {
        memset(self->grad, 0, self->numel * sizeof(*self->grad));
    }
}

// i/o

static void print_data(const Tensor *self, long offset, int dim)
{
    printf("[");
    if (dim == self->ndim - 1) {
        for (int i = 0; i < self->shape[dim]; i++) {
            printf("%.8g", self->data[offset + (i * self->stride[dim])]);
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
            print_data(self, offset + (i * self->stride[dim]), dim + 1);
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
        printf("%.8g", self->data[0]);
    }
    else {
        print_data(self, 0, 0);
    }
    printf("\n\n");
}

void tensor_save(const Tensor *self, const char *fname)
{
    assert(self && fname);

    stack_save();

    if (!is_contiguous(self)) {
        self = tensor_contiguous(self);
    }

    // build shape string: "()" or "(n,)" or "(a, b, c)"
    char shape[128];
    int slen = sprintf(shape, "(");
    for (int i = 0; i < self->ndim; i++) {
        slen += sprintf(shape + slen, "%d", self->shape[i]);
        if (i < self->ndim - 1) {
            slen += sprintf(shape + slen, ", ");
        }
    }
    if (self->ndim == 1) {
        slen += sprintf(shape + slen, ",");
    }
    slen += sprintf(shape + slen, ")");

    // build header dict
    char header[256];
    int hlen = sprintf(header, "{'descr': '<f4', 'fortran_order': False, 'shape': %s, }", shape);

    // pad so that 10 + hlen + pad + 1 (\n) is a multiple of 64
    int total = 10 + hlen + 1;
    int padding = (64 - (total % 64)) % 64;
    uint16_t header_len = (uint16_t)(hlen + padding + 1);

    FILE *file = fopen(fname, "wb");
    assert(file);
    fwrite((uint8_t[]){0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00}, 1, 8, file);  // magic + version
    fwrite(&header_len, sizeof(header_len), 1, file);
    fwrite(header, 1, hlen, file);
    for (int i = 0; i < padding; i++) {
        fputc(' ', file);
    }
    fputc('\n', file);
    fwrite(self->data, sizeof(float), self->numel, file);
    fclose(file);

    stack_restore();
}

Tensor *tensor_load(const char *fname)
{
    assert(fname);

    FILE *file = fopen(fname, "rb");
    assert(file);

    // magic + version
    uint8_t magic[8];
    fread(magic, 1, sizeof(magic), file);
    assert(magic[0] == 0x93 && magic[1] == 'N' && magic[2] == 'U' && magic[3] == 'M' &&
           magic[4] == 'P' && magic[5] == 'Y');

    // header length (2 bytes for v1, 4 bytes for v2)
    int header_len;
    if (magic[6] == 1) {
        uint16_t len;
        fread(&len, sizeof(len), 1, file);
        header_len = len;
    }
    else {
        uint32_t len;
        fread(&len, sizeof(len), 1, file);
        assert(len <= INT_MAX);
        header_len = (int)len;
    }

    // header dict
    char header[1024];
    assert(header_len < (int)sizeof(header));
    fread(header, 1, header_len, file);
    header[header_len] = 0;

    // parse shape tuple
    char *pos = strstr(header, "'shape'");
    assert(pos);
    pos = strchr(pos, '(');
    assert(pos);
    pos += 1;
    int shape[MAX_NDIM];
    int ndim = 0;
    while (*pos != ')' && *pos != '\0') {
        while (*pos == ' ' || *pos == ',') {
            pos++;
        }
        if (*pos >= '0' && *pos <= '9') {
            assert(ndim < MAX_NDIM);
            long size = strtol(pos, &pos, 10);
            assert(size <= INT_MAX);
            shape[ndim++] = (int)size;
        }
    }
    assert(*pos == ')');

    Tensor *out = tensor_empty(shape, ndim);
    fread(out->data, sizeof(float), out->numel, file);

    fclose(file);
    return out;
}
