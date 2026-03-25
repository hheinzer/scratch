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
static Stack *g_save[MAX_SAVE];
static int g_index = 0;

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

static void stack_save(void)
{
    assert(g_index < MAX_SAVE);
    g_save[g_index++] = g_head;
}

static void stack_restore(void)
{
    assert(g_index > 0);
    Stack *save = g_save[--g_index];
    while (g_head != save) {
        free(stack_pop());
    }
}

// context

static int g_grad_enabled = 1;

void tensor_frame_begin(void)
{
    stack_save();
}

void tensor_frame_end(void)
{
    stack_restore();
}

void tensor_no_grad_begin(void)
{
    g_grad_enabled = 0;
}

void tensor_no_grad_end(void)
{
    g_grad_enabled = 1;
}

// tensor

enum { MAX_NDIM = 8 };

typedef union {
    int axis;
    int keepdim;
} Saved;

typedef struct {
    int num_inputs;
    Tensor *input[3];
    Saved saved[3];
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

Tensor *tensor_requires_grad(Tensor *self)
{
    assert(self);
    self->requires_grad = 1;
    self->grad = stack_calloc(self->numel, sizeof(*self->grad));
    return self;
}

Tensor *tensor_grad(const Tensor *self)
{
    assert(self);
    return self->grad ? tensor_wrap(self->shape, self->ndim, self->grad) : 0;
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
    Tensor *out = tensor_empty(shape, ndim);
    for (long i = 0; i < out->numel; i++) {
        out->data[i] = value;
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
    const Tensor *uns[num];
    for (int i = 0; i < num; i++) {
        uns[i] = tensor_unsqueeze(src[i], dim);
    }
    return tensor_cat(uns, num, dim);
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

static void neg_kernel(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = -src[i * stride_src];
    }
}

static void abs_kernel(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = fabsf(src[i * stride_src]);
    }
}

static void sign_kernel(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        float val = src[i * stride_src];
        out[i] = (float)((val > 0) - (val < 0));
    }
}

static void square_kernel(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        float val = src[i * stride_src];
        out[i] = val * val;
    }
}

static void sqrt_kernel(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = sqrtf(src[i * stride_src]);
    }
}

static void rsqrt_kernel(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = 1 / sqrtf(src[i * stride_src]);
    }
}

static void exp_kernel(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = expf(src[i * stride_src]);
    }
}

static void log_kernel(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = logf(src[i * stride_src]);
    }
}

static void relu_kernel(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        float val = src[i * stride_src];
        out[i] = (val > 0) ? val : 0;
    }
}

static void sigmoid_kernel(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = 1 / (1 + expf(-src[i * stride_src]));
    }
}

static void tanh_kernel(float *out, const float *src, long stride_src, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = tanhf(src[i * stride_src]);
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

static void accumulate_grad(Tensor *self, const Tensor *grad)
{
    if (!self->grad) {
        self->grad = stack_calloc(self->numel, sizeof(*self->grad));
    }
    stack_save();
    int offset = grad->ndim - self->ndim;
    for (int i = 0; i < offset; i++) {
        grad = tensor_sum(grad, 0, 0);
    }
    for (int i = 0; i < self->ndim; i++) {
        if (self->shape[i] == 1) {
            grad = tensor_sum(grad, i, 1);
        }
    }
    for (long i = 0; i < self->numel; i++) {
        self->grad[i] += grad->data[i];
    }
    stack_restore();
}

static void neg_backward(Tensor *out)
{
    // out = -src  =>  d/d(src) = -1
    Tensor *src = out->ctx->input[0];
    Tensor *grad = tensor_grad(out);
    if (src->requires_grad) {
        accumulate_grad(src, tensor_neg(grad));
    }
}

Tensor *tensor_neg(const Tensor *src)
{
    Tensor *out = unary(src, neg_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = neg_backward;
    }
    return out;
}

static void abs_backward(Tensor *out)
{
    // out = |src|  =>  d/d(src) = sign(src)
    Tensor *src = out->ctx->input[0];
    Tensor *grad = tensor_grad(out);
    if (src->requires_grad) {
        accumulate_grad(src, tensor_mul(grad, tensor_sign(src)));
    }
}

Tensor *tensor_abs(const Tensor *src)
{
    Tensor *out = unary(src, abs_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = abs_backward;
    }
    return out;
}

Tensor *tensor_sign(const Tensor *src)
{
    return unary(src, sign_kernel);
}

static void square_backward(Tensor *out)
{
    // out = src^2  =>  d/d(src) = 2*src
    Tensor *src = out->ctx->input[0];
    Tensor *grad = tensor_grad(out);
    if (src->requires_grad) {
        accumulate_grad(src, tensor_mul(grad, tensor_mul(tensor_scalar(2), src)));
    }
}

Tensor *tensor_square(const Tensor *src)
{
    Tensor *out = unary(src, square_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = square_backward;
    }
    return out;
}

static void sqrt_backward(Tensor *out)
{
    // out = sqrt(src)  =>  d/d(src) = 1/(2*out)
    Tensor *src = out->ctx->input[0];
    Tensor *grad = tensor_grad(out);
    if (src->requires_grad) {
        accumulate_grad(src, tensor_mul(grad, tensor_div(tensor_scalar(0.5F), out)));
    }
}

Tensor *tensor_sqrt(const Tensor *src)
{
    Tensor *out = unary(src, sqrt_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = sqrt_backward;
    }
    return out;
}

static void rsqrt_backward(Tensor *out)
{
    // out = 1/sqrt(src)  =>  d/d(src) = -out^3/2
    Tensor *src = out->ctx->input[0];
    Tensor *grad = tensor_grad(out);
    if (src->requires_grad) {
        accumulate_grad(src, tensor_mul(tensor_scalar(-0.5F),
                                        tensor_mul(grad, tensor_mul(out, tensor_square(out)))));
    }
}

Tensor *tensor_rsqrt(const Tensor *src)
{
    Tensor *out = unary(src, rsqrt_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = rsqrt_backward;
    }
    return out;
}

static void exp_backward(Tensor *out)
{
    // out = exp(src)  =>  d/d(src) = out
    Tensor *src = out->ctx->input[0];
    Tensor *grad = tensor_grad(out);
    if (src->requires_grad) {
        accumulate_grad(src, tensor_mul(grad, out));
    }
}

Tensor *tensor_exp(const Tensor *src)
{
    Tensor *out = unary(src, exp_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = exp_backward;
    }
    return out;
}

static void log_backward(Tensor *out)
{
    // out = log(src)  =>  d/d(src) = 1/src
    Tensor *src = out->ctx->input[0];
    Tensor *grad = tensor_grad(out);
    if (src->requires_grad) {
        accumulate_grad(src, tensor_div(grad, src));
    }
}

Tensor *tensor_log(const Tensor *src)
{
    Tensor *out = unary(src, log_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = log_backward;
    }
    return out;
}

static void relu_backward(Tensor *out)
{
    // out = relu(src)  =>  d/d(src) = sign(out)  (1 if src > 0, else 0)
    Tensor *src = out->ctx->input[0];
    Tensor *grad = tensor_grad(out);
    if (src->requires_grad) {
        accumulate_grad(src, tensor_mul(grad, tensor_sign(out)));
    }
}

Tensor *tensor_relu(const Tensor *src)
{
    Tensor *out = unary(src, relu_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = relu_backward;
    }
    return out;
}

static void sigmoid_backward(Tensor *out)
{
    // out = sigmoid(src)  =>  d/d(src) = out*(1-out)
    Tensor *src = out->ctx->input[0];
    Tensor *grad = tensor_grad(out);
    if (src->requires_grad) {
        accumulate_grad(src, tensor_mul(grad, tensor_mul(out, tensor_sub(tensor_scalar(1), out))));
    }
}

Tensor *tensor_sigmoid(const Tensor *src)
{
    Tensor *out = unary(src, sigmoid_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = sigmoid_backward;
    }
    return out;
}

static void tanh_backward(Tensor *out)
{
    // out = tanh(src)  =>  d/d(src) = 1-out^2
    Tensor *src = out->ctx->input[0];
    Tensor *grad = tensor_grad(out);
    if (src->requires_grad) {
        accumulate_grad(src, tensor_mul(grad, tensor_sub(tensor_scalar(1), tensor_square(out))));
    }
}

Tensor *tensor_tanh(const Tensor *src)
{
    Tensor *out = unary(src, tanh_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = tanh_backward;
    }
    return out;
}

// binary

typedef void Binary(float *, const float *, long, const float *, long, int);

static void add_kernel(float *out, const float *lhs, long stride_lhs, const float *rhs,
                       long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = lhs[i * stride_lhs] + rhs[i * stride_rhs];
    }
}

static void sub_kernel(float *out, const float *lhs, long stride_lhs, const float *rhs,
                       long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = lhs[i * stride_lhs] - rhs[i * stride_rhs];
    }
}

static void mul_kernel(float *out, const float *lhs, long stride_lhs, const float *rhs,
                       long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = lhs[i * stride_lhs] * rhs[i * stride_rhs];
    }
}

static void div_kernel(float *out, const float *lhs, long stride_lhs, const float *rhs,
                       long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = lhs[i * stride_lhs] / rhs[i * stride_rhs];
    }
}

static void pow_kernel(float *out, const float *lhs, long stride_lhs, const float *rhs,
                       long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        out[i] = powf(lhs[i * stride_lhs], rhs[i * stride_rhs]);
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

static void add_backward(Tensor *out)
{
    // out = lhs + rhs  =>  d/d(lhs) = 1,  d/d(rhs) = 1
    Tensor *lhs = out->ctx->input[0];
    Tensor *rhs = out->ctx->input[1];
    Tensor *grad = tensor_grad(out);
    if (lhs->requires_grad) {
        accumulate_grad(lhs, grad);
    }
    if (rhs->requires_grad) {
        accumulate_grad(rhs, grad);
    }
}

Tensor *tensor_add(const Tensor *lhs, const Tensor *rhs)
{
    Tensor *out = binary(lhs, rhs, add_kernel);
    if (g_grad_enabled && (lhs->requires_grad || rhs->requires_grad)) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 2;
        out->ctx->input[0] = (Tensor *)lhs;
        out->ctx->input[1] = (Tensor *)rhs;
        out->ctx->backward = add_backward;
    }
    return out;
}

static void sub_backward(Tensor *out)
{
    // out = lhs - rhs  =>  d/d(lhs) = 1,  d/d(rhs) = -1
    Tensor *lhs = out->ctx->input[0];
    Tensor *rhs = out->ctx->input[1];
    Tensor *grad = tensor_grad(out);
    if (lhs->requires_grad) {
        accumulate_grad(lhs, grad);
    }
    if (rhs->requires_grad) {
        accumulate_grad(rhs, tensor_neg(grad));
    }
}

Tensor *tensor_sub(const Tensor *lhs, const Tensor *rhs)
{
    Tensor *out = binary(lhs, rhs, sub_kernel);
    if (g_grad_enabled && (lhs->requires_grad || rhs->requires_grad)) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 2;
        out->ctx->input[0] = (Tensor *)lhs;
        out->ctx->input[1] = (Tensor *)rhs;
        out->ctx->backward = sub_backward;
    }
    return out;
}

static void mul_backward(Tensor *out)
{
    // out = lhs * rhs  =>  d/d(lhs) = rhs,  d/d(rhs) = lhs
    Tensor *lhs = out->ctx->input[0];
    Tensor *rhs = out->ctx->input[1];
    Tensor *grad = tensor_grad(out);
    if (lhs->requires_grad) {
        accumulate_grad(lhs, tensor_mul(grad, rhs));
    }
    if (rhs->requires_grad) {
        accumulate_grad(rhs, tensor_mul(lhs, grad));
    }
}

Tensor *tensor_mul(const Tensor *lhs, const Tensor *rhs)
{
    Tensor *out = binary(lhs, rhs, mul_kernel);
    if (g_grad_enabled && (lhs->requires_grad || rhs->requires_grad)) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 2;
        out->ctx->input[0] = (Tensor *)lhs;
        out->ctx->input[1] = (Tensor *)rhs;
        out->ctx->backward = mul_backward;
    }
    return out;
}

static void div_backward(Tensor *out)
{
    // out = lhs / rhs  =>  d/d(lhs) = 1/rhs,  d/d(rhs) = -out/rhs
    Tensor *lhs = out->ctx->input[0];
    Tensor *rhs = out->ctx->input[1];
    Tensor *grad = tensor_grad(out);
    if (lhs->requires_grad) {
        accumulate_grad(lhs, tensor_div(grad, rhs));
    }
    if (rhs->requires_grad) {
        accumulate_grad(rhs, tensor_neg(tensor_mul(grad, tensor_div(out, rhs))));
    }
}

Tensor *tensor_div(const Tensor *lhs, const Tensor *rhs)
{
    Tensor *out = binary(lhs, rhs, div_kernel);
    if (g_grad_enabled && (lhs->requires_grad || rhs->requires_grad)) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 2;
        out->ctx->input[0] = (Tensor *)lhs;
        out->ctx->input[1] = (Tensor *)rhs;
        out->ctx->backward = div_backward;
    }
    return out;
}

static void pow_backward(Tensor *out)
{
    // out = lhs^rhs  =>  d/d(lhs) = rhs*lhs^(rhs-1),  d/d(rhs) = out*log(lhs)
    Tensor *lhs = out->ctx->input[0];
    Tensor *rhs = out->ctx->input[1];
    Tensor *grad = tensor_grad(out);
    if (lhs->requires_grad) {
        accumulate_grad(
            lhs,
            tensor_mul(grad, tensor_mul(rhs, tensor_pow(lhs, tensor_sub(rhs, tensor_scalar(1))))));
    }
    if (rhs->requires_grad) {
        accumulate_grad(rhs, tensor_mul(grad, tensor_mul(out, tensor_log(lhs))));
    }
}

Tensor *tensor_pow(const Tensor *lhs, const Tensor *rhs)
{
    Tensor *out = binary(lhs, rhs, pow_kernel);
    if (g_grad_enabled && (lhs->requires_grad || rhs->requires_grad)) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 2;
        out->ctx->input[0] = (Tensor *)lhs;
        out->ctx->input[1] = (Tensor *)rhs;
        out->ctx->backward = pow_backward;
    }
    return out;
}

// ternary

typedef void Ternary(float *, const float *, long, const float *, long, const float *, long, int);

static void where_kernel(float *out, const float *lhs, long stride_lhs, const float *mid,
                         long stride_mid, const float *rhs, long stride_rhs, int num)
{
    for (int i = 0; i < num; i++) {
        float cond = lhs[i * stride_lhs];
        float if_true = mid[i * stride_mid];
        float if_false = rhs[i * stride_rhs];
        out[i] = (cond != 0) ? if_true : if_false;
    }
}

static void clamp_kernel(float *out, const float *lhs, long stride_lhs, const float *mid,
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

static void where_backward(Tensor *out)
{
    // out = where(cond, if_true, if_false)
    // d/d(if_true)  = grad if cond != 0, else 0
    // d/d(if_false) = grad if cond == 0, else 0
    Tensor *cond = out->ctx->input[0];
    Tensor *if_true = out->ctx->input[1];
    Tensor *if_false = out->ctx->input[2];
    Tensor *grad = tensor_grad(out);
    if (if_true->requires_grad) {
        accumulate_grad(if_true, tensor_where(cond, grad, tensor_scalar(0)));
    }
    if (if_false->requires_grad) {
        accumulate_grad(if_false, tensor_where(cond, tensor_scalar(0), grad));
    }
}

Tensor *tensor_where(const Tensor *cond, const Tensor *if_true, const Tensor *if_false)
{
    Tensor *out = ternary(cond, if_true, if_false, where_kernel);
    if (g_grad_enabled && (if_true->requires_grad || if_false->requires_grad)) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 3;
        out->ctx->input[0] = (Tensor *)cond;
        out->ctx->input[1] = (Tensor *)if_true;
        out->ctx->input[2] = (Tensor *)if_false;
        out->ctx->backward = where_backward;
    }
    return out;
}

static void clamp_backward(Tensor *out)
{
    // out = clamp(src, min, max),  sign = sign(out - src)
    // d/d(src) = 1 - |sign|   (sign ==  0: not clamped)
    // d/d(min) = relu(sign)   (sign ==  1: clamped to min)
    // d/d(max) = relu(-sign)  (sign == -1: clamped to max)
    Tensor *src = out->ctx->input[0];
    Tensor *min = out->ctx->input[1];
    Tensor *max = out->ctx->input[2];
    Tensor *grad = tensor_grad(out);
    Tensor *sign = tensor_sign(tensor_sub(out, src));
    if (src->requires_grad) {
        accumulate_grad(src, tensor_mul(grad, tensor_sub(tensor_scalar(1), tensor_abs(sign))));
    }
    if (min->requires_grad) {
        accumulate_grad(min, tensor_mul(grad, tensor_relu(sign)));
    }
    if (max->requires_grad) {
        accumulate_grad(max, tensor_mul(grad, tensor_relu(tensor_neg(sign))));
    }
}

Tensor *tensor_clamp(const Tensor *src, const Tensor *min, const Tensor *max)
{
    Tensor *out = ternary(src, min, max, clamp_kernel);
    if (g_grad_enabled && (src->requires_grad || min->requires_grad || max->requires_grad)) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 3;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->input[1] = (Tensor *)min;
        out->ctx->input[2] = (Tensor *)max;
        out->ctx->backward = clamp_backward;
    }
    return out;
}

// reduction

typedef void Reduce(float *, const float *, long, long);

static void min_kernel(float *out, const float *src, long stride_src, long num)
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

static void max_kernel(float *out, const float *src, long stride_src, long num)
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

static void sum_kernel(float *out, const float *src, long stride_src, long num)
{
    float acc = 0;
    for (long i = 0; i < num; i++) {
        acc += src[i * stride_src];
    }
    *out = acc;
}

static void var_kernel(float *out, const float *src, long stride_src, long num)
{
    float mean = 0;
    float acc = 0;
    for (long i = 0; i < num; i++) {
        float val = src[i * stride_src];
        float delta = val - mean;
        mean += delta / (float)(i + 1);
        acc += delta * (val - mean);
    }
    *out = acc / (float)num;
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

static void min_backward(Tensor *out)
{
    // out = min(src, axis)  =>  d/d(src) = grad at argmin positions, 0 elsewhere
    Tensor *src = out->ctx->input[0];
    if (src->requires_grad) {
        int axis = out->ctx->saved[0].axis;
        int keepdim = out->ctx->saved[1].keepdim;
        Tensor *grad = tensor_grad(out);
        if (!keepdim && axis != INT_MAX) {
            out = tensor_unsqueeze(out, axis);
            grad = tensor_unsqueeze(grad, axis);
        }
        Tensor *mask = tensor_where(tensor_sub(src, out), tensor_scalar(0), tensor_scalar(1));
        accumulate_grad(src, tensor_mul(grad, tensor_div(mask, tensor_sum(mask, axis, 1))));
    }
}

Tensor *tensor_min(const Tensor *src, int axis, int keepdim)
{
    Tensor *out = reduce(src, axis, keepdim, min_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->saved[0].axis = axis;
        out->ctx->saved[1].keepdim = keepdim;
        out->ctx->backward = min_backward;
    }
    return out;
}

static void max_backward(Tensor *out)
{
    // out = max(src, axis)  =>  d/d(src) = grad at argmax positions, 0 elsewhere
    Tensor *src = out->ctx->input[0];
    if (src->requires_grad) {
        int axis = out->ctx->saved[0].axis;
        int keepdim = out->ctx->saved[1].keepdim;
        Tensor *grad = tensor_grad(out);
        if (!keepdim && axis != INT_MAX) {
            out = tensor_unsqueeze(out, axis);
            grad = tensor_unsqueeze(grad, axis);
        }
        Tensor *mask = tensor_where(tensor_sub(src, out), tensor_scalar(0), tensor_scalar(1));
        accumulate_grad(src, tensor_mul(grad, tensor_div(mask, tensor_sum(mask, axis, 1))));
    }
}

Tensor *tensor_max(const Tensor *src, int axis, int keepdim)
{
    Tensor *out = reduce(src, axis, keepdim, max_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->saved[0].axis = axis;
        out->ctx->saved[1].keepdim = keepdim;
        out->ctx->backward = max_backward;
    }
    return out;
}

static void expand_grad(Tensor *src, Tensor *grad, int axis, int keepdim)
{
    if (!keepdim && axis != INT_MAX) {
        grad = tensor_unsqueeze(grad, axis);
    }
    grad = tensor_contiguous(tensor_expand(grad, src->shape, src->ndim));
    if (!src->grad) {
        src->grad = stack_calloc(src->numel, sizeof(*src->grad));
    }
    for (long i = 0; i < src->numel; i++) {
        src->grad[i] += grad->data[i];
    }
}

static void sum_backward(Tensor *out)
{
    // out = sum(src, axis)  =>  d/d(src) = 1 broadcast to src->shape
    Tensor *src = out->ctx->input[0];
    if (src->requires_grad) {
        int axis = out->ctx->saved[0].axis;
        int keepdim = out->ctx->saved[1].keepdim;
        expand_grad(src, tensor_grad(out), axis, keepdim);
    }
}

Tensor *tensor_sum(const Tensor *src, int axis, int keepdim)
{
    Tensor *out = reduce(src, axis, keepdim, sum_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->saved[0].axis = axis;
        out->ctx->saved[1].keepdim = keepdim;
        out->ctx->backward = sum_backward;
    }
    return out;
}

static void mean_backward(Tensor *out)
{
    // out = mean(src, axis)  =>  d/d(src) = 1/n broadcast to src->shape
    Tensor *src = out->ctx->input[0];
    if (src->requires_grad) {
        int axis = out->ctx->saved[0].axis;
        int keepdim = out->ctx->saved[1].keepdim;
        long count = (axis == INT_MAX) ? src->numel : src->shape[normalize_dim(axis, src->ndim)];
        Tensor *grad = tensor_mul(tensor_grad(out), tensor_scalar(1 / (float)count));
        expand_grad(src, grad, axis, keepdim);
    }
}

Tensor *tensor_mean(const Tensor *src, int axis, int keepdim)
{
    Tensor *out = reduce(src, axis, keepdim, sum_kernel);
    long count = (axis == INT_MAX) ? src->numel : src->shape[normalize_dim(axis, src->ndim)];
    for (long i = 0; i < out->numel; i++) {
        out->data[i] /= (float)count;
    }
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->saved[0].axis = axis;
        out->ctx->saved[1].keepdim = keepdim;
        out->ctx->backward = mean_backward;
    }
    return out;
}

static void var_backward(Tensor *out)
{
    // out = var(src, axis)  =>  d/d(src) = 2*(src - mean(src, axis)) / n
    Tensor *src = out->ctx->input[0];
    if (src->requires_grad) {
        int axis = out->ctx->saved[0].axis;
        long count = (axis == INT_MAX) ? src->numel : src->shape[normalize_dim(axis, src->ndim)];
        Tensor *diff = tensor_sub(src, tensor_mean(src, axis, 1));
        Tensor *grad =
            tensor_mul(tensor_mul(tensor_grad(out), tensor_scalar(2 / (float)count)), diff);
        accumulate_grad(src, grad);
    }
}

Tensor *tensor_var(const Tensor *src, int axis, int keepdim)
{
    Tensor *out = reduce(src, axis, keepdim, var_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->saved[0].axis = axis;
        out->ctx->backward = var_backward;
    }
    return out;
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

static void argmin_kernel(long *index, const float *src, long stride_src, long num)
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

static void argmax_kernel(long *index, const float *src, long stride_src, long num)
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
    argreduce(src, index, axis, argmin_kernel);
}

void tensor_argmax(const Tensor *src, long *index, int axis)
{
    argreduce(src, index, axis, argmax_kernel);
}

// processing

static void matmul_backward(Tensor *out)
{
    // out = lhs @ rhs  =>  d/d(lhs) = grad @ rhs.T,  d/d(rhs) = lhs.T @ grad
    Tensor *lhs = out->ctx->input[0];
    Tensor *rhs = out->ctx->input[1];
    Tensor *grad = tensor_grad(out);
    if (lhs->requires_grad) {
        accumulate_grad(lhs, tensor_matmul(grad, tensor_transpose(rhs, -2, -1)));
    }
    if (rhs->requires_grad) {
        accumulate_grad(rhs, tensor_matmul(tensor_transpose(lhs, -2, -1), grad));
    }
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

    if (g_grad_enabled && (lhs->requires_grad || rhs->requires_grad)) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->num_inputs = 2;
        out->ctx->input[0] = (Tensor *)lhs;
        out->ctx->input[1] = (Tensor *)rhs;
        out->ctx->backward = matmul_backward;
    }

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

    if (grad) {
        assert(grad->numel == self->numel);
        self->grad = grad->data;
    }
    else {
        if (!self->grad) {
            self->grad = stack_malloc(self->numel, sizeof(*self->grad));
        }
        for (long i = 0; i < self->numel; i++) {
            self->grad[i] = 1;
        }
    }

    Tensor *topo[MAX_TOPO];
    int count = 0;
    build_topo(self, topo, &count);

    int grad_enabled = g_grad_enabled;
    g_grad_enabled = 0;

    for (int i = count - 1; i >= 0; i--) {
        if (topo[i]->ctx && topo[i]->ctx->backward) {
            topo[i]->ctx->backward(topo[i]);
        }
    }

    g_grad_enabled = grad_enabled;
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

static const char *backward_name(void (*func)(Tensor *))
{
    static const struct {
        void (*func)(Tensor *);
        const char *name;
    } map[] = {
        {neg_backward, "neg"},     {abs_backward, "abs"},     {square_backward, "square"},
        {sqrt_backward, "sqrt"},   {rsqrt_backward, "rsqrt"}, {exp_backward, "exp"},
        {log_backward, "log"},     {relu_backward, "relu"},   {sigmoid_backward, "sigmoid"},
        {tanh_backward, "tanh"},   {add_backward, "add"},     {sub_backward, "sub"},
        {mul_backward, "mul"},     {div_backward, "div"},     {pow_backward, "pow"},
        {where_backward, "where"}, {clamp_backward, "clamp"}, {sum_backward, "sum"},
        {mean_backward, "mean"},   {var_backward, "var"},     {matmul_backward, "matmul"},
        {min_backward, "min"},     {max_backward, "max"},
    };
    for (int i = 0; i < (int)(sizeof(map) / sizeof(*map)); i++) {
        if (map[i].func == func) {
            return map[i].name;
        }
    }
    return "?";
}

static void print_backward(const Tensor *self, int depth)
{
    for (int i = 0; i < depth; i++) {
        printf("  ");
    }
    if (self->ctx) {
        printf("%s", backward_name(self->ctx->backward));
    }
    else {
        printf("leaf");
    }
    printf(" [");
    for (int i = 0; i < self->ndim; i++) {
        printf("%d%s", self->shape[i], i < self->ndim - 1 ? ", " : "");
    }
    printf("]");
    if (self->requires_grad) {
        printf(" (requires_grad)");
    }
    printf("\n");
    if (self->ctx) {
        for (int i = 0; i < self->ctx->num_inputs; i++) {
            print_backward(self->ctx->input[i], depth + 1);
        }
    }
}

void tensor_print_backward(const Tensor *self)
{
    assert(self);
    print_backward(self, 0);
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
