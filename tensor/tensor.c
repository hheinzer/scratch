#include "tensor.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_BLAS
#include <cblas.h>
const char *__lsan_default_options(void)  // NOLINT(bugprone-reserved-identifier)
{
    return "print_suppressions=0";
}
const char *__lsan_default_suppressions(void)  // NOLINT(bugprone-reserved-identifier)
{
    // suppress openblas/openmp internal leak from cpu affinity detection at startup
    return "leak:gotoblas_init\n";
}
#endif

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
enum { MAX_INPUTS = 8 };
enum { MAX_SAVED = 3 };

typedef union {
    int axis;
    int beg;
    int dim;
    int keepdim;
    int padding;
    int perm[MAX_NDIM];
    int step;
    int stride;
} Saved;

typedef struct {
    int inputs;
    Tensor *input[MAX_INPUTS];
    Saved saved[MAX_SAVED];
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

Tensor *tensor_uniform(const int *shape, int ndim, float low, float high)
{
    Tensor *out = tensor_rand(shape, ndim);
    for (long i = 0; i < out->numel; i++) {
        out->data[i] = low + (out->data[i] * (high - low));
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

Tensor *tensor_normal(const int *shape, int ndim, float mean, float std)
{
    Tensor *out = tensor_randn(shape, ndim);
    for (long i = 0; i < out->numel; i++) {
        out->data[i] = mean + (out->data[i] * std);
    }
    return out;
}

// movement

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

static void ensure_grad(Tensor *self)
{
    if (!self->grad) {
        self->grad = stack_calloc(self->numel, sizeof(*self->grad));
    }
    stack_save();
}

static void reduce_grad(const Tensor *self, const long *stride_self, long offset_self,
                        const Tensor *src, long offset_src, int dim_self, int dim_src)
{
    if (dim_src == src->ndim) {
        self->grad[offset_self] += src->data[offset_src];
        return;
    }
    int extra = src->ndim - self->ndim;
    if (dim_src < extra) {
        for (int i = 0; i < src->shape[dim_src]; i++) {
            reduce_grad(self, stride_self, offset_self, src,
                        offset_src + (i * src->stride[dim_src]), dim_self, dim_src + 1);
        }
    }
    else if (self->shape[dim_self] == 1) {
        for (int i = 0; i < src->shape[dim_src]; i++) {
            reduce_grad(self, stride_self, offset_self, src,
                        offset_src + (i * src->stride[dim_src]), dim_self + 1, dim_src + 1);
        }
    }
    else {
        for (int i = 0; i < self->shape[dim_self]; i++) {
            reduce_grad(self, stride_self, offset_self + (i * stride_self[dim_self]), src,
                        offset_src + (i * src->stride[dim_src]), dim_self + 1, dim_src + 1);
        }
    }
}

static void accumulate_grad(Tensor *self, const Tensor *grad)
{
    int fast = (self->ndim == grad->ndim);
    for (int i = 0; i < self->ndim && fast; i++) {
        fast &= (self->shape[i] == grad->shape[i]);
    }
    if (fast) {
        long stride = 1;
        for (int i = grad->ndim - 1; i >= 0 && fast; i--) {
            if (grad->stride[i] != stride) {
                fast = 0;
            }
            stride *= grad->shape[i];
        }
    }
    if (fast) {
        for (long i = 0; i < self->numel; i++) {
            self->grad[i] += grad->data[i];
        }
    }
    else {
        long stride[MAX_NDIM];
        if (self->ndim > 0) {
            compute_stride(stride, self->shape, self->ndim);
        }
        reduce_grad(self, stride, 0, grad, 0, 0, 0);
    }
    stack_restore();
}

static void clone_backward(Tensor *out)
{
    Tensor *src = out->ctx->input[0];
    if (src->requires_grad) {
        ensure_grad(src);
        accumulate_grad(src, tensor_grad(out));
    }
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
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = clone_backward;
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

static void reshape_backward(Tensor *out)
{
    Tensor *src = out->ctx->input[0];
    if (src->requires_grad) {
        ensure_grad(src);
        Tensor *grad = tensor_grad(out);
        grad->ndim = src->ndim;
        memcpy(grad->shape, src->shape, src->ndim * sizeof(*src->shape));
        compute_stride(grad->stride, grad->shape, src->ndim);
        accumulate_grad(src, grad);
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
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = reshape_backward;
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
    int shape[MAX_NDIM];
    int ndim = 0;
    if (dim == INT_MAX) {
        for (int i = 0; i < src->ndim; i++) {
            if (src->shape[i] != 1) {
                shape[ndim++] = src->shape[i];
            }
        }
    }
    else {
        dim = normalize_dim(dim, src->ndim);
        assert(src->shape[dim] == 1);
        for (int i = 0; i < src->ndim; i++) {
            if (i != dim) {
                shape[ndim++] = src->shape[i];
            }
        }
    }
    return tensor_reshape(src, shape, ndim);
}

Tensor *tensor_unsqueeze(const Tensor *src, int dim)
{
    assert(src && src->ndim < MAX_NDIM);
    dim = normalize_dim(dim, src->ndim + 1);
    int shape[MAX_NDIM];
    int ndim = 0;
    for (int i = 0; i < dim; i++) {
        shape[ndim++] = src->shape[i];
    }
    shape[ndim++] = 1;
    for (int i = dim; i < src->ndim; i++) {
        shape[ndim++] = src->shape[i];
    }
    return tensor_reshape(src, shape, ndim);
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

static void permute_backward(Tensor *out)
{
    Tensor *src = out->ctx->input[0];
    if (src->requires_grad) {
        ensure_grad(src);
        int *perm = out->ctx->saved[0].perm;
        Tensor *grad = tensor_grad(out);
        int shape[MAX_NDIM];
        long stride[MAX_NDIM];
        for (int i = 0; i < src->ndim; i++) {
            shape[i] = grad->shape[perm[i]];
            stride[i] = grad->stride[perm[i]];
        }
        memcpy(grad->shape, shape, src->ndim * sizeof(*shape));
        memcpy(grad->stride, stride, src->ndim * sizeof(*stride));
        accumulate_grad(src, grad);
    }
}

Tensor *tensor_permute(const Tensor *src, const int *order_)
{
    assert(src && order_);
    int order[MAX_NDIM];
    for (int i = 0; i < src->ndim; i++) {
        order[i] = normalize_dim(order_[i], src->ndim);
    }
    assert(valid_permute(src, order));
    Tensor *out = tensor_detach(src);
    for (int i = 0; i < src->ndim; i++) {
        out->shape[i] = src->shape[order[i]];
        out->stride[i] = src->stride[order[i]];
    }
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        for (int i = 0; i < src->ndim; i++) {
            out->ctx->saved[0].perm[order[i]] = i;
        }
        out->ctx->backward = permute_backward;
    }
    return out;
}

Tensor *tensor_transpose(const Tensor *src, int dim0, int dim1)
{
    assert(src);
    dim0 = normalize_dim(dim0, src->ndim);
    dim1 = normalize_dim(dim1, src->ndim);
    int order[MAX_NDIM];
    for (int i = 0; i < src->ndim; i++) {
        order[i] = i;
    }
    order[dim0] = dim1;
    order[dim1] = dim0;
    return tensor_permute(src, order);
}

static void scatter_add(Tensor *out, long offset_out, const Tensor *src, long offset_src, int dim)
{
    if (dim == src->ndim - 1) {
        for (int i = 0; i < src->shape[dim]; i++) {
            out->data[offset_out + (i * out->stride[dim])] +=
                src->data[offset_src + (i * src->stride[dim])];
        }
    }
    else {
        for (int i = 0; i < src->shape[dim]; i++) {
            scatter_add(out, offset_out + (i * out->stride[dim]), src,
                        offset_src + (i * src->stride[dim]), dim + 1);
        }
    }
}

static void slice_backward(Tensor *out)
{
    Tensor *src = out->ctx->input[0];
    if (src->requires_grad) {
        ensure_grad(src);
        int dim = out->ctx->saved[0].dim;
        int beg = out->ctx->saved[1].beg;
        int step = out->ctx->saved[2].step;
        Tensor grad = {0};
        grad.ndim = src->ndim;
        grad.numel = out->numel;
        memcpy(grad.shape, out->shape, src->ndim * sizeof(*out->shape));
        compute_stride(grad.stride, src->shape, src->ndim);
        grad.data = src->grad + (beg * grad.stride[dim]);
        grad.stride[dim] *= step;
        scatter_add(&grad, 0, tensor_grad(out), 0, 0);
        stack_restore();
    }
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
    assert((step > 0) ? (beg <= end && end <= size) : (-1 <= end && end <= beg));
    Tensor *out = tensor_detach(src);
    out->data += beg * src->stride[dim];
    if (step > 0) {
        out->shape[dim] = (end - beg + step - 1) / step;
    }
    else {
        out->shape[dim] = (end - beg + step + 1) / step;
    }
    out->stride[dim] *= step;
    out->numel = compute_numel(out->shape, out->ndim);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->saved[0].dim = dim;
        out->ctx->saved[1].beg = beg;
        out->ctx->saved[2].step = step;
        out->ctx->backward = slice_backward;
    }
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

static void expand_backward(Tensor *out)
{
    Tensor *src = out->ctx->input[0];
    if (src->requires_grad) {
        ensure_grad(src);
        accumulate_grad(src, tensor_grad(out));
    }
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
    Tensor *out = tensor_detach(src);
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
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = expand_backward;
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

static int any_requires_grad(const Tensor **src, int num)
{
    int any = 0;
    for (int i = 0; i < num; i++) {
        if (src[i]->requires_grad) {
            any = 1;
            break;
        }
    }
    return any;
}

static void cat_backward(Tensor *out)
{
    int dim = out->ctx->saved[0].dim;
    Tensor base = *tensor_grad(out);
    int offset = 0;
    for (int i = 0; i < out->ctx->inputs; i++) {
        Tensor *src = out->ctx->input[i];
        int len = src->shape[dim];
        if (src->requires_grad) {
            ensure_grad(src);
            Tensor grad = base;
            grad.data += offset * base.stride[dim];
            grad.shape[dim] = len;
            grad.numel = src->numel;
            accumulate_grad(src, &grad);
        }
        offset += len;
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
    if (g_grad_enabled && any_requires_grad(src, num)) {
        assert(num <= MAX_INPUTS);
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = num;
        for (int i = 0; i < num; i++) {
            out->ctx->input[i] = (Tensor *)src[i];
        }
        out->ctx->saved[0].dim = dim;
        out->ctx->backward = cat_backward;
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

static void neg_backward(Tensor *out)
{
    // out = -src  =>  d/d(src) = -1
    Tensor *src = out->ctx->input[0];
    if (src->requires_grad) {
        ensure_grad(src);
        accumulate_grad(src, tensor_neg(tensor_grad(out)));
    }
}

Tensor *tensor_neg(const Tensor *src)
{
    Tensor *out = unary(src, neg_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = neg_backward;
    }
    return out;
}

static void abs_backward(Tensor *out)
{
    // out = |src|  =>  d/d(src) = sign(src)
    Tensor *src = out->ctx->input[0];
    if (src->requires_grad) {
        ensure_grad(src);
        accumulate_grad(src, tensor_mul(tensor_grad(out), tensor_sign(src)));
    }
}

Tensor *tensor_abs(const Tensor *src)
{
    Tensor *out = unary(src, abs_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = abs_backward;
    }
    return out;
}

Tensor *tensor_sign(const Tensor *src)
{
    return unary(src, sign_kernel);
}

Tensor *tensor_square(const Tensor *src)
{
    return tensor_mul(src, src);
}

static void sqrt_backward(Tensor *out)
{
    // out = sqrt(src)  =>  d/d(src) = 1/(2*out)
    Tensor *src = out->ctx->input[0];
    if (src->requires_grad) {
        ensure_grad(src);
        accumulate_grad(src, tensor_mul(tensor_grad(out), tensor_div(tensor_scalar(0.5F), out)));
    }
}

Tensor *tensor_sqrt(const Tensor *src)
{
    Tensor *out = unary(src, sqrt_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = sqrt_backward;
    }
    return out;
}

static void rsqrt_backward(Tensor *out)
{
    // out = 1/sqrt(src)  =>  d/d(src) = -out^3/2
    Tensor *src = out->ctx->input[0];
    if (src->requires_grad) {
        ensure_grad(src);
        accumulate_grad(
            src, tensor_mul(tensor_scalar(-0.5F),
                            tensor_mul(tensor_grad(out), tensor_mul(out, tensor_square(out)))));
    }
}

Tensor *tensor_rsqrt(const Tensor *src)
{
    Tensor *out = unary(src, rsqrt_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = rsqrt_backward;
    }
    return out;
}

static void exp_backward(Tensor *out)
{
    // out = exp(src)  =>  d/d(src) = out
    Tensor *src = out->ctx->input[0];
    if (src->requires_grad) {
        ensure_grad(src);
        accumulate_grad(src, tensor_mul(tensor_grad(out), out));
    }
}

Tensor *tensor_exp(const Tensor *src)
{
    Tensor *out = unary(src, exp_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = exp_backward;
    }
    return out;
}

static void log_backward(Tensor *out)
{
    // out = log(src)  =>  d/d(src) = 1/src
    Tensor *src = out->ctx->input[0];
    if (src->requires_grad) {
        ensure_grad(src);
        accumulate_grad(src, tensor_div(tensor_grad(out), src));
    }
}

Tensor *tensor_log(const Tensor *src)
{
    Tensor *out = unary(src, log_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = log_backward;
    }
    return out;
}

static void relu_backward(Tensor *out)
{
    // out = relu(src)  =>  d/d(src) = sign(out)  (1 if src > 0, else 0)
    Tensor *src = out->ctx->input[0];
    if (src->requires_grad) {
        ensure_grad(src);
        accumulate_grad(src, tensor_mul(tensor_grad(out), tensor_sign(out)));
    }
}

Tensor *tensor_relu(const Tensor *src)
{
    Tensor *out = unary(src, relu_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = relu_backward;
    }
    return out;
}

static void sigmoid_backward(Tensor *out)
{
    // out = sigmoid(src)  =>  d/d(src) = out*(1-out)
    Tensor *src = out->ctx->input[0];
    if (src->requires_grad) {
        ensure_grad(src);
        accumulate_grad(
            src, tensor_mul(tensor_grad(out), tensor_mul(out, tensor_sub(tensor_scalar(1), out))));
    }
}

Tensor *tensor_sigmoid(const Tensor *src)
{
    Tensor *out = unary(src, sigmoid_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->backward = sigmoid_backward;
    }
    return out;
}

static void tanh_backward(Tensor *out)
{
    // out = tanh(src)  =>  d/d(src) = 1-out^2
    Tensor *src = out->ctx->input[0];
    if (src->requires_grad) {
        ensure_grad(src);
        accumulate_grad(
            src, tensor_mul(tensor_grad(out), tensor_sub(tensor_scalar(1), tensor_square(out))));
    }
}

Tensor *tensor_tanh(const Tensor *src)
{
    Tensor *out = unary(src, tanh_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 1;
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
    int shape[MAX_NDIM] = {0};
    int ndim = broadcast_binary(shape, lhs, rhs);
    tensor_no_grad_begin();
    lhs = tensor_expand(lhs, shape, ndim);
    rhs = tensor_expand(rhs, shape, ndim);
    tensor_no_grad_end();
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
    if (lhs->requires_grad) {
        ensure_grad(lhs);
        accumulate_grad(lhs, tensor_grad(out));
    }
    if (rhs->requires_grad) {
        ensure_grad(rhs);
        accumulate_grad(rhs, tensor_grad(out));
    }
}

Tensor *tensor_add(const Tensor *lhs, const Tensor *rhs)
{
    Tensor *out = binary(lhs, rhs, add_kernel);
    if (g_grad_enabled && (lhs->requires_grad || rhs->requires_grad)) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 2;
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
    if (lhs->requires_grad) {
        ensure_grad(lhs);
        accumulate_grad(lhs, tensor_grad(out));
    }
    if (rhs->requires_grad) {
        ensure_grad(rhs);
        accumulate_grad(rhs, tensor_neg(tensor_grad(out)));
    }
}

Tensor *tensor_sub(const Tensor *lhs, const Tensor *rhs)
{
    Tensor *out = binary(lhs, rhs, sub_kernel);
    if (g_grad_enabled && (lhs->requires_grad || rhs->requires_grad)) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 2;
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
    if (lhs->requires_grad) {
        ensure_grad(lhs);
        accumulate_grad(lhs, tensor_mul(tensor_grad(out), rhs));
    }
    if (rhs->requires_grad) {
        ensure_grad(rhs);
        accumulate_grad(rhs, tensor_mul(lhs, tensor_grad(out)));
    }
}

Tensor *tensor_mul(const Tensor *lhs, const Tensor *rhs)
{
    Tensor *out = binary(lhs, rhs, mul_kernel);
    if (g_grad_enabled && (lhs->requires_grad || rhs->requires_grad)) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 2;
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
    if (lhs->requires_grad) {
        ensure_grad(lhs);
        accumulate_grad(lhs, tensor_div(tensor_grad(out), rhs));
    }
    if (rhs->requires_grad) {
        ensure_grad(rhs);
        accumulate_grad(rhs, tensor_neg(tensor_mul(tensor_grad(out), tensor_div(out, rhs))));
    }
}

Tensor *tensor_div(const Tensor *lhs, const Tensor *rhs)
{
    Tensor *out = binary(lhs, rhs, div_kernel);
    if (g_grad_enabled && (lhs->requires_grad || rhs->requires_grad)) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 2;
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
    if (lhs->requires_grad) {
        ensure_grad(lhs);
        accumulate_grad(
            lhs, tensor_mul(tensor_grad(out),
                            tensor_mul(rhs, tensor_pow(lhs, tensor_sub(rhs, tensor_scalar(1))))));
    }
    if (rhs->requires_grad) {
        ensure_grad(rhs);
        accumulate_grad(rhs, tensor_mul(tensor_grad(out), tensor_mul(out, tensor_log(lhs))));
    }
}

Tensor *tensor_pow(const Tensor *lhs, const Tensor *rhs)
{
    Tensor *out = binary(lhs, rhs, pow_kernel);
    if (g_grad_enabled && (lhs->requires_grad || rhs->requires_grad)) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 2;
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
    int shape[MAX_NDIM] = {0};
    int ndim = broadcast_ternary(shape, lhs, mid, rhs);
    tensor_no_grad_begin();
    lhs = tensor_expand(lhs, shape, ndim);
    mid = tensor_expand(mid, shape, ndim);
    rhs = tensor_expand(rhs, shape, ndim);
    tensor_no_grad_end();
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
    if (if_true->requires_grad) {
        ensure_grad(if_true);
        accumulate_grad(if_true, tensor_where(cond, tensor_grad(out), tensor_scalar(0)));
    }
    if (if_false->requires_grad) {
        ensure_grad(if_false);
        accumulate_grad(if_false, tensor_where(cond, tensor_scalar(0), tensor_grad(out)));
    }
}

Tensor *tensor_where(const Tensor *cond, const Tensor *if_true, const Tensor *if_false)
{
    Tensor *out = ternary(cond, if_true, if_false, where_kernel);
    if (g_grad_enabled && (if_true->requires_grad || if_false->requires_grad)) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 3;
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
    if (src->requires_grad) {
        ensure_grad(src);
        accumulate_grad(src, tensor_mul(tensor_grad(out),
                                        tensor_sub(tensor_scalar(1),
                                                   tensor_abs(tensor_sign(tensor_sub(out, src))))));
    }
    if (min->requires_grad) {
        ensure_grad(min);
        accumulate_grad(
            min, tensor_mul(tensor_grad(out), tensor_relu(tensor_sign(tensor_sub(out, src)))));
    }
    if (max->requires_grad) {
        ensure_grad(max);
        accumulate_grad(max,
                        tensor_mul(tensor_grad(out),
                                   tensor_relu(tensor_neg(tensor_sign(tensor_sub(out, src))))));
    }
}

Tensor *tensor_clamp(const Tensor *src, const Tensor *min, const Tensor *max)
{
    Tensor *out = ternary(src, min, max, clamp_kernel);
    if (g_grad_enabled && (src->requires_grad || min->requires_grad || max->requires_grad)) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 3;
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
        ensure_grad(src);
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
        out->ctx->inputs = 1;
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
        ensure_grad(src);
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
        out->ctx->inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->saved[0].axis = axis;
        out->ctx->saved[1].keepdim = keepdim;
        out->ctx->backward = max_backward;
    }
    return out;
}

static void sum_backward(Tensor *out)
{
    // out = sum(src, axis)  =>  d/d(src) = 1 broadcast to src->shape
    Tensor *src = out->ctx->input[0];
    if (src->requires_grad) {
        ensure_grad(src);
        int axis = out->ctx->saved[0].axis;
        int keepdim = out->ctx->saved[1].keepdim;
        Tensor *grad = tensor_grad(out);
        if (!keepdim && axis != INT_MAX) {
            grad = tensor_unsqueeze(grad, axis);
        }
        accumulate_grad(src, tensor_expand(grad, src->shape, src->ndim));
    }
}

Tensor *tensor_sum(const Tensor *src, int axis, int keepdim)
{
    Tensor *out = reduce(src, axis, keepdim, sum_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 1;
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
        ensure_grad(src);
        int axis = out->ctx->saved[0].axis;
        int keepdim = out->ctx->saved[1].keepdim;
        long count = (axis == INT_MAX) ? src->numel : src->shape[normalize_dim(axis, src->ndim)];
        Tensor *grad = tensor_mul(tensor_grad(out), tensor_scalar(1 / (float)count));
        if (!keepdim && axis != INT_MAX) {
            grad = tensor_unsqueeze(grad, axis);
        }
        accumulate_grad(src, tensor_expand(grad, src->shape, src->ndim));
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
        out->ctx->inputs = 1;
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
        ensure_grad(src);
        int axis = out->ctx->saved[0].axis;
        int keepdim = out->ctx->saved[1].keepdim;
        long count = (axis == INT_MAX) ? src->numel : src->shape[normalize_dim(axis, src->ndim)];
        Tensor *grad = tensor_grad(out);
        if (!keepdim && axis != INT_MAX) {
            grad = tensor_unsqueeze(grad, axis);
        }
        accumulate_grad(src, tensor_mul(tensor_mul(grad, tensor_scalar(2 / (float)count)),
                                        tensor_sub(src, tensor_mean(src, axis, 1))));
    }
}

Tensor *tensor_var(const Tensor *src, int axis, int keepdim)
{
    Tensor *out = reduce(src, axis, keepdim, var_kernel);
    if (g_grad_enabled && src->requires_grad) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 1;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->saved[0].axis = axis;
        out->ctx->saved[1].keepdim = keepdim;
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
    if (lhs->requires_grad) {
        ensure_grad(lhs);
        accumulate_grad(lhs, tensor_matmul(tensor_grad(out), tensor_transpose(rhs, -2, -1)));
    }
    if (rhs->requires_grad) {
        ensure_grad(rhs);
        accumulate_grad(rhs, tensor_matmul(tensor_transpose(lhs, -2, -1), tensor_grad(out)));
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
#ifdef USE_BLAS
    assert(stride_lhs <= INT_MAX && stride_rhs <= INT_MAX && stride_out <= INT_MAX);
    cblas_sgemm(CblasRowMajor, trans_lhs ? CblasTrans : CblasNoTrans,
                trans_rhs ? CblasTrans : CblasNoTrans, rows, cols, inner, 1, lhs, (int)stride_lhs,
                rhs, (int)stride_rhs, 0, out, (int)stride_out);
#else
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
#endif
}

Tensor *tensor_matmul(const Tensor *lhs, const Tensor *rhs)
{
    assert(lhs && rhs && lhs->ndim >= 2 && rhs->ndim >= 2);

#ifdef USE_BLAS
    static int blas_init = 0;
    if (!blas_init) {
        openblas_set_num_threads(1);
        blas_init = 1;
    }
#endif

    int rows = lhs->shape[lhs->ndim - 2];
    int cols = rhs->shape[rhs->ndim - 1];
    int inner = lhs->shape[lhs->ndim - 1];
    assert(inner == rhs->shape[rhs->ndim - 2]);

    int ndim = (lhs->ndim > rhs->ndim) ? lhs->ndim : rhs->ndim;
    int shape[MAX_NDIM];
    long batch = 1;
    for (int i = 0; i < ndim - 2; i++) {
        int dim_lhs = i - (ndim - lhs->ndim);
        int dim_rhs = i - (ndim - rhs->ndim);
        int size_lhs = (dim_lhs >= 0) ? lhs->shape[dim_lhs] : 1;
        int size_rhs = (dim_rhs >= 0) ? rhs->shape[dim_rhs] : 1;
        assert(size_lhs == size_rhs || size_lhs == 1 || size_rhs == 1);
        shape[i] = (size_lhs > size_rhs) ? size_lhs : size_rhs;
        batch *= shape[i];
    }
    shape[ndim - 2] = rows;
    shape[ndim - 1] = cols;

    Tensor *out = tensor_empty(shape, ndim);
    long stride_out = out->stride[out->ndim - 2];
    long stride_batch = (out->ndim > 2) ? out->stride[out->ndim - 3] : 0;

    if (g_grad_enabled && (lhs->requires_grad || rhs->requires_grad)) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = 2;
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

    for (long i = 0; i < batch; i++) {
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
        float *data_out = out->data + (i * stride_batch);
        const float *data_lhs = lhs->data + offset_lhs;
        const float *data_rhs = rhs->data + offset_rhs;
        matmul(data_out, stride_out, data_lhs, stride_lhs, data_rhs, stride_rhs, rows, cols, inner,
               trans_lhs, trans_rhs);
    }

    stack_restore();
    return out;
}

static void img2col(float *col, int h_wgt, int w_wgt, int h_out, int w_out, const float *src,
                    int c_src, int h_src, int w_src, int stride, int padding)
{
    for (int k = 0; k < c_src; k++) {
        for (int i = 0; i < h_wgt; i++) {
            for (int j = 0; j < w_wgt; j++) {
                int row = (((k * h_wgt) + i) * w_wgt) + j;
                for (int ii = 0; ii < h_out; ii++) {
                    int i_src = (ii * stride) - padding + i;
                    for (int jj = 0; jj < w_out; jj++) {
                        int j_src = (jj * stride) - padding + j;
                        if ((0 <= i_src && i_src < h_src) && (0 <= j_src && j_src < w_src)) {
                            col[(((row * h_out) + ii) * w_out) + jj] =
                                src[(((k * h_src) + i_src) * w_src) + j_src];
                        }
                        else {
                            col[(((row * h_out) + ii) * w_out) + jj] = 0;
                        }
                    }
                }
            }
        }
    }
}

static void col2img(float *grad_src, int c_src, int h_src, int w_src, const float *grad_col,
                    int h_wgt, int w_wgt, int h_out, int w_out, int stride, int padding)
{
    for (int k = 0; k < c_src; k++) {
        for (int i = 0; i < h_wgt; i++) {
            for (int j = 0; j < w_wgt; j++) {
                int row = (((k * h_wgt) + i) * w_wgt) + j;
                for (int ii = 0; ii < h_out; ii++) {
                    int i_src = (ii * stride) - padding + i;
                    for (int jj = 0; jj < w_out; jj++) {
                        int j_src = (jj * stride) - padding + j;
                        if (i_src >= 0 && i_src < h_src && j_src >= 0 && j_src < w_src) {
                            grad_src[(((k * h_src) + i_src) * w_src) + j_src] +=
                                grad_col[(((row * h_out) + ii) * w_out) + jj];
                        }
                    }
                }
            }
        }
    }
}

static void conv2d_backward(Tensor *out)
{
    // out = weight @ col(src) + bias
    // d/d(src) = col2img(weight.T @ grad)
    // d/d(weight) = sum_batch(grad @ col(src).T)
    // d/d(bias) = sum_batch_spatial(grad)
    Tensor *src = out->ctx->input[0];
    Tensor *weight = out->ctx->input[1];
    Tensor *bias = out->ctx->input[2];
    int stride = out->ctx->saved[0].stride;
    int padding = out->ctx->saved[1].padding;

    int batch = src->shape[0];
    int c_src = src->shape[1];
    int h_src = src->shape[2];
    int w_src = src->shape[3];
    int c_out = weight->shape[0];
    int h_wgt = weight->shape[2];
    int w_wgt = weight->shape[3];
    int h_out = out->shape[2];
    int w_out = out->shape[3];

    int cols = h_out * w_out;
    int inner = c_src * h_wgt * w_wgt;

    float *grad = out->grad;

    if (src->requires_grad) {
        ensure_grad(src);
        float *grad_col = stack_malloc((long)inner * cols, sizeof(*grad_col));
        for (long k = 0; k < batch; k++) {
            float *grad_out = grad + (k * c_out * cols);
            float *grad_src = src->grad + (k * c_src * h_src * w_src);
            // NOLINTNEXTLINE(readability-suspicious-call-argument)
            matmul(grad_col, cols, weight->data, inner, grad_out, cols, inner, cols, c_out, 1, 0);
            col2img(grad_src, c_src, h_src, w_src, grad_col, h_wgt, w_wgt, h_out, w_out, stride,
                    padding);
        }
        stack_restore();
    }

    if (weight->requires_grad) {
        ensure_grad(weight);
        float *col = stack_malloc((long)inner * cols, sizeof(*col));
        float *grad_wgt = stack_malloc((long)c_out * inner, sizeof(*grad_wgt));
        for (long k = 0; k < batch; k++) {
            float *grad_out = grad + (k * c_out * cols);
            float *data_src = src->data + (k * c_src * h_src * w_src);
            img2col(col, h_wgt, w_wgt, h_out, w_out, data_src, c_src, h_src, w_src, stride,
                    padding);
            // NOLINTNEXTLINE(readability-suspicious-call-argument)
            matmul(grad_wgt, inner, grad_out, cols, col, cols, c_out, inner, cols, 0, 1);
            for (long i = 0; i < ((long)c_out * inner); i++) {
                weight->grad[i] += grad_wgt[i];
            }
        }
        stack_restore();
    }

    if (bias && bias->requires_grad) {
        ensure_grad(bias);
        for (int k = 0; k < batch; k++) {
            for (int i = 0; i < c_out; i++) {
                for (int j = 0; j < cols; j++) {
                    bias->grad[(i * bias->stride[0])] += grad[(((k * c_out) + i) * cols) + j];
                }
            }
        }
        stack_restore();
    }
}

Tensor *tensor_conv2d(const Tensor *src, const Tensor *weight, const Tensor *bias, int stride,
                      int padding)
{
    assert(src && weight && src->ndim == 4 && weight->ndim == 4 && stride > 0 && padding >= 0);
    assert(!bias || (bias->ndim == 1 && bias->shape[0] == weight->shape[0]));

    int batch = src->shape[0];
    int c_src = src->shape[1];
    int h_src = src->shape[2];
    int w_src = src->shape[3];
    int c_out = weight->shape[0];
    assert(weight->shape[1] == c_src);
    int h_wgt = weight->shape[2];
    int w_wgt = weight->shape[3];
    int h_out = ((h_src + (2 * padding) - h_wgt) / stride) + 1;
    int w_out = ((w_src + (2 * padding) - w_wgt) / stride) + 1;

    Tensor *out = tensor_empty((int[]){batch, c_out, h_out, w_out}, 4);

    if (g_grad_enabled &&
        (src->requires_grad || weight->requires_grad || (bias && bias->requires_grad))) {
        out->requires_grad = 1;
        out->ctx = stack_calloc(1, sizeof(*out->ctx));
        out->ctx->inputs = bias ? 3 : 2;
        out->ctx->input[0] = (Tensor *)src;
        out->ctx->input[1] = (Tensor *)weight;
        out->ctx->input[2] = (Tensor *)bias;
        out->ctx->saved[0].stride = stride;
        out->ctx->saved[1].padding = padding;
        out->ctx->backward = conv2d_backward;
    }

    stack_save();

    int cols = h_out * w_out;
    int inner = c_src * h_wgt * w_wgt;

    float *col = stack_malloc((long)inner * cols, sizeof(*col));

    for (long k = 0; k < batch; k++) {
        float *data_src = src->data + (k * c_src * h_src * w_src);
        float *data_out = out->data + (k * c_out * h_out * w_out);
        img2col(col, h_wgt, w_wgt, h_out, w_out, data_src, c_src, h_src, w_src, stride, padding);
        matmul(data_out, cols, weight->data, inner, col, cols, c_out, cols, inner, 0, 0);
        if (bias) {
            for (int i = 0; i < c_out; i++) {
                for (int j = 0; j < cols; j++) {
                    data_out[(i * cols) + j] += bias->data[(i * bias->stride[0])];
                }
            }
        }
    }

    stack_restore();
    return out;
}

// loss

Tensor *tensor_mse(const Tensor *pred, const Tensor *target)
{
    return tensor_mean(tensor_square(tensor_sub(pred, target)), INT_MAX, 0);
}

Tensor *tensor_cross_entropy(const Tensor *logit, const Tensor *label)
{
    assert(logit && label && logit->ndim == 2 && label->ndim == 1);
    int batch = logit->shape[0];
    int classes = logit->shape[1];
    assert(label->shape[0] == batch);
    float (*buf)[classes] = stack_calloc(batch, sizeof(*buf));
    const float *idx = label->data;
    for (int i = 0; i < batch; i++) {
        buf[i][(int)idx[i * label->stride[0]]] = 1;
    }
    Tensor *one_hot = tensor_wrap((int[]){batch, classes}, 2, *buf);
    Tensor *shifted = tensor_sub(logit, tensor_max(logit, 1, 1));
    Tensor *log_softmax = tensor_sub(shifted, tensor_log(tensor_sum(tensor_exp(shifted), 1, 1)));
    return tensor_neg(tensor_mean(tensor_sum(tensor_mul(one_hot, log_softmax), 1, 0), INT_MAX, 0));
}

// utility

void tensor_shuffle(Tensor **self, int num, int axis)
{
    assert(self && self[0] && num > 0);
    axis = normalize_dim(axis, self[0]->ndim);
    int count = self[0]->shape[axis];
    for (int i = count - 1; i > 0; i--) {
        int idx = rand() % (i + 1);
        for (int j = 0; j < num; j++) {
            long inner = self[j]->stride[axis];
            long outer_step = count * inner;
            long outer_count = self[j]->numel / outer_step;
            float *data = self[j]->data;
            for (long ext = 0; ext < outer_count; ext++) {
                long base = ext * outer_step;
                for (long k = 0; k < inner; k++) {
                    float tmp = data[base + (i * inner) + k];
                    data[base + (i * inner) + k] = data[base + (idx * inner) + k];
                    data[base + (idx * inner) + k] = tmp;
                }
            }
        }
    }
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
        for (int i = 0; i < self->ctx->inputs; i++) {
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
