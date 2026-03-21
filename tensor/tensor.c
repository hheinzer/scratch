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

// i/o

void tensor_print(const Tensor *self);

//
// --- SOURCE ---
//

#include <assert.h>
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

typedef struct stack {
    void *ptr;
    struct stack *prev;
} Stack;

static Stack *head = 0;

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
    next->prev = head;
    head = next;
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
    if (!head) {
        return 0;
    }
    void *ptr = head->ptr;
    Stack *prev = head->prev;
    free(head);
    head = prev;
    return ptr;
}

static void stack_clear(void *until)
{
    while (head) {
        void *ptr = stack_pop();
        free(ptr);
        if (ptr == until) {
            break;
        }
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

static void pack_data(const Tensor *src, Tensor *out, int dim, long off_src, long *offset)
{
    if (dim == src->ndim - 1) {
        long off_out = *offset;
        if (src->stride[dim] == 1) {
            memcpy(out->data + off_out, src->data + off_src, src->shape[dim] * sizeof(*src->data));
        }
        else {
            for (int i = 0; i < src->shape[dim]; i++) {
                out->data[off_out + i] = src->data[off_src + (i * src->stride[dim])];
            }
        }
        *offset += src->shape[dim];
    }
    else {
        for (int i = 0; i < src->shape[dim]; i++) {
            pack_data(src, out, dim + 1, off_src + (i * src->stride[dim]), offset);
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
        pack_data(src, out, 0, 0, &offset);
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

static void cat_data(const Tensor *src, Tensor *dst, int dim, long off_src, long off_dst)
{
    if (dim == src->ndim - 1) {
        if (src->stride[dim] == 1 && dst->stride[dim] == 1) {
            memcpy(dst->data + off_dst, src->data + off_src, src->shape[dim] * sizeof(*src->data));
        }
        else {
            for (int i = 0; i < src->shape[dim]; i++) {
                dst->data[off_dst + (i * dst->stride[dim])] =
                    src->data[off_src + (i * src->stride[dim])];
            }
        }
    }
    else {
        for (int i = 0; i < src->shape[dim]; i++) {
            cat_data(src, dst, dim + 1, off_src + (i * src->stride[dim]),
                     off_dst + (i * dst->stride[dim]));
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
        cat_data(src[i], out, 0, 0, offset * out->stride[dim]);
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

// i/o

static void print_data(const Tensor *self, int dim, long off)
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
            print_data(self, dim + 1, off + (i * self->stride[dim]));
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

int main(void)
{
    srand(time(0));

    Tensor *ten = tensor_range(1, 16, 1);
    tensor_print(ten);

    ten = tensor_reshape(ten, (int[]){4, 4}, 2);
    tensor_print(ten);

    ten = tensor_unsqueeze(ten, -2);
    tensor_print(ten);

    ten = tensor_squeeze_all(ten);
    tensor_print(ten);

    ten = tensor_transpose(ten, 0, -1);
    tensor_print(ten);

    ten = tensor_slice(ten, 0, INT_MIN, INT_MAX, -1);
    tensor_print(ten);

    ten = tensor_stack((const Tensor *[]){ten, ten}, 2, 0);
    tensor_print(ten);

    ten = tensor_flatten_all(ten);
    tensor_print(ten);

    ten = tensor_reshape(ten, (int[]){-1, 4, 4}, 3);
    tensor_print(ten);

    ten = tensor_select(ten, 0, 0);
    tensor_print(ten);

    ten = tensor_expand(ten, (int[]){2, -1, -1}, 3);
    tensor_print(ten);

    stack_clear(0);
}
