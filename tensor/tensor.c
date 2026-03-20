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
Tensor *tensor_uniform(const int *shape, int ndim, float low, float high);
Tensor *tensor_normal(const int *shape, int ndim, float mean, float std);

// movement

Tensor *tensor_clone(const Tensor *src);
Tensor *tensor_contiguous(const Tensor *src);
Tensor *tensor_view(const Tensor *src, const int *shape, int ndim);
Tensor *tensor_reshape(const Tensor *src, const int *shape, int ndim);
Tensor *tensor_unsqueeze(const Tensor *src, int dim);
Tensor *tensor_squeeze(const Tensor *src, int dim);
Tensor *tensor_squeeze_all(const Tensor *src);
Tensor *tensor_permute(const Tensor *src, const int *order);
Tensor *tensor_transpose(const Tensor *src, int dim0, int dim1);
Tensor *tensor_cat(const Tensor **src, int num, int dim);
Tensor *tensor_stack(const Tensor **src, int num, int dim);

// i/o

void tensor_print(const Tensor *self);

//
// --- SOURCE ---
//

#include <assert.h>
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

static void stack_clear(void *until)
{
    while (head) {
        void *ptr = head->ptr;
        free(ptr);
        Stack *prev = head->prev;
        free(head);
        head = prev;
        if (ptr == until) {
            return;
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
    assert(steps > 0);
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

Tensor *tensor_uniform(const int *shape, int ndim, float low, float high)
{
    Tensor *out = tensor_rand(shape, ndim);
    for (long i = 0; i < out->numel; i++) {
        out->data[i] = low + (out->data[i] * (high - low));
    }
    return out;
}

Tensor *tensor_normal(const int *shape, int ndim, float mean, float std)
{
    assert(std >= 0);
    Tensor *out = tensor_randn(shape, ndim);
    for (long i = 0; i < out->numel; i++) {
        out->data[i] = mean + (out->data[i] * std);
    }
    return out;
}

// movement

static void copy_data(const Tensor *src, Tensor *out, int dim, long off_src, long off_out)
{
    if (dim == src->ndim - 1) {
        if (src->stride[dim] == 1) {
            memcpy(out->data + off_out, src->data + off_src, src->shape[dim] * sizeof(*src->data));
        }
        else {
            for (int i = 0; i < src->shape[dim]; i++) {
                out->data[off_out + i] = src->data[off_src + (i * src->stride[dim])];
            }
        }
    }
    else {
        for (int i = 0; i < src->shape[dim]; i++) {
            copy_data(src, out, dim + 1, off_src + (i * src->stride[dim]),
                      off_out + (i * out->stride[dim]));
        }
    }
}

Tensor *tensor_clone(const Tensor *src)
{
    assert(src);
    Tensor *out = tensor_empty(src->shape, src->ndim);
    copy_data(src, out, 0, 0, 0);
    return out;
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

Tensor *tensor_contiguous(const Tensor *src)
{
    assert(src);
    return is_contiguous(src) ? (Tensor *)src : tensor_clone(src);
}

static int valid_view(const Tensor *src, const int *shape, int ndim)
{
    return valid_shape(shape, ndim) && compute_numel(shape, ndim) == src->numel;
}

Tensor *tensor_view(const Tensor *src, const int *shape, int ndim)
{
    assert(src && is_contiguous(src) && valid_view(src, shape, ndim));
    Tensor *out = stack_calloc(1, sizeof(*out));
    out->ndim = ndim;
    out->numel = src->numel;
    if (ndim > 0) {
        memcpy(out->shape, shape, ndim * sizeof(*shape));
        compute_stride(out->stride, shape, ndim);
    }
    out->data = src->data;
    return out;
}

Tensor *tensor_reshape(const Tensor *src, const int *shape, int ndim)
{
    return tensor_view(tensor_contiguous(src), shape, ndim);
}

static int normalize_dim(int dim, int ndim)
{
    assert(-ndim <= dim && dim < ndim);
    return (dim < 0) ? (dim + ndim) : dim;
}

Tensor *tensor_unsqueeze(const Tensor *src, int dim)
{
    assert(src && src->ndim < MAX_NDIM);
    dim = normalize_dim(dim, src->ndim + 1);
    int shape[MAX_NDIM];
    for (int i = 0; i < dim; i++) {
        shape[i] = src->shape[i];
    }
    shape[dim] = 1;
    for (int i = dim; i < src->ndim; i++) {
        shape[i + 1] = src->shape[i];
    }
    return tensor_reshape(src, shape, src->ndim + 1);
}

Tensor *tensor_squeeze(const Tensor *src, int dim)
{
    assert(src);
    dim = normalize_dim(dim, src->ndim);
    assert(src->shape[dim] == 1);
    int shape[MAX_NDIM];
    for (int i = 0; i < dim; i++) {
        shape[i] = src->shape[i];
    }
    for (int i = dim; i < src->ndim; i++) {
        shape[i] = src->shape[i + 1];
    }
    return tensor_reshape(src, shape, src->ndim - 1);
}

Tensor *tensor_squeeze_all(const Tensor *src)
{
    assert(src);
    int shape[MAX_NDIM];
    int ndim = 0;
    for (int i = 0; i < src->ndim; i++) {
        if (src->shape[i] != 1) {
            shape[ndim++] = src->shape[i];
        }
    }
    return tensor_reshape(src, shape, ndim);
}

Tensor *tensor_permute(const Tensor *src, const int *order)
{
}

Tensor *tensor_transpose(const Tensor *src, int dim0, int dim1)
{
}

Tensor *tensor_cat(const Tensor **src, int num, int dim)
{
}

Tensor *tensor_stack(const Tensor **src, int num, int dim)
{
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

    Tensor *ten = tensor_range(1, 9, 1);
    tensor_print(ten);

    ten = tensor_reshape(ten, (int[]){3, 3}, 2);
    tensor_print(ten);

    ten = tensor_unsqueeze(ten, -1);
    tensor_print(ten);

    ten = tensor_squeeze_all(ten);
    tensor_print(ten);

    stack_clear(0);
}
