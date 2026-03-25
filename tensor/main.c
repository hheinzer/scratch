#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"

static const float m_e = 2.7182818284590452354F;

#define ensure(cond)                                                                         \
    do {                                                                                     \
        if (!(cond)) {                                                                       \
            fprintf(stderr, "%s:%d %s: `%s` failed\n", __FILE__, __LINE__, __func__, #cond); \
            abort();                                                                         \
        }                                                                                    \
    } while (0)

#define isclose(a, b) (fabsf((a) - (b)) < FLT_EPSILON)

static void test_creation(void)
{
    tensor_frame_begin();

    // tensor_empty
    Tensor *tensor = tensor_empty((int[]){2, 3}, 2);
    ensure(tensor_ndim(tensor) == 2 && tensor_numel(tensor) == 6);
    ensure(tensor_shape(tensor)[0] == 2 && tensor_shape(tensor)[1] == 3);
    ensure(tensor_stride(tensor)[0] == 3 && tensor_stride(tensor)[1] == 1);

    // tensor_empty 0-dim
    tensor = tensor_empty(0, 0);
    ensure(tensor_ndim(tensor) == 0 && tensor_numel(tensor) == 1);

    // tensor_zeros
    tensor = tensor_zeros((int[]){3}, 1);
    ensure(tensor_data(tensor)[0] == 0 && tensor_data(tensor)[1] == 0 &&
           tensor_data(tensor)[2] == 0);

    // tensor_ones
    tensor = tensor_ones((int[]){3}, 1);
    ensure(tensor_data(tensor)[0] == 1 && tensor_data(tensor)[1] == 1 &&
           tensor_data(tensor)[2] == 1);

    // tensor_fill
    tensor = tensor_fill((int[]){3}, 1, 5);
    ensure(tensor_data(tensor)[0] == 5 && tensor_data(tensor)[1] == 5 &&
           tensor_data(tensor)[2] == 5);

    // tensor_from
    float data[] = {1, 2, 3, 4};
    tensor = tensor_from((int[]){4}, 1, data);
    ensure(tensor_data(tensor)[0] == 1 && tensor_data(tensor)[3] == 4);
    data[0] = 99;
    ensure(tensor_data(tensor)[0] == 1);  // copy, not a reference

    // tensor_wrap
    float wrap_data[] = {1, 2, 3};
    tensor = tensor_wrap((int[]){3}, 1, wrap_data);
    ensure(tensor_data(tensor) == wrap_data);  // view, not a copy
    wrap_data[0] = 99;
    ensure(tensor_data(tensor)[0] == 99);  // reflects mutation

    // tensor_scalar
    tensor = tensor_scalar(7);
    ensure(tensor_ndim(tensor) == 0 && tensor_data(tensor)[0] == 7);

    // tensor_rand: values in [0, 1)
    tensor = tensor_rand((int[]){100}, 1);
    ensure(tensor_numel(tensor) == 100);
    for (long i = 0; i < tensor_numel(tensor); i++) {
        ensure(tensor_data(tensor)[i] >= 0 && tensor_data(tensor)[i] < 1);
    }

    // tensor_randn: correct shape
    tensor = tensor_randn((int[]){100}, 1);
    ensure(tensor_numel(tensor) == 100);

    tensor_frame_end();
}

static void test_movement(void)
{
    tensor_frame_begin();

    // tensor_reshape: [6] -> [2, 3]
    Tensor *tensor = tensor_from((int[]){6}, 1, (float[]){1, 2, 3, 4, 5, 6});
    tensor = tensor_reshape(tensor, (int[]){2, 3}, 2);
    ensure(tensor_ndim(tensor) == 2 && tensor_shape(tensor)[0] == 2 &&
           tensor_shape(tensor)[1] == 3);
    ensure(tensor_data(tensor)[0] == 1 && tensor_data(tensor)[5] == 6);

    // tensor_reshape: infer dim with -1, [6] -> [2, 3]
    tensor = tensor_from((int[]){6}, 1, (float[]){1, 2, 3, 4, 5, 6});
    tensor = tensor_reshape(tensor, (int[]){-1, 3}, 2);
    ensure(tensor_shape(tensor)[0] == 2 && tensor_shape(tensor)[1] == 3);

    // tensor_flatten: dims 0..1 of [2, 3] -> [6]
    tensor = tensor_from((int[]){6}, 1, (float[]){1, 2, 3, 4, 5, 6});
    tensor = tensor_reshape(tensor, (int[]){2, 3}, 2);
    tensor = tensor_flatten(tensor, 0, 1);
    ensure(tensor_ndim(tensor) == 1 && tensor_numel(tensor) == 6);

    // tensor_flatten: all dims (INT_MIN to INT_MAX) of [2, 3] -> [6]
    tensor = tensor_from((int[]){6}, 1, (float[]){1, 2, 3, 4, 5, 6});
    tensor = tensor_reshape(tensor, (int[]){2, 3}, 2);
    tensor = tensor_flatten(tensor, INT_MIN, INT_MAX);
    ensure(tensor_ndim(tensor) == 1 && tensor_numel(tensor) == 6);

    // tensor_squeeze: [1, 3, 1] remove dim 0 -> [3, 1]
    tensor = tensor_zeros((int[]){1, 3, 1}, 3);
    tensor = tensor_squeeze(tensor, 0);
    ensure(tensor_ndim(tensor) == 2 && tensor_shape(tensor)[0] == 3 &&
           tensor_shape(tensor)[1] == 1);

    // tensor_squeeze: negative dim
    tensor = tensor_zeros((int[]){1, 3, 1}, 3);
    tensor = tensor_squeeze(tensor, -1);
    ensure(tensor_ndim(tensor) == 2 && tensor_shape(tensor)[0] == 1 &&
           tensor_shape(tensor)[1] == 3);

    // tensor_squeeze: INT_MAX removes all size-1 dims, [1, 3, 1] -> [3]
    tensor = tensor_zeros((int[]){1, 3, 1}, 3);
    tensor = tensor_squeeze(tensor, INT_MAX);
    ensure(tensor_ndim(tensor) == 1 && tensor_shape(tensor)[0] == 3);

    // tensor_unsqueeze: [3] insert at 0 -> [1, 3]
    tensor = tensor_zeros((int[]){3}, 1);
    tensor = tensor_unsqueeze(tensor, 0);
    ensure(tensor_ndim(tensor) == 2 && tensor_shape(tensor)[0] == 1 &&
           tensor_shape(tensor)[1] == 3);

    // tensor_permute: [2, 3] -> [3, 2], check strides swapped
    tensor = tensor_from((int[]){6}, 1, (float[]){1, 2, 3, 4, 5, 6});
    tensor = tensor_reshape(tensor, (int[]){2, 3}, 2);
    tensor = tensor_permute(tensor, (int[]){1, 0});
    ensure(tensor_shape(tensor)[0] == 3 && tensor_shape(tensor)[1] == 2);
    ensure(tensor_stride(tensor)[0] == 1 && tensor_stride(tensor)[1] == 3);

    // tensor_transpose: same as permute for 2D
    tensor = tensor_from((int[]){6}, 1, (float[]){1, 2, 3, 4, 5, 6});
    tensor = tensor_reshape(tensor, (int[]){2, 3}, 2);
    tensor = tensor_transpose(tensor, 0, 1);
    ensure(tensor_shape(tensor)[0] == 3 && tensor_shape(tensor)[1] == 2);
    ensure(tensor_stride(tensor)[0] == 1 && tensor_stride(tensor)[1] == 3);
    // element [1][0]: data[1*1 + 0*3] = data[1] = 2
    ensure(tensor_data(tensor)[(tensor_stride(tensor)[0] * 1) + (tensor_stride(tensor)[1] * 0)] ==
           2);

    // tensor_slice: [0,1,2,3,4,5] slice [1:4] -> [1,2,3]
    tensor = tensor_from((int[]){6}, 1, (float[]){0, 1, 2, 3, 4, 5});
    tensor = tensor_slice(tensor, 0, 1, 4, 1);
    ensure(tensor_numel(tensor) == 3);
    ensure(tensor_data(tensor)[0] == 1 && tensor_data(tensor)[1] == 2 &&
           tensor_data(tensor)[2] == 3);

    // tensor_slice: reverse step, stride is -1, access via stride
    tensor = tensor_from((int[]){6}, 1, (float[]){0, 1, 2, 3, 4, 5});
    tensor = tensor_slice(tensor, 0, INT_MIN, INT_MAX, -1);
    ensure(tensor_numel(tensor) == 6 && tensor_stride(tensor)[0] == -1);
    ensure(tensor_data(tensor)[0] == 5 && tensor_data(tensor)[5 * tensor_stride(tensor)[0]] == 0);

    // tensor_slice: negative start index
    tensor = tensor_from((int[]){6}, 1, (float[]){0, 1, 2, 3, 4, 5});
    tensor = tensor_slice(tensor, 0, -4, -2, 1);  // beg=-4->2, end=-2->4 -> [2, 3]
    ensure(tensor_numel(tensor) == 2 && tensor_data(tensor)[0] == 2 && tensor_data(tensor)[1] == 3);

    // tensor_select: row 1 of [2, 3] -> [4, 5, 6]
    tensor = tensor_from((int[]){6}, 1, (float[]){1, 2, 3, 4, 5, 6});
    tensor = tensor_reshape(tensor, (int[]){2, 3}, 2);
    tensor = tensor_select(tensor, 0, 1);
    ensure(tensor_ndim(tensor) == 1 && tensor_numel(tensor) == 3);
    ensure(tensor_data(tensor)[0] == 4 && tensor_data(tensor)[1] == 5 &&
           tensor_data(tensor)[2] == 6);

    // tensor_select: negative index
    tensor = tensor_from((int[]){6}, 1, (float[]){1, 2, 3, 4, 5, 6});
    tensor = tensor_reshape(tensor, (int[]){2, 3}, 2);
    tensor = tensor_select(tensor, 0, -1);
    ensure(tensor_data(tensor)[0] == 4 && tensor_data(tensor)[2] == 6);

    // tensor_expand: [1, 3] -> [2, 3], broadcast dim gets stride 0
    tensor = tensor_from((int[]){1, 3}, 2, (float[]){1, 2, 3});
    tensor = tensor_expand(tensor, (int[]){2, 3}, 2);
    ensure(tensor_shape(tensor)[0] == 2 && tensor_shape(tensor)[1] == 3 &&
           tensor_stride(tensor)[0] == 0);
    ensure(tensor_data(tensor)[tensor_stride(tensor)[1] * 0] == 1 &&
           tensor_data(tensor)[tensor_stride(tensor)[1] * 2] == 3);

    // tensor_expand: -1 preserves dim
    tensor = tensor_from((int[]){1, 3}, 2, (float[]){1, 2, 3});
    tensor = tensor_expand(tensor, (int[]){2, -1}, 2);
    ensure(tensor_shape(tensor)[0] == 2 && tensor_shape(tensor)[1] == 3);

    // tensor_cat: [1,2,3] and [4,5,6] along dim 0 -> [1,2,3,4,5,6]
    Tensor *lhs = tensor_from((int[]){3}, 1, (float[]){1, 2, 3});
    Tensor *rhs = tensor_from((int[]){3}, 1, (float[]){4, 5, 6});
    tensor = tensor_cat((const Tensor *[]){lhs, rhs}, 2, 0);
    ensure(tensor_ndim(tensor) == 1 && tensor_numel(tensor) == 6);
    ensure(tensor_data(tensor)[0] == 1 && tensor_data(tensor)[5] == 6);

    // tensor_cat: along dim 1
    lhs = tensor_from((int[]){2, 2}, 2, (float[]){1, 2, 3, 4});
    rhs = tensor_from((int[]){2, 1}, 2, (float[]){5, 6});
    tensor = tensor_cat((const Tensor *[]){lhs, rhs}, 2, 1);
    ensure(tensor_shape(tensor)[0] == 2 && tensor_shape(tensor)[1] == 3);
    ensure(tensor_data(tensor)[2] == 5);  // first element of appended column

    // tensor_stack: stack [1,2,3] and [4,5,6] along dim 0 -> [2, 3]
    lhs = tensor_from((int[]){3}, 1, (float[]){1, 2, 3});
    rhs = tensor_from((int[]){3}, 1, (float[]){4, 5, 6});
    tensor = tensor_stack((const Tensor *[]){lhs, rhs}, 2, 0);
    ensure(tensor_ndim(tensor) == 2 && tensor_shape(tensor)[0] == 2 &&
           tensor_shape(tensor)[1] == 3);
    ensure(tensor_data(tensor)[0] == 1 && tensor_data(tensor)[3] == 4);

    // tensor_contiguous: transposed non-contiguous [2, 3] -> contiguous [3, 2]
    tensor = tensor_from((int[]){2, 3}, 2, (float[]){1, 2, 3, 4, 5, 6});
    tensor = tensor_transpose(tensor, 0, 1);
    tensor = tensor_contiguous(tensor);
    ensure(tensor_stride(tensor)[0] == 2 && tensor_stride(tensor)[1] == 1);
    ensure(tensor_data(tensor)[0] == 1 && tensor_data(tensor)[1] == 4);  // row 0: [1, 4]
    ensure(tensor_data(tensor)[2] == 2 && tensor_data(tensor)[3] == 5);  // row 1: [2, 5]
    ensure(tensor_data(tensor)[4] == 3 && tensor_data(tensor)[5] == 6);  // row 2: [3, 6]

    // tensor_clone: transposed [2, 3] -> independent contiguous [3, 2]
    tensor = tensor_from((int[]){2, 3}, 2, (float[]){1, 2, 3, 4, 5, 6});
    const Tensor *clone = tensor_clone(tensor_transpose(tensor, 0, 1));
    ensure(tensor_data(clone) != tensor_data(tensor));
    ensure(tensor_data(clone)[0] == 1 && tensor_data(clone)[1] == 4);  // row 0: [1, 4]
    ensure(tensor_data(clone)[2] == 2 && tensor_data(clone)[3] == 5);  // row 1: [2, 5]
    ensure(tensor_data(clone)[4] == 3 && tensor_data(clone)[5] == 6);  // row 2: [3, 6]

    tensor_frame_end();
}

static void test_unary(void)
{
    tensor_frame_begin();

    // tensor_neg
    Tensor *src = tensor_from((int[]){3}, 1, (float[]){-1, 0, 2});
    Tensor *out = tensor_neg(src);
    ensure(tensor_data(out)[0] == 1 && tensor_data(out)[1] == 0 && tensor_data(out)[2] == -2);

    // tensor_abs
    src = tensor_from((int[]){3}, 1, (float[]){-2, 0, 3});
    out = tensor_abs(src);
    ensure(tensor_data(out)[0] == 2 && tensor_data(out)[1] == 0 && tensor_data(out)[2] == 3);

    // tensor_sign
    src = tensor_from((int[]){3}, 1, (float[]){-5, 0, 3});
    out = tensor_sign(src);
    ensure(tensor_data(out)[0] == -1 && tensor_data(out)[1] == 0 && tensor_data(out)[2] == 1);

    // tensor_square
    src = tensor_from((int[]){3}, 1, (float[]){-2, 3, 4});
    out = tensor_square(src);
    ensure(tensor_data(out)[0] == 4 && tensor_data(out)[1] == 9 && tensor_data(out)[2] == 16);

    // tensor_sqrt
    src = tensor_from((int[]){3}, 1, (float[]){0, 1, 4});
    out = tensor_sqrt(src);
    ensure(tensor_data(out)[0] == 0 && tensor_data(out)[1] == 1 && isclose(tensor_data(out)[2], 2));

    // tensor_rsqrt
    src = tensor_from((int[]){2}, 1, (float[]){1, 4});
    out = tensor_rsqrt(src);
    ensure(isclose(tensor_data(out)[0], 1) && isclose(tensor_data(out)[1], 0.5F));

    // tensor_exp: exp(0)=1, exp(1)~=e
    src = tensor_from((int[]){2}, 1, (float[]){0, 1});
    out = tensor_exp(src);
    ensure(tensor_data(out)[0] == expf(0) && isclose(tensor_data(out)[1], m_e));

    // tensor_log: log(1)=0, log(e)~=1
    src = tensor_from((int[]){2}, 1, (float[]){1, m_e});
    out = tensor_log(src);
    ensure(tensor_data(out)[0] == logf(1) && isclose(tensor_data(out)[1], 1));

    // tensor_relu
    src = tensor_from((int[]){3}, 1, (float[]){-1, 0, 2});
    out = tensor_relu(src);
    ensure(tensor_data(out)[0] == 0 && tensor_data(out)[1] == 0 && tensor_data(out)[2] == 2);

    // tensor_sigmoid: sigmoid(0)=0.5, sigmoid(large)~=1
    src = tensor_from((int[]){2}, 1, (float[]){0, 1e6F});
    out = tensor_sigmoid(src);
    ensure(isclose(tensor_data(out)[0], 0.5F) && isclose(tensor_data(out)[1], 1));

    // tensor_tanh: tanh(0)=0, tanh(large)~=1
    src = tensor_from((int[]){2}, 1, (float[]){0, 1e6F});
    out = tensor_tanh(src);
    ensure(isclose(tensor_data(out)[0], 0) && isclose(tensor_data(out)[1], 1));

    // non-contiguous input (transposed)
    src = tensor_from((int[]){2, 2}, 2, (float[]){1, 2, 3, 4});
    src = tensor_transpose(src, 0, 1);
    out = tensor_neg(src);
    ensure(tensor_data(out)[0] == -1 && tensor_data(out)[1] == -3 && tensor_data(out)[2] == -2 &&
           tensor_data(out)[3] == -4);

    tensor_frame_end();
}

static void test_binary(void)
{
    tensor_frame_begin();

    Tensor *lhs = tensor_from((int[]){3}, 1, (float[]){1, 2, 3});
    Tensor *rhs = tensor_from((int[]){3}, 1, (float[]){4, 5, 6});

    // tensor_add
    Tensor *out = tensor_add(lhs, rhs);
    ensure(tensor_data(out)[0] == 5 && tensor_data(out)[1] == 7 && tensor_data(out)[2] == 9);

    // tensor_sub
    out = tensor_sub(lhs, rhs);
    ensure(tensor_data(out)[0] == -3 && tensor_data(out)[1] == -3 && tensor_data(out)[2] == -3);

    // tensor_mul
    out = tensor_mul(lhs, rhs);
    ensure(tensor_data(out)[0] == 4 && tensor_data(out)[1] == 10 && tensor_data(out)[2] == 18);

    // tensor_div
    out = tensor_div(lhs, rhs);
    ensure(isclose(tensor_data(out)[0], 0.25F) && isclose(tensor_data(out)[1], 0.4F) &&
           isclose(tensor_data(out)[2], 0.5F));

    // tensor_pow
    lhs = tensor_from((int[]){3}, 1, (float[]){2, 3, 4});
    rhs = tensor_from((int[]){3}, 1, (float[]){2, 2, 2});
    out = tensor_pow(lhs, rhs);
    ensure(tensor_data(out)[0] == 4 && tensor_data(out)[1] == 9 && tensor_data(out)[2] == 16);

    // broadcasting: [3] + [1] -> [3]
    lhs = tensor_from((int[]){3}, 1, (float[]){1, 2, 3});
    rhs = tensor_from((int[]){1}, 1, (float[]){10});
    out = tensor_add(lhs, rhs);
    ensure(tensor_data(out)[0] == 11 && tensor_data(out)[1] == 12 && tensor_data(out)[2] == 13);

    // broadcasting: [2, 3] + [3] -> [2, 3]
    lhs = tensor_from((int[]){2, 3}, 2, (float[]){1, 2, 3, 4, 5, 6});
    rhs = tensor_from((int[]){3}, 1, (float[]){10, 20, 30});
    out = tensor_add(lhs, rhs);
    ensure(tensor_data(out)[0] == 11 && tensor_data(out)[3] == 14 && tensor_data(out)[5] == 36);

    tensor_frame_end();
}

static void test_ternary(void)
{
    tensor_frame_begin();

    // tensor_where: [1,0,1] ? [10,20,30] : [40,50,60] -> [10,50,30]
    Tensor *cond = tensor_from((int[]){3}, 1, (float[]){1, 0, 1});
    Tensor *lhs = tensor_from((int[]){3}, 1, (float[]){10, 20, 30});
    Tensor *rhs = tensor_from((int[]){3}, 1, (float[]){40, 50, 60});
    Tensor *out = tensor_where(cond, lhs, rhs);
    ensure(tensor_data(out)[0] == 10 && tensor_data(out)[1] == 50 && tensor_data(out)[2] == 30);

    // tensor_clamp: [-2,0,3] clamped to [-1, 2] -> [-1,0,2]
    lhs = tensor_from((int[]){3}, 1, (float[]){-2, 0, 3});
    out = tensor_clamp(lhs, tensor_scalar(-1), tensor_scalar(2));
    ensure(tensor_data(out)[0] == -1 && tensor_data(out)[1] == 0 && tensor_data(out)[2] == 2);

    tensor_frame_end();
}

static void test_reduction(void)
{
    tensor_frame_begin();

    // tensor_min / tensor_max along axis
    Tensor *src = tensor_from((int[]){2, 3}, 2, (float[]){3, 1, 4, 1, 5, 9});
    Tensor *out = tensor_min(src, 1, 0);
    ensure(tensor_ndim(out) == 1 && tensor_shape(out)[0] == 2);
    ensure(tensor_data(out)[0] == 1 && tensor_data(out)[1] == 1);
    out = tensor_max(src, 1, 0);
    ensure(tensor_data(out)[0] == 4 && tensor_data(out)[1] == 9);

    // keepdim
    out = tensor_min(src, 0, 1);
    ensure(tensor_ndim(out) == 2 && tensor_shape(out)[0] == 1 && tensor_shape(out)[1] == 3);
    ensure(tensor_data(out)[0] == 1 && tensor_data(out)[1] == 1 && tensor_data(out)[2] == 4);

    // tensor_sum along axis 0: [[1,2,3],[4,5,6]] -> [5, 7, 9]
    src = tensor_from((int[]){2, 3}, 2, (float[]){1, 2, 3, 4, 5, 6});
    out = tensor_sum(src, 0, 0);
    ensure(tensor_data(out)[0] == 5 && tensor_data(out)[1] == 7 && tensor_data(out)[2] == 9);

    // tensor_mean along axis
    src = tensor_from((int[]){2, 3}, 2, (float[]){1, 2, 3, 4, 5, 6});
    out = tensor_mean(src, 0, 0);
    ensure(isclose(tensor_data(out)[0], 2.5F) && isclose(tensor_data(out)[1], 3.5F) &&
           isclose(tensor_data(out)[2], 4.5F));

    // full reduction (axis == INT_MAX): result is 0-dim scalar
    out = tensor_sum(src, INT_MAX, 0);
    ensure(tensor_ndim(out) == 0 && isclose(tensor_data(out)[0], 21));
    out = tensor_min(src, INT_MAX, 0);
    ensure(tensor_data(out)[0] == 1);
    out = tensor_max(src, INT_MAX, 0);
    ensure(tensor_data(out)[0] == 6);
    out = tensor_mean(src, INT_MAX, 0);
    ensure(isclose(tensor_data(out)[0], 3.5F));

    // full reduction keepdim
    out = tensor_sum(src, INT_MAX, 1);
    ensure(tensor_ndim(out) == 2 && tensor_shape(out)[0] == 1 && tensor_shape(out)[1] == 1);
    ensure(isclose(tensor_data(out)[0], 21));

    // non-contiguous input (transposed): [[1,2,3],[4,5,6]]^T = [[1,4],[2,5],[3,6]]
    src = tensor_from((int[]){2, 3}, 2, (float[]){1, 2, 3, 4, 5, 6});
    src = tensor_transpose(src, 0, 1);  // [3, 2]
    out = tensor_sum(src, 0, 0);        // sum along dim 0 -> [6, 15]
    ensure(tensor_data(out)[0] == 1 + 2 + 3 && tensor_data(out)[1] == 4 + 5 + 6);

    // tensor_var: [0, 2, 4] -> mean=2, var=8/3
    src = tensor_from((int[]){3}, 1, (float[]){0, 2, 4});
    out = tensor_var(src, 0, 0);
    ensure(isclose(tensor_data(out)[0], 8 / 3.F));

    // tensor_std: std of [0, 2, 4] = sqrt(8/3)
    out = tensor_std(src, 0, 0);
    ensure(isclose(tensor_data(out)[0], sqrtf(8 / 3.F)));

    // tensor_std along axis 1: [[0,2],[4,6]] -> std per row = [1, 1]
    src = tensor_from((int[]){2, 2}, 2, (float[]){0, 2, 4, 6});
    out = tensor_std(src, 1, 0);
    ensure(isclose(tensor_data(out)[0], 1) && isclose(tensor_data(out)[1], 1));

    // tensor_var along axis 1: [[1,3],[2,4]] -> [1, 1]
    src = tensor_from((int[]){2, 2}, 2, (float[]){1, 3, 2, 4});
    out = tensor_var(src, 1, 0);
    ensure(isclose(tensor_data(out)[0], 1) && isclose(tensor_data(out)[1], 1));

    // global reduction: [1,2,3,4] -> mean=2.5, var=1.25
    src = tensor_from((int[]){4}, 1, (float[]){1, 2, 3, 4});
    out = tensor_var(src, INT_MAX, 0);
    ensure(isclose(tensor_data(out)[0], 1.25F));

    // tensor_norm: [3, 4] -> sqrt(9 + 16) = 5
    src = tensor_from((int[]){2}, 1, (float[]){3, 4});
    out = tensor_norm(src, INT_MAX, 0);
    ensure(isclose(tensor_data(out)[0], 5));

    // tensor_norm along axis: [[1,0],[0,2]] -> [1, 2]
    src = tensor_from((int[]){2, 2}, 2, (float[]){1, 0, 0, 2});
    out = tensor_norm(src, 1, 0);
    ensure(isclose(tensor_data(out)[0], 1) && isclose(tensor_data(out)[1], 2));

    tensor_frame_end();
}

static void test_argreduction(void)
{
    tensor_frame_begin();

    // argmin / argmax along axis: [[3,1,4],[1,5,9]] -> argmin row = [1,0], argmax row = [2,2]
    Tensor *src = tensor_from((int[]){2, 3}, 2, (float[]){3, 1, 4, 1, 5, 9});
    long row[2];
    tensor_argmin(src, row, 1);
    ensure(row[0] == 1 && row[1] == 0);
    tensor_argmax(src, row, 1);
    ensure(row[0] == 2 && row[1] == 2);

    // axis 0: argmin col = [1,0,0], argmax col = [0,1,1]
    long col[3];
    tensor_argmin(src, col, 0);
    ensure(col[0] == 1 && col[1] == 0 && col[2] == 0);
    tensor_argmax(src, col, 0);
    ensure(col[0] == 0 && col[1] == 1 && col[2] == 1);

    // global reduction (axis == INT_MAX): flat [3,1,4,1,5,9], argmin=1, argmax=5
    long flat;
    tensor_argmin(src, &flat, INT_MAX);
    ensure(flat == 1);
    tensor_argmax(src, &flat, INT_MAX);
    ensure(flat == 5);

    // 1-D input: result is single index
    src = tensor_from((int[]){4}, 1, (float[]){5, 2, 8, 1});
    tensor_argmin(src, &flat, 0);
    ensure(flat == 3);
    tensor_argmax(src, &flat, 0);
    ensure(flat == 2);

    // non-contiguous input (transposed): [[1,4],[2,5],[3,6]], argmin along dim 0 -> [0,0]
    src = tensor_from((int[]){2, 3}, 2, (float[]){1, 2, 3, 4, 5, 6});
    src = tensor_transpose(src, 0, 1);  // [3, 2]
    tensor_argmin(src, row, 0);
    ensure(row[0] == 0 && row[1] == 0);
    tensor_argmax(src, row, 0);
    ensure(row[0] == 2 && row[1] == 2);

    tensor_frame_end();
}

static void test_processing(void)
{
    tensor_frame_begin();

    // matmul: general rectangular [2, 3] @ [3, 2] -> [2, 2]
    Tensor *lhs = tensor_from((int[]){2, 3}, 2, (float[]){1, 2, 3, 4, 5, 6});
    Tensor *rhs = tensor_from((int[]){3, 2}, 2, (float[]){7, 8, 9, 10, 11, 12});
    Tensor *out = tensor_matmul(lhs, rhs);
    ensure(tensor_ndim(out) == 2 && tensor_shape(out)[0] == 2 && tensor_shape(out)[1] == 2);
    ensure(tensor_data(out)[0] == 58 && tensor_data(out)[1] == 64 && tensor_data(out)[2] == 139 &&
           tensor_data(out)[3] == 154);

    // outer product: [3, 1] @ [1, 3] -> [3, 3]
    lhs = tensor_from((int[]){3, 1}, 2, (float[]){1, 2, 3});
    rhs = tensor_from((int[]){1, 3}, 2, (float[]){4, 5, 6});
    out = tensor_matmul(lhs, rhs);
    ensure(tensor_ndim(out) == 2 && tensor_shape(out)[0] == 3 && tensor_shape(out)[1] == 3);
    ensure(tensor_data(out)[0] == 4 && tensor_data(out)[1] == 5 && tensor_data(out)[2] == 6 &&
           tensor_data(out)[3] == 8 && tensor_data(out)[4] == 10 && tensor_data(out)[5] == 12 &&
           tensor_data(out)[6] == 12 && tensor_data(out)[7] == 15 && tensor_data(out)[8] == 18);

    // inner product: [1, 3] @ [3, 1] -> [1, 1]
    out = tensor_matmul(rhs, lhs);  // NOLINT(readability-suspicious-call-argument)
    ensure(tensor_ndim(out) == 2 && tensor_shape(out)[0] == 1 && tensor_shape(out)[1] == 1);
    ensure(tensor_data(out)[0] == 32);  // 4*1 + 5*2 + 6*3 = 32

    // matrix-vector: [2, 2] @ [2, 1] -> [2, 1]
    lhs = tensor_from((int[]){2, 2}, 2, (float[]){1, 2, 3, 4});
    rhs = tensor_from((int[]){2, 1}, 2, (float[]){5, 6});
    out = tensor_matmul(lhs, rhs);
    ensure(tensor_ndim(out) == 2 && tensor_shape(out)[0] == 2 && tensor_shape(out)[1] == 1);
    ensure(tensor_data(out)[0] == 17 && tensor_data(out)[1] == 39);

    // batch matmul with broadcasting: [2, 2, 2] @ [2, 2] -> [2, 2, 2]
    lhs = tensor_from((int[]){2, 2, 2}, 3, (float[]){1, 0, 0, 1, 2, 0, 0, 2});
    rhs = tensor_from((int[]){2, 2}, 2, (float[]){1, 2, 3, 4});
    out = tensor_matmul(lhs, rhs);
    ensure(tensor_ndim(out) == 3 && tensor_shape(out)[0] == 2 && tensor_shape(out)[1] == 2 &&
           tensor_shape(out)[2] == 2);
    ensure(tensor_data(out)[0] == 1 && tensor_data(out)[1] == 2 && tensor_data(out)[2] == 3 &&
           tensor_data(out)[3] == 4);
    ensure(tensor_data(out)[4] == 2 && tensor_data(out)[5] == 4 && tensor_data(out)[6] == 6 &&
           tensor_data(out)[7] == 8);

    // matmul with transposed rhs: [2, 3] @ [2, 3]^T = [2, 3] @ [3, 2] -> [2, 2]
    lhs = tensor_from((int[]){2, 3}, 2, (float[]){1, 2, 3, 4, 5, 6});
    rhs = tensor_from((int[]){2, 3}, 2, (float[]){1, 1, 1, 2, 2, 2});
    rhs = tensor_transpose(rhs, 0, 1);  // now [3, 2]
    out = tensor_matmul(lhs, rhs);
    ensure(tensor_ndim(out) == 2 && tensor_shape(out)[0] == 2 && tensor_shape(out)[1] == 2);
    ensure(tensor_data(out)[0] == 6 && tensor_data(out)[1] == 12 && tensor_data(out)[2] == 15 &&
           tensor_data(out)[3] == 30);

    // matmul with transposed lhs: [3, 2]^T @ [3, 2] = [2, 3] @ [3, 2] -> [2, 2]
    lhs = tensor_from((int[]){3, 2}, 2, (float[]){1, 2, 3, 4, 5, 6});
    lhs = tensor_transpose(lhs, 0, 1);  // now [2, 3]: [[1,3,5],[2,4,6]]
    rhs = tensor_from((int[]){3, 2}, 2, (float[]){1, 0, 0, 1, 1, 0});
    out = tensor_matmul(lhs, rhs);
    ensure(tensor_ndim(out) == 2 && tensor_shape(out)[0] == 2 && tensor_shape(out)[1] == 2);
    ensure(tensor_data(out)[0] == 6 && tensor_data(out)[1] == 3 && tensor_data(out)[2] == 8 &&
           tensor_data(out)[3] == 4);

    tensor_frame_end();
}

static void test_autograd(void)
{
    tensor_frame_begin();

    // tensor_neg: grad is negated
    Tensor *lhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){1, 2, 3}));
    Tensor *out = tensor_neg(lhs);
    tensor_backward(out, 0);
    ensure(tensor_data(tensor_grad(lhs))[0] == -1 && tensor_data(tensor_grad(lhs))[1] == -1 &&
           tensor_data(tensor_grad(lhs))[2] == -1);

    // tensor_abs: grad is sign(src)
    lhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){-2, 3, -4}));
    out = tensor_abs(lhs);
    tensor_backward(out, 0);
    ensure(tensor_data(tensor_grad(lhs))[0] == -1 && tensor_data(tensor_grad(lhs))[1] == 1 &&
           tensor_data(tensor_grad(lhs))[2] == -1);

    // tensor_square: grad is 2*src
    lhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){2, 3, 4}));
    out = tensor_square(lhs);
    tensor_backward(out, 0);
    ensure(tensor_data(tensor_grad(lhs))[0] == 4 && tensor_data(tensor_grad(lhs))[1] == 6 &&
           tensor_data(tensor_grad(lhs))[2] == 8);

    // tensor_sqrt: grad is 1/(2*out) = 1/(2*sqrt(src))
    lhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){4, 9, 16}));
    out = tensor_sqrt(lhs);
    tensor_backward(out, 0);
    ensure(isclose(tensor_data(tensor_grad(lhs))[0], 0.25F) &&
           isclose(tensor_data(tensor_grad(lhs))[1], 1 / 6.F) &&
           isclose(tensor_data(tensor_grad(lhs))[2], 0.125F));

    // tensor_rsqrt: grad is -out^3/2 = -1/(2*src^(3/2))
    lhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){4, 9, 16}));
    out = tensor_rsqrt(lhs);
    tensor_backward(out, 0);
    ensure(isclose(tensor_data(tensor_grad(lhs))[0], -0.0625F) &&    // -0.5^3/2
           isclose(tensor_data(tensor_grad(lhs))[1], -1 / 54.F) &&   // -(1/3)^3/2
           isclose(tensor_data(tensor_grad(lhs))[2], -0.0078125F));  // -0.25^3/2

    // tensor_exp: grad is out = exp(src)
    lhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){1, 2, 3}));
    out = tensor_exp(lhs);
    tensor_backward(out, 0);
    ensure(isclose(tensor_data(tensor_grad(lhs))[0], expf(1)) &&
           isclose(tensor_data(tensor_grad(lhs))[1], expf(2)) &&
           isclose(tensor_data(tensor_grad(lhs))[2], expf(3)));

    // tensor_log: grad is 1/src
    lhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){1, 2, 4}));
    out = tensor_log(lhs);
    tensor_backward(out, 0);
    ensure(isclose(tensor_data(tensor_grad(lhs))[0], 1) &&
           isclose(tensor_data(tensor_grad(lhs))[1], 0.5F) &&
           isclose(tensor_data(tensor_grad(lhs))[2], 0.25F));

    // tensor_relu: grad is 1 if src > 0, else 0
    lhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){-1, 2, 3}));
    out = tensor_relu(lhs);
    tensor_backward(out, 0);
    ensure(tensor_data(tensor_grad(lhs))[0] == 0 && tensor_data(tensor_grad(lhs))[1] == 1 &&
           tensor_data(tensor_grad(lhs))[2] == 1);

    // tensor_sigmoid: grad is out*(1-out)
    lhs = tensor_requires_grad(tensor_from((int[]){1}, 1, (float[]){0}));
    out = tensor_sigmoid(lhs);
    tensor_backward(out, 0);
    ensure(isclose(tensor_data(tensor_grad(lhs))[0], 0.25F));  // 0.5 * 0.5

    // tensor_tanh: grad is 1 - out^2
    lhs = tensor_requires_grad(tensor_from((int[]){2}, 1, (float[]){0, 1}));
    out = tensor_tanh(lhs);
    tensor_backward(out, 0);
    ensure(isclose(tensor_data(tensor_grad(lhs))[0], 1) &&  // 1 - tanh(0)^2 = 1
           isclose(tensor_data(tensor_grad(lhs))[1], 1 - (tanhf(1) * tanhf(1))));

    // tensor_add: grad flows to both inputs
    lhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){1, 2, 3}));
    Tensor *rhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){4, 5, 6}));
    out = tensor_add(lhs, rhs);
    tensor_backward(out, 0);
    ensure(tensor_data(tensor_grad(lhs))[0] == 1 && tensor_data(tensor_grad(lhs))[1] == 1 &&
           tensor_data(tensor_grad(lhs))[2] == 1);
    ensure(tensor_data(tensor_grad(rhs))[0] == 1 && tensor_data(tensor_grad(rhs))[1] == 1 &&
           tensor_data(tensor_grad(rhs))[2] == 1);

    // tensor_add: grad does not appear on input without requires_grad
    lhs = tensor_from((int[]){3}, 1, (float[]){1, 2, 3});
    rhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){4, 5, 6}));
    out = tensor_add(lhs, rhs);
    tensor_backward(out, 0);
    ensure(tensor_grad(lhs) == 0);
    ensure(tensor_data(tensor_grad(rhs))[0] == 1 && tensor_data(tensor_grad(rhs))[2] == 1);

    // tensor_add: gradient accumulation; backward on same graph twice accumulates into leaf grads
    lhs = tensor_requires_grad(tensor_from((int[]){2}, 1, (float[]){1, 2}));
    rhs = tensor_requires_grad(tensor_from((int[]){2}, 1, (float[]){3, 4}));
    out = tensor_add(lhs, rhs);
    tensor_backward(out, 0);  // lhs->grad = 1, rhs->grad = 1; out->grad = 1
    tensor_backward(out, 0);  // lhs->grad = 2, rhs->grad = 2; out->grad = 1
    ensure(tensor_data(tensor_grad(lhs))[0] == 2 && tensor_data(tensor_grad(lhs))[1] == 2);
    ensure(tensor_data(tensor_grad(rhs))[0] == 2 && tensor_data(tensor_grad(rhs))[1] == 2);

    // tensor_add: broadcasting [2, 3] + [3] -> rhs grad summed along axis 0
    lhs = tensor_requires_grad(tensor_from((int[]){2, 3}, 2, (float[]){1, 2, 3, 4, 5, 6}));
    rhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){1, 2, 3}));
    out = tensor_add(lhs, rhs);
    tensor_backward(out, 0);
    ensure(tensor_data(tensor_grad(lhs))[0] == 1 && tensor_data(tensor_grad(lhs))[5] == 1);
    ensure(tensor_data(tensor_grad(rhs))[0] == 2 && tensor_data(tensor_grad(rhs))[1] == 2 &&
           tensor_data(tensor_grad(rhs))[2] == 2);

    // tensor_sub: lhs grad is +1, rhs grad is -1
    lhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){1, 2, 3}));
    rhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){4, 5, 6}));
    out = tensor_sub(lhs, rhs);
    tensor_backward(out, 0);
    ensure(tensor_data(tensor_grad(lhs))[0] == 1 && tensor_data(tensor_grad(lhs))[2] == 1);
    ensure(tensor_data(tensor_grad(rhs))[0] == -1 && tensor_data(tensor_grad(rhs))[2] == -1);

    // tensor_mul: lhs grad is out->grad * rhs, rhs grad is out->grad * lhs
    lhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){2, 3, 4}));
    rhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){5, 6, 7}));
    out = tensor_mul(lhs, rhs);
    tensor_backward(out, 0);
    ensure(tensor_data(tensor_grad(lhs))[0] == 5 && tensor_data(tensor_grad(lhs))[1] == 6 &&
           tensor_data(tensor_grad(lhs))[2] == 7);
    ensure(tensor_data(tensor_grad(rhs))[0] == 2 && tensor_data(tensor_grad(rhs))[1] == 3 &&
           tensor_data(tensor_grad(rhs))[2] == 4);

    // tensor_div: lhs grad is grad/rhs, rhs grad is -grad*out/rhs
    lhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){4, 9, 16}));
    rhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){2, 3, 4}));
    out = tensor_div(lhs, rhs);
    tensor_backward(out, 0);
    ensure(isclose(tensor_data(tensor_grad(lhs))[0], 0.5F) &&
           isclose(tensor_data(tensor_grad(lhs))[1], 1 / 3.F) &&
           isclose(tensor_data(tensor_grad(lhs))[2], 0.25F));
    ensure(isclose(tensor_data(tensor_grad(rhs))[0], -1) &&
           isclose(tensor_data(tensor_grad(rhs))[1], -1) &&
           isclose(tensor_data(tensor_grad(rhs))[2], -1));

    // tensor_pow: base grad is exp*out/base, exp grad is out*log(base)
    lhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){2, 3, 4}));
    rhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){3, 2, 2}));
    out = tensor_pow(lhs, rhs);
    tensor_backward(out, 0);
    ensure(isclose(tensor_data(tensor_grad(lhs))[0], 12) &&  // 3*8/2
           isclose(tensor_data(tensor_grad(lhs))[1], 6) &&   // 2*9/3
           isclose(tensor_data(tensor_grad(lhs))[2], 8));    // 2*16/4
    ensure(isclose(tensor_data(tensor_grad(rhs))[0], 8 * logf(2)) &&
           isclose(tensor_data(tensor_grad(rhs))[1], 9 * logf(3)) &&
           isclose(tensor_data(tensor_grad(rhs))[2], 16 * logf(4)));

    // tensor_where: grad routed to if_true where cond != 0, to if_false otherwise
    lhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){1, 2, 3}));
    rhs = tensor_requires_grad(tensor_from((int[]){3}, 1, (float[]){4, 5, 6}));
    out = tensor_where(tensor_from((int[]){3}, 1, (float[]){1, 0, 1}), lhs, rhs);
    tensor_backward(out, 0);
    ensure(tensor_data(tensor_grad(lhs))[0] == 1 && tensor_data(tensor_grad(lhs))[1] == 0 &&
           tensor_data(tensor_grad(lhs))[2] == 1);
    ensure(tensor_data(tensor_grad(rhs))[0] == 0 && tensor_data(tensor_grad(rhs))[1] == 1 &&
           tensor_data(tensor_grad(rhs))[2] == 0);

    // tensor_clamp: grad passes through where not clamped, flows to min/max otherwise
    lhs = tensor_requires_grad(tensor_from((int[]){4}, 1, (float[]){-1, 2, 5, 8}));
    Tensor *min = tensor_requires_grad(tensor_scalar(0));
    Tensor *max = tensor_requires_grad(tensor_scalar(6));
    out = tensor_clamp(lhs, min, max);
    tensor_backward(out, 0);
    ensure(tensor_data(tensor_grad(lhs))[0] == 0 && tensor_data(tensor_grad(lhs))[1] == 1 &&
           tensor_data(tensor_grad(lhs))[2] == 1 && tensor_data(tensor_grad(lhs))[3] == 0);
    ensure(isclose(tensor_data(tensor_grad(min))[0], 1));  // one element clamped to min
    ensure(isclose(tensor_data(tensor_grad(max))[0], 1));  // one element clamped to max

    // tensor_sum: grad is 1 broadcast to src->shape; axis reduction unsqueezes before expanding
    lhs = tensor_requires_grad(tensor_from((int[]){2, 3}, 2, (float[]){1, 2, 3, 4, 5, 6}));
    out = tensor_sum(lhs, 0, 0);  // sum along axis 0: [5, 7, 9]
    tensor_backward(out, 0);
    for (int i = 0; i < 6; i++) {
        ensure(tensor_data(tensor_grad(lhs))[i] == 1);
    }

    // tensor_sum: global sum -> scalar, grad is 1 for all elements
    lhs = tensor_requires_grad(tensor_from((int[]){4}, 1, (float[]){1, 2, 3, 4}));
    out = tensor_sum(lhs, INT_MAX, 0);
    tensor_backward(out, 0);
    for (int i = 0; i < 4; i++) {
        ensure(tensor_data(tensor_grad(lhs))[i] == 1);
    }

    // tensor_mean: grad is 1/n broadcast to src->shape
    lhs = tensor_requires_grad(tensor_from((int[]){4}, 1, (float[]){1, 2, 3, 4}));
    out = tensor_mean(lhs, INT_MAX, 0);
    tensor_backward(out, 0);
    for (int i = 0; i < 4; i++) {
        ensure(isclose(tensor_data(tensor_grad(lhs))[i], 0.25F));
    }

    // tensor_var: grad is 2*(src - mean)/n
    lhs = tensor_requires_grad(tensor_from((int[]){4}, 1, (float[]){1, 2, 3, 4}));
    out = tensor_var(lhs, INT_MAX, 0);
    tensor_backward(out, 0);
    // mean=2.5, 2*(src-mean)/4: [-0.75, -0.25, 0.25, 0.75]
    ensure(isclose(tensor_data(tensor_grad(lhs))[0], -0.75F) &&
           isclose(tensor_data(tensor_grad(lhs))[1], -0.25F) &&
           isclose(tensor_data(tensor_grad(lhs))[2], 0.25F) &&
           isclose(tensor_data(tensor_grad(lhs))[3], 0.75F));

    tensor_frame_end();
}

static void test_io(void)
{
    tensor_frame_begin();

    // save and load roundtrip: shape and values preserved
    Tensor *save = tensor_from((int[]){2, 3}, 2, (float[]){1, 2, 3, 4, 5, 6});
    tensor_save(save, "/tmp/tensor.npy");

    Tensor *load = tensor_load("/tmp/tensor.npy");
    ensure(tensor_ndim(load) == 2 && tensor_shape(load)[0] == 2 && tensor_shape(load)[1] == 3);
    for (long i = 0; i < tensor_numel(load); i++) {
        ensure(tensor_data(load)[i] == tensor_data(save)[i]);
    }

    // verify the file is readable by numpy
    ensure(system("python3 -c \""
                  "import numpy as np;"
                  "a = np.load('/tmp/tensor.npy');"
                  "assert a.shape == (2, 3);"
                  "assert a.dtype == np.float32;"
                  "assert (a == [[1,2,3],[4,5,6]]).all()"
                  "\"") == 0);

    // verify we can load a file saved by numpy
    ensure(system("python3 -c \""
                  "import numpy as np;"
                  "np.save('/tmp/tensor.npy', np.array([[4,5,6],[7,8,9]], dtype=np.float32))"
                  "\"") == 0);
    load = tensor_load("/tmp/tensor.npy");
    ensure(tensor_ndim(load) == 2 && tensor_shape(load)[0] == 2 && tensor_shape(load)[1] == 3);
    ensure(tensor_data(load)[0] == 4 && tensor_data(load)[1] == 5 && tensor_data(load)[5] == 9);

    // 0D tensor (scalar)
    save = tensor_scalar(42);
    tensor_save(save, "/tmp/scalar.npy");

    load = tensor_load("/tmp/scalar.npy");
    ensure(tensor_ndim(load) == 0 && tensor_numel(load) == 1 && tensor_data(load)[0] == 42);

    ensure(system("python3 -c \""
                  "import numpy as np;"
                  "a = np.load('/tmp/scalar.npy');"
                  "assert a.shape == ();"
                  "assert a.dtype == np.float32;"
                  "assert a == 42"
                  "\"") == 0);

    tensor_frame_end();
}

int main(void)
{
    test_creation();
    test_movement();
    test_unary();
    test_binary();
    test_ternary();
    test_reduction();
    test_argreduction();
    test_processing();
    test_autograd();
    test_io();
}
