#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"

static const float m_pi = 3.14159265358979323846F;
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

    // tensor_scalar
    tensor = tensor_scalar(7);
    ensure(tensor_ndim(tensor) == 0 && tensor_data(tensor)[0] == 7);

    // tensor_range: [1, 3] -> [1, 2, 3]
    tensor = tensor_range(1, 3, 1);
    ensure(tensor_numel(tensor) == 3);
    ensure(tensor_data(tensor)[0] == 1 && tensor_data(tensor)[1] == 2 &&
           tensor_data(tensor)[2] == 3);

    // tensor_arange: [1, 4) -> [1, 2, 3]
    tensor = tensor_arange(1, 4, 1);
    ensure(tensor_numel(tensor) == 3);
    ensure(tensor_data(tensor)[0] == 1 && tensor_data(tensor)[1] == 2 &&
           tensor_data(tensor)[2] == 3);

    // tensor_arange: negative step (3, 0, -1) -> [3, 2, 1]
    tensor = tensor_arange(3, 0, -1);
    ensure(tensor_numel(tensor) == 3);
    ensure(tensor_data(tensor)[0] == 3 && tensor_data(tensor)[1] == 2 &&
           tensor_data(tensor)[2] == 1);

    // tensor_linspace: 0 to 1, 3 steps -> [0, 0.5, 1]
    tensor = tensor_linspace(0, 1, 3);
    ensure(tensor_numel(tensor) == 3);
    ensure(tensor_data(tensor)[0] == 0 && isclose(tensor_data(tensor)[1], 0.5F) &&
           tensor_data(tensor)[2] == 1);

    // tensor_logspace: base 10, 0 to 2, 3 steps -> [1, 10, 100]
    tensor = tensor_logspace(10, 0, 2, 3);
    ensure(tensor_numel(tensor) == 3);
    ensure(isclose(tensor_data(tensor)[0], 1) && isclose(tensor_data(tensor)[1], 10) &&
           isclose(tensor_data(tensor)[2], 100));

    // tensor_eye: shape [3, 2], diagonal is 1
    tensor = tensor_eye(3, 2);
    ensure(tensor_shape(tensor)[0] == 3 && tensor_shape(tensor)[1] == 2);
    ensure(tensor_data(tensor)[0] == 1 && tensor_data(tensor)[1] == 0);  // row 0
    ensure(tensor_data(tensor)[2] == 0 && tensor_data(tensor)[3] == 1);  // row 1
    ensure(tensor_data(tensor)[4] == 0 && tensor_data(tensor)[5] == 0);  // row 2

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
    Tensor *tensor = tensor_arange(1, 7, 1);
    tensor = tensor_reshape(tensor, (int[]){2, 3}, 2);
    ensure(tensor_ndim(tensor) == 2 && tensor_shape(tensor)[0] == 2 &&
           tensor_shape(tensor)[1] == 3);
    ensure(tensor_data(tensor)[0] == 1 && tensor_data(tensor)[5] == 6);

    // tensor_reshape: infer dim with -1, [6] -> [2, 3]
    tensor = tensor_arange(1, 7, 1);
    tensor = tensor_reshape(tensor, (int[]){-1, 3}, 2);
    ensure(tensor_shape(tensor)[0] == 2 && tensor_shape(tensor)[1] == 3);

    // tensor_flatten: dims 0..1 of [2, 3] -> [6]
    tensor = tensor_arange(1, 7, 1);
    tensor = tensor_reshape(tensor, (int[]){2, 3}, 2);
    tensor = tensor_flatten(tensor, 0, 1);
    ensure(tensor_ndim(tensor) == 1 && tensor_numel(tensor) == 6);

    // tensor_flatten: all dims (INT_MIN to INT_MAX) of [2, 3] -> [6]
    tensor = tensor_arange(1, 7, 1);
    tensor = tensor_reshape(tensor, (int[]){2, 3}, 2);
    tensor = tensor_flatten(tensor, INT_MIN, INT_MAX);
    ensure(tensor_ndim(tensor) == 1 && tensor_numel(tensor) == 6);

    // tensor_unflatten: [6] -> [2, 3]
    tensor = tensor_arange(1, 7, 1);
    tensor = tensor_unflatten(tensor, 0, (int[]){2, 3}, 2);
    ensure(tensor_ndim(tensor) == 2 && tensor_shape(tensor)[0] == 2 &&
           tensor_shape(tensor)[1] == 3);

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

    // tensor_squeeze_all: [1, 3, 1] -> [3]
    tensor = tensor_zeros((int[]){1, 3, 1}, 3);
    tensor = tensor_squeeze(tensor, INT_MAX);
    ensure(tensor_ndim(tensor) == 1 && tensor_shape(tensor)[0] == 3);

    // tensor_unsqueeze: [3] insert at 0 -> [1, 3]
    tensor = tensor_zeros((int[]){3}, 1);
    tensor = tensor_unsqueeze(tensor, 0);
    ensure(tensor_ndim(tensor) == 2 && tensor_shape(tensor)[0] == 1 &&
           tensor_shape(tensor)[1] == 3);

    // tensor_permute: [2, 3] -> [3, 2], check strides swapped
    tensor = tensor_arange(1, 7, 1);
    tensor = tensor_reshape(tensor, (int[]){2, 3}, 2);
    tensor = tensor_permute(tensor, (int[]){1, 0});
    ensure(tensor_shape(tensor)[0] == 3 && tensor_shape(tensor)[1] == 2);
    ensure(tensor_stride(tensor)[0] == 1 && tensor_stride(tensor)[1] == 3);

    // tensor_transpose: same as permute for 2D
    tensor = tensor_arange(1, 7, 1);
    tensor = tensor_reshape(tensor, (int[]){2, 3}, 2);
    tensor = tensor_transpose(tensor, 0, 1);
    ensure(tensor_shape(tensor)[0] == 3 && tensor_shape(tensor)[1] == 2);
    ensure(tensor_stride(tensor)[0] == 1 && tensor_stride(tensor)[1] == 3);
    // element [1][0]: data[1*1 + 0*3] = data[1] = 2
    ensure(tensor_data(tensor)[(tensor_stride(tensor)[0] * 1) + (tensor_stride(tensor)[1] * 0)] ==
           2);

    // tensor_flip along axis 0: [1,2,3] -> [3,2,1]
    tensor = tensor_from((int[]){3}, 1, (float[]){1, 2, 3});
    tensor = tensor_contiguous(tensor_flip(tensor, 0));
    ensure(tensor_data(tensor)[0] == 3 && tensor_data(tensor)[1] == 2 &&
           tensor_data(tensor)[2] == 1);

    // tensor_flip along axis 1: [[1,2],[3,4]] -> [[2,1],[4,3]]
    tensor = tensor_from((int[]){2, 2}, 2, (float[]){1, 2, 3, 4});
    tensor = tensor_contiguous(tensor_flip(tensor, 1));
    ensure(tensor_data(tensor)[0] == 2 && tensor_data(tensor)[1] == 1 &&
           tensor_data(tensor)[2] == 4 && tensor_data(tensor)[3] == 3);

    // tensor_slice: [0,1,2,3,4,5] slice [1:4] -> [1,2,3]
    tensor = tensor_arange(0, 6, 1);
    tensor = tensor_slice(tensor, 0, 1, 4, 1);
    ensure(tensor_numel(tensor) == 3);
    ensure(tensor_data(tensor)[0] == 1 && tensor_data(tensor)[1] == 2 &&
           tensor_data(tensor)[2] == 3);

    // tensor_slice: reverse step, stride is -1, access via stride
    tensor = tensor_arange(0, 6, 1);
    tensor = tensor_slice(tensor, 0, INT_MIN, INT_MAX, -1);
    ensure(tensor_numel(tensor) == 6 && tensor_stride(tensor)[0] == -1);
    ensure(tensor_data(tensor)[0] == 5 && tensor_data(tensor)[5 * tensor_stride(tensor)[0]] == 0);

    // tensor_slice: negative start index
    tensor = tensor_arange(0, 6, 1);
    tensor = tensor_slice(tensor, 0, -4, -2, 1);  // beg=-4->2, end=-2->4 -> [2, 3]
    ensure(tensor_numel(tensor) == 2 && tensor_data(tensor)[0] == 2 && tensor_data(tensor)[1] == 3);

    // tensor_select: row 1 of [2, 3] -> [4, 5, 6]
    tensor = tensor_arange(1, 7, 1);
    tensor = tensor_reshape(tensor, (int[]){2, 3}, 2);
    tensor = tensor_select(tensor, 0, 1);
    ensure(tensor_ndim(tensor) == 1 && tensor_numel(tensor) == 3);
    ensure(tensor_data(tensor)[0] == 4 && tensor_data(tensor)[1] == 5 &&
           tensor_data(tensor)[2] == 6);

    // tensor_select: negative index
    tensor = tensor_arange(1, 7, 1);
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

    // tensor_cat: [[1,2,3], [4,5,6]] along dim 0 -> [1..6]
    Tensor *lhs = tensor_arange(1, 4, 1);
    Tensor *rhs = tensor_arange(4, 7, 1);
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
    lhs = tensor_arange(1, 4, 1);
    rhs = tensor_arange(4, 7, 1);
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

    // tensor_sin: sin(0)=0, sin(pi/2)=1
    src = tensor_from((int[]){2}, 1, (float[]){0, m_pi / 2});
    out = tensor_sin(src);
    ensure(tensor_data(out)[0] == 0 && isclose(tensor_data(out)[1], 1));

    // tensor_cos: cos(0)=1, cos(pi/2)~=0
    src = tensor_from((int[]){2}, 1, (float[]){0, m_pi / 2});
    out = tensor_cos(src);
    ensure(isclose(tensor_data(out)[0], 1) && isclose(tensor_data(out)[1], 0));

    // tensor_tan: tan(0)=0, tan(pi/4)=1
    src = tensor_from((int[]){2}, 1, (float[]){0, m_pi / 4});
    out = tensor_tan(src);
    ensure(tensor_data(out)[0] == 0 && isclose(tensor_data(out)[1], 1));

    // tensor_log: log(1)=0, log(e)~=1
    src = tensor_from((int[]){2}, 1, (float[]){1, m_e});
    out = tensor_log(src);
    ensure(tensor_data(out)[0] == logf(1) && isclose(tensor_data(out)[1], 1));

    // tensor_floor: [-1.7, 1.2, 2.9] -> [-2, 1, 2]
    src = tensor_from((int[]){3}, 1, (float[]){-1.7F, 1.2F, 2.9F});
    out = tensor_floor(src);
    ensure(tensor_data(out)[0] == -2 && tensor_data(out)[1] == 1 && tensor_data(out)[2] == 2);

    // tensor_ceil: [-1.7, 1.2, 2.9] -> [-1, 2, 3]
    src = tensor_from((int[]){3}, 1, (float[]){-1.7F, 1.2F, 2.9F});
    out = tensor_ceil(src);
    ensure(tensor_data(out)[0] == -1 && tensor_data(out)[1] == 2 && tensor_data(out)[2] == 3);

    // tensor_round: [-1.7, 0.2, 1.5] -> [-2, 0, 2] (half away from zero)
    src = tensor_from((int[]){3}, 1, (float[]){-1.7F, 0.2F, 1.5F});
    out = tensor_round(src);
    ensure(tensor_data(out)[0] == -2 && tensor_data(out)[1] == 0 && tensor_data(out)[2] == 2);

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

    // tensor_logical_not: 0->1, nonzero->0
    src = tensor_from((int[]){3}, 1, (float[]){0, 1, -5});
    out = tensor_logical_not(src);
    ensure(tensor_data(out)[0] == 1 && tensor_data(out)[1] == 0 && tensor_data(out)[2] == 0);

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

    // tensor_mod: Python convention - sign follows divisor
    lhs = tensor_from((int[]){4}, 1, (float[]){7, -7, 7, -7});
    rhs = tensor_from((int[]){4}, 1, (float[]){3, 3, -3, -3});
    out = tensor_mod(lhs, rhs);
    ensure(isclose(tensor_data(out)[0], 1) && isclose(tensor_data(out)[1], 2));  // 7%3=1, -7%3=2
    ensure(isclose(tensor_data(out)[2], -2) &&
           isclose(tensor_data(out)[3], -1));  // 7%-3=-2, -7%-3=-1

    // tensor_pow
    lhs = tensor_from((int[]){3}, 1, (float[]){2, 3, 4});
    rhs = tensor_from((int[]){3}, 1, (float[]){2, 2, 2});
    out = tensor_pow(lhs, rhs);
    ensure(tensor_data(out)[0] == 4 && tensor_data(out)[1] == 9 && tensor_data(out)[2] == 16);

    // tensor_eq / tensor_ne
    lhs = tensor_from((int[]){3}, 1, (float[]){1, 2, 3});
    rhs = tensor_from((int[]){3}, 1, (float[]){1, 0, 3});
    out = tensor_eq(lhs, rhs);
    ensure(tensor_data(out)[0] == 1 && tensor_data(out)[1] == 0 && tensor_data(out)[2] == 1);
    out = tensor_ne(lhs, rhs);
    ensure(tensor_data(out)[0] == 0 && tensor_data(out)[1] == 1 && tensor_data(out)[2] == 0);

    // tensor_lt / tensor_le / tensor_gt / tensor_ge
    out = tensor_lt(lhs, rhs);
    ensure(tensor_data(out)[0] == 0 && tensor_data(out)[1] == 0 && tensor_data(out)[2] == 0);
    out = tensor_le(lhs, rhs);
    ensure(tensor_data(out)[0] == 1 && tensor_data(out)[1] == 0 && tensor_data(out)[2] == 1);
    out = tensor_gt(lhs, rhs);
    ensure(tensor_data(out)[0] == 0 && tensor_data(out)[1] == 1 && tensor_data(out)[2] == 0);
    out = tensor_ge(lhs, rhs);
    ensure(tensor_data(out)[0] == 1 && tensor_data(out)[1] == 1 && tensor_data(out)[2] == 1);

    // tensor_logical_and / tensor_logical_or / tensor_logical_xor
    lhs = tensor_from((int[]){3}, 1, (float[]){1, 0, 1});
    rhs = tensor_from((int[]){3}, 1, (float[]){1, 1, 0});
    out = tensor_logical_and(lhs, rhs);
    ensure(tensor_data(out)[0] == 1 && tensor_data(out)[1] == 0 && tensor_data(out)[2] == 0);
    out = tensor_logical_or(lhs, rhs);
    ensure(tensor_data(out)[0] == 1 && tensor_data(out)[1] == 1 && tensor_data(out)[2] == 1);
    out = tensor_logical_xor(lhs, rhs);
    ensure(tensor_data(out)[0] == 0 && tensor_data(out)[1] == 1 && tensor_data(out)[2] == 1);

    // tensor_minimum / tensor_maximum
    lhs = tensor_from((int[]){3}, 1, (float[]){1, 5, 3});
    rhs = tensor_from((int[]){3}, 1, (float[]){4, 2, 3});
    out = tensor_minimum(lhs, rhs);
    ensure(tensor_data(out)[0] == 1 && tensor_data(out)[1] == 2 && tensor_data(out)[2] == 3);
    out = tensor_maximum(lhs, rhs);
    ensure(tensor_data(out)[0] == 4 && tensor_data(out)[1] == 5 && tensor_data(out)[2] == 3);

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

    // tensor_lerp: start=[0,0], stop=[10,20], weight=[0.5,0.25] -> [5,5]
    lhs = tensor_from((int[]){2}, 1, (float[]){0, 0});
    rhs = tensor_from((int[]){2}, 1, (float[]){10, 20});
    cond = tensor_from((int[]){2}, 1, (float[]){0.5F, 0.25F});
    out = tensor_lerp(lhs, rhs, cond);
    ensure(tensor_data(out)[0] == 5 && tensor_data(out)[1] == 5);

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

    // tensor_prod along axis 1: [[1,2,3],[4,5,6]] -> [6, 120]
    out = tensor_prod(src, 1, 0);
    ensure(tensor_data(out)[0] == 6 && tensor_data(out)[1] == 120);

    // tensor_all: [1, 1, 1] -> 1; [1, 0, 1] -> 0
    src = tensor_from((int[]){3}, 1, (float[]){1, 1, 1});
    out = tensor_all(src, INT_MAX, 0);
    ensure(tensor_data(out)[0] == 1);
    src = tensor_from((int[]){3}, 1, (float[]){1, 0, 1});
    out = tensor_all(src, INT_MAX, 0);
    ensure(tensor_data(out)[0] == 0);

    // tensor_all along axis: [[1,1],[1,0]] -> [1, 0]
    src = tensor_from((int[]){2, 2}, 2, (float[]){1, 1, 1, 0});
    out = tensor_all(src, 1, 0);
    ensure(tensor_data(out)[0] == 1 && tensor_data(out)[1] == 0);

    // tensor_any: [0, 0, 0] -> 0; [0, 1, 0] -> 1
    src = tensor_from((int[]){3}, 1, (float[]){0, 0, 0});
    out = tensor_any(src, INT_MAX, 0);
    ensure(tensor_data(out)[0] == 0);
    src = tensor_from((int[]){3}, 1, (float[]){0, 1, 0});
    out = tensor_any(src, INT_MAX, 0);
    ensure(tensor_data(out)[0] == 1);

    // tensor_any along axis: [[0,0],[0,1]] -> [0, 1]
    src = tensor_from((int[]){2, 2}, 2, (float[]){0, 0, 0, 1});
    out = tensor_any(src, 1, 0);
    ensure(tensor_data(out)[0] == 0 && tensor_data(out)[1] == 1);

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
    out = tensor_prod(src, INT_MAX, 0);
    ensure(isclose(tensor_data(out)[0], 720));

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
    ensure(isclose(tensor_data(out)[0], 8 / 3.0F));

    // tensor_std: global std = sqrt(var)
    out = tensor_std(src, 0, 0);
    ensure(isclose(tensor_data(out)[0], sqrtf(8 / 3.0F)));

    // tensor_std along axis 1: [[0,2],[4,6]] -> std per row = [1, 1]
    src = tensor_from((int[]){2, 2}, 2, (float[]){0, 2, 4, 6});
    out = tensor_std(src, 1, 0);
    ensure(isclose(tensor_data(out)[0], 1) && isclose(tensor_data(out)[1], 1));

    // var along axis of 2D: [[1,3],[2,4]] -> row vars = [1, 1]
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

    // 1-D input reduced along its only axis -> scalar output (ndim=0 path)
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

    // softmax: same shape, values sum to 1, correct per-element values
    Tensor *src = tensor_from((int[]){3}, 1, (float[]){1, 2, 3});
    Tensor *out = tensor_softmax(src, 0);
    ensure(tensor_ndim(out) == 1 && tensor_shape(out)[0] == 3 &&
           tensor_data(tensor_sum(out, 0, 0))[0] == 1);
    ensure(isclose(tensor_data(out)[0], 0.090030573F) &&
           isclose(tensor_data(out)[1], 0.24472848F) && isclose(tensor_data(out)[2], 0.66524094F));

    // log_softmax: same shape and correct per-element values
    out = tensor_log_softmax(src, 0);
    ensure(tensor_ndim(out) == 1 && tensor_shape(out)[0] == 3);
    ensure(isclose(tensor_data(out)[0], -2.4076059F) && isclose(tensor_data(out)[1], -1.4076059F) &&
           isclose(tensor_data(out)[2], -0.40760595F));

    // cross_entropy: intputs [[1,2,3],[1,2,3]], targets [2,0] -> mean(-lsm[0][2], -lsm[1][0])
    Tensor *input = tensor_from((int[]){2, 3}, 2, (float[]){1, 2, 3, 1, 2, 3});
    Tensor *target = tensor_from((int[]){2}, 1, (float[]){2, 0});
    out = tensor_cross_entropy(input, target);
    ensure(tensor_ndim(out) == 0 && isclose(tensor_data(out)[0], 1.4076059F));

    // softmax: 2D along axis 0
    src = tensor_from((int[]){2, 2}, 2, (float[]){1, 2, 3, 4});
    out = tensor_softmax(src, 0);
    ensure(tensor_ndim(out) == 2 && tensor_shape(out)[0] == 2 && tensor_shape(out)[1] == 2);
    ensure(isclose(tensor_data(out)[0], 0.11920292F) && isclose(tensor_data(out)[2], 0.88079708F));

    // tensor_dot: [1,2,3] · [4,5,6] = 32
    Tensor *lhs = tensor_from((int[]){3}, 1, (float[]){1, 2, 3});
    Tensor *rhs = tensor_from((int[]){3}, 1, (float[]){4, 5, 6});
    out = tensor_dot(lhs, rhs);
    ensure(tensor_ndim(out) == 0 && tensor_data(out)[0] == 32);

    // matmul: general rectangular [2, 3] @ [3, 2] -> [2, 2]
    lhs = tensor_from((int[]){2, 3}, 2, (float[]){1, 2, 3, 4, 5, 6});
    rhs = tensor_from((int[]){3, 2}, 2, (float[]){7, 8, 9, 10, 11, 12});
    out = tensor_matmul(lhs, rhs);
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
    test_io();
}
