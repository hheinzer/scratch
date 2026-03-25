#pragma once

// context: tensor usage should be wrapped in frames to prevent memory leaks

void tensor_frame_begin(void);
void tensor_frame_end(void);
void tensor_no_grad_begin(void);
void tensor_no_grad_end(void);

// tensor

typedef struct tensor Tensor;

int tensor_ndim(const Tensor *self);
long tensor_numel(const Tensor *self);
const int *tensor_shape(const Tensor *self);
const long *tensor_stride(const Tensor *self);
float *tensor_data(const Tensor *self);
Tensor *tensor_requires_grad(Tensor *self);
Tensor *tensor_grad(const Tensor *self);  // returns 0 if no grad

// creation

Tensor *tensor_empty(const int *shape, int ndim);
Tensor *tensor_zeros(const int *shape, int ndim);
Tensor *tensor_ones(const int *shape, int ndim);
Tensor *tensor_fill(const int *shape, int ndim, float value);
Tensor *tensor_from(const int *shape, int ndim, const float *data);  // copies data
Tensor *tensor_wrap(const int *shape, int ndim, float *data);        // view, no copy
Tensor *tensor_scalar(float value);
Tensor *tensor_rand(const int *shape, int ndim);
Tensor *tensor_uniform(const int *shape, int ndim, float low, float high);
Tensor *tensor_randn(const int *shape, int ndim);
Tensor *tensor_normal(const int *shape, int ndim, float mean, float std);

// movement: return views where possible

Tensor *tensor_reshape(const Tensor *src, const int *shape, int ndim);  // use -1 to infer one dim
Tensor *tensor_flatten(const Tensor *src, int beg_dim, int end_dim);  // INT_MIN/INT_MAX flatten all
Tensor *tensor_squeeze(const Tensor *src, int dim);  // INT_MAX removes all dims of size 1
Tensor *tensor_unsqueeze(const Tensor *src, int dim);
Tensor *tensor_permute(const Tensor *src, const int *order);
Tensor *tensor_transpose(const Tensor *src, int dim0, int dim1);
Tensor *tensor_slice(const Tensor *src, int dim, int beg, int end, int step);  // supports negatives
Tensor *tensor_select(const Tensor *src, int dim, int index);
Tensor *tensor_expand(const Tensor *src, const int *shape, int ndim);  // -1 preserves dim
Tensor *tensor_cat(const Tensor **src, int num, int dim);
Tensor *tensor_stack(const Tensor **src, int num, int dim);
Tensor *tensor_contiguous(const Tensor *src);  // view if already contiguous, copy otherwise
Tensor *tensor_clone(const Tensor *src);
Tensor *tensor_detach(const Tensor *src);

// unary

Tensor *tensor_neg(const Tensor *src);
Tensor *tensor_abs(const Tensor *src);
Tensor *tensor_sign(const Tensor *src);
Tensor *tensor_square(const Tensor *src);
Tensor *tensor_sqrt(const Tensor *src);
Tensor *tensor_rsqrt(const Tensor *src);  // 1/sqrt(x)
Tensor *tensor_exp(const Tensor *src);
Tensor *tensor_log(const Tensor *src);
Tensor *tensor_relu(const Tensor *src);
Tensor *tensor_sigmoid(const Tensor *src);
Tensor *tensor_tanh(const Tensor *src);

// binary: all functions support broadcasting

Tensor *tensor_add(const Tensor *lhs, const Tensor *rhs);
Tensor *tensor_sub(const Tensor *lhs, const Tensor *rhs);
Tensor *tensor_mul(const Tensor *lhs, const Tensor *rhs);
Tensor *tensor_div(const Tensor *lhs, const Tensor *rhs);
Tensor *tensor_pow(const Tensor *lhs, const Tensor *rhs);

// ternary: all functions support broadcasting

Tensor *tensor_where(const Tensor *cond, const Tensor *if_true, const Tensor *if_false);
Tensor *tensor_clamp(const Tensor *src, const Tensor *min, const Tensor *max);

// reduction: axis=INT_MAX reduces all dims to a scalar; keepdim retains the axis as size 1

Tensor *tensor_min(const Tensor *src, int axis, int keepdim);
Tensor *tensor_max(const Tensor *src, int axis, int keepdim);
Tensor *tensor_sum(const Tensor *src, int axis, int keepdim);
Tensor *tensor_mean(const Tensor *src, int axis, int keepdim);
Tensor *tensor_var(const Tensor *src, int axis, int keepdim);   // population variance
Tensor *tensor_std(const Tensor *src, int axis, int keepdim);   // population std
Tensor *tensor_norm(const Tensor *src, int axis, int keepdim);  // L2 norm

// argreduction: writes into caller-provided array; axis=INT_MAX gives index into flat array

void tensor_argmin(const Tensor *src, long *index, int axis);
void tensor_argmax(const Tensor *src, long *index, int axis);

// processing

Tensor *tensor_matmul(const Tensor *lhs, const Tensor *rhs);  // supports batched with broadcasting

// autograd

void tensor_backward(Tensor *self, const Tensor *grad);  // pass 0 for grad when loss is scalar
void tensor_zero_grad(Tensor *self);

// i/o: tensor_save and tensor_load use numpy NPY format

void tensor_print(const Tensor *self);
void tensor_print_backward(const Tensor *self);
void tensor_save(const Tensor *self, const char *fname);
Tensor *tensor_load(const char *fname);
