# tensor

A float32 tensor library written in C99 with broadcasting, autograd, and NumPy-compatible I/O.
Intended for small neural network experiments. See `xor.c` and `mnist.c` for worked examples.

## Usage

Compile `tensor.c` alongside your project and include `tensor.h`. See `Makefile` for recommended
compilation flags.

## Memory model

All tensors are allocated from an internal stack allocator. Tensors are not freed individually;
instead, code is wrapped in frames and all memory allocated within a frame is reclaimed when the
frame ends.

```c
tensor_frame_begin();
Tensor *x = tensor_zeros((int[]){3, 4}, 2);
// ...
tensor_frame_end();  // x and all intermediates are freed here
```

Tensors that need to outlive a frame, such as model weights, must be created before the frame
begins. The frame only reclaims tensors allocated inside it.

## Autograd

Mark a tensor as a leaf variable with `tensor_requires_grad`. After a forward pass, call
`tensor_backward` on the loss to accumulate gradients into all leaf variables that contributed to
it. Gradients are accessed with `tensor_grad`. Call `tensor_zero_grad` before each backward pass to
clear accumulated gradients.

```c
Tensor *w = tensor_requires_grad(tensor_randn((int[]){4, 4}, 2));
// ... forward pass ...
tensor_backward(loss, 0);  // pass 0 when loss is a scalar
tensor_grad(w);            // gradient for w
tensor_zero_grad(w);
```

Wrap inference code in `tensor_no_grad_begin` / `tensor_no_grad_end` to skip building the
computation graph.

## Operations

All operations (except `tensor_shuffle`) allocate new tensors inside the current frame; inputs are
never modified in place. Binary and ternary operations and reductions support broadcasting.
Reductions accept an `axis` argument; `INT_MAX` reduces over all dimensions. `keepdim=1` retains the
reduced axis as a size-1 dimension.

**Creation** `tensor_from` copies data; `tensor_wrap` creates a view over an existing pointer
without copying; `tensor_scalar` wraps a single float. `tensor_fill` fills with a constant. Random
tensors: `tensor_rand` (uniform [0, 1)), `tensor_uniform` (uniform [low, high)), `tensor_randn`
(standard normal), `tensor_normal` (normal with given mean and std).

**Movement** `tensor_reshape`, `tensor_flatten`, `tensor_squeeze`, `tensor_unsqueeze`,
`tensor_transpose`, `tensor_permute`, `tensor_slice`, `tensor_select`, `tensor_expand`,
`tensor_cat`, `tensor_stack`. Most return views; a copy is made only when the layout requires it.
`tensor_clone` always returns a contiguous copy. `tensor_detach` returns a copy with autograd
disabled. `tensor_contiguous` returns a view if the tensor is already contiguous, otherwise a copy.

**Unary** `tensor_neg`, `tensor_abs`, `tensor_sign`, `tensor_square`, `tensor_sqrt`, `tensor_rsqrt`,
`tensor_exp`, `tensor_log`, `tensor_relu`, `tensor_sigmoid`, `tensor_tanh`.

**Binary** `tensor_add`, `tensor_sub`, `tensor_mul`, `tensor_div`, `tensor_pow`.

**Ternary** `tensor_where`, `tensor_clamp`.

**Reduction** `tensor_min`, `tensor_max`, `tensor_sum`, `tensor_mean`, `tensor_var`, `tensor_std`,
`tensor_norm`. Argmin/argmax write into a caller-allocated array of indices.

**`tensor_matmul`** Matrix multiply. Supports batched matmul with broadcasting over leading
dimensions, and handles transposed inputs without copying.

**Loss** `tensor_mse` (mean squared error), `tensor_cross_entropy` (softmax cross-entropy; expects
raw logits and integer class labels).

**`tensor_shuffle`** In-place operation. Apply the same random permutation to a set of tensors along
a given axis. Useful for shuffling paired arrays (e.g. inputs and labels) in lockstep.

**I/O** `tensor_print` prints shape and data to stdout. `tensor_save` and `tensor_load` use the
NumPy NPY format, making it straightforward to exchange tensors with Python.

## Implementation notes

- The stack allocator is a bump pointer with save/restore support; frame boundaries are implemented
  as save/restore pairs
- Autograd uses a topological sort of the computation graph; backward functions are stored as
  function pointers on each node
- Views share the underlying data pointer; strides are used throughout to support non-contiguous
  layouts
