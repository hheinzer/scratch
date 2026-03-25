// NOLINTBEGIN(readability-identifier-length)
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "tensor.h"

static Tensor *forward(Tensor *X, Tensor *W1, Tensor *b1, Tensor *W2, Tensor *b2)
{
    Tensor *a1 = tensor_tanh(tensor_add(tensor_matmul(X, W1), b1));
    return tensor_sigmoid(tensor_add(tensor_matmul(a1, W2), b2));
}

static void update(Tensor *param, float lr)
{
    Tensor *grad = tensor_grad(param);
    if (!grad) {
        return;
    }
    float *pdata = tensor_data(param);
    float *gdata = tensor_data(grad);
    long numel = tensor_numel(param);
    for (long i = 0; i < numel; i++) {
        pdata[i] -= lr * gdata[i];
    }
    tensor_zero_grad(param);
}

int main(void)
{
    tensor_frame_begin();

    Tensor *X = tensor_from((int[]){4, 2}, 2, (float[]){0, 0, 0, 1, 1, 0, 1, 1});
    Tensor *y = tensor_from((int[]){4, 1}, 2, (float[]){0, 1, 1, 0});

    srand((unsigned)time(0));
    Tensor *W1 = tensor_requires_grad(tensor_randn((int[]){2, 4}, 2));
    Tensor *b1 = tensor_requires_grad(tensor_zeros((int[]){1, 4}, 2));
    Tensor *W2 = tensor_requires_grad(tensor_randn((int[]){4, 1}, 2));
    Tensor *b2 = tensor_requires_grad(tensor_zeros((int[]){1, 1}, 2));

    Tensor *param[] = {W1, b1, W2, b2};
    float lr = 0.1F;

    for (int epoch = 0; epoch < 5000; epoch++) {
        tensor_frame_begin();

        Tensor *pred = forward(X, W1, b1, W2, b2);
        Tensor *loss = tensor_mean(tensor_square(tensor_sub(pred, y)), INT_MAX, 0);
        tensor_backward(loss, 0);

        tensor_no_grad_begin();
        for (int i = 0; i < 4; i++) {
            update(param[i], lr);
        }
        tensor_no_grad_end();

        if (epoch == 0) {
            tensor_print_backward(loss);
            printf("\n");
        }
        if ((epoch + 1) % 1000 == 0) {
            printf("Epoch %4d  Loss: %.4f\n", epoch + 1, tensor_data(loss)[0]);
        }

        tensor_frame_end();
    }

    tensor_no_grad_begin();
    Tensor *pred = forward(X, W1, b1, W2, b2);
    tensor_no_grad_end();

    printf("\nResults:\n");
    float *x = tensor_data(X);
    float *p = tensor_data(pred);
    for (int i = 0; i < 4; i++) {
        printf("  %d XOR %d = %.4f\n", (int)x[(i * 2) + 0], (int)x[(i * 2) + 1], p[i]);
    }

    tensor_frame_end();
}
// NOLINTEND(readability-identifier-length)
