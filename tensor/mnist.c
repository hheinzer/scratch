// NOLINTBEGIN(readability-identifier-length)
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "tensor.h"

#define BATCH 128
#define LR 0.1F
#define MOMENTUM 0.9F

static Tensor *forward(Tensor *X, Tensor *Wc1, Tensor *bc1, Tensor *Wc2, Tensor *bc2, Tensor *W,
                       Tensor *b)
{
    X = tensor_reshape(X, (int[]){-1, 1, 28, 28}, 4);
    X = tensor_relu(tensor_conv2d(X, Wc1, bc1, 2, 1));
    X = tensor_relu(tensor_conv2d(X, Wc2, bc2, 2, 1));
    return tensor_add(tensor_matmul(tensor_flatten(X, 1, INT_MAX), W), b);
}

static void update(Tensor *param, Tensor *vel, float lr, float momentum)
{
    Tensor *grad = tensor_grad(param);
    if (!grad) {
        return;
    }
    float *pdata = tensor_data(param);
    float *gdata = tensor_data(grad);
    float *vdata = tensor_data(vel);
    for (long i = 0; i < tensor_numel(param); i++) {
        vdata[i] = (momentum * vdata[i]) + gdata[i];
        pdata[i] -= lr * vdata[i];
    }
    tensor_zero_grad(param);
}

static float accuracy(Tensor *X, Tensor *y, Tensor **params)
{
    tensor_frame_begin();

    tensor_no_grad_begin();
    Tensor *logit = forward(X, params[0], params[1], params[2], params[3], params[4], params[5]);
    tensor_no_grad_end();

    int n = tensor_shape(X)[0];
    int correct = 0;
    float *logit_data = tensor_data(logit);
    float *y_data = tensor_data(y);
    for (int i = 0; i < n; i++) {
        int pred = 0;
        for (int j = 1; j < 10; j++) {
            if (logit_data[((long)i * 10) + j] > logit_data[((long)i * 10) + pred]) {
                pred = j;
            }
        }
        if (pred == (int)y_data[i]) {
            correct += 1;
        }
    }
    float result = (float)correct / (float)n;

    tensor_frame_end();
    return result;
}

int main(void)
{
    tensor_frame_begin();

    Tensor *X_train = tensor_load("data/X_train.npy");
    Tensor *y_train = tensor_load("data/y_train.npy");
    Tensor *X_test = tensor_load("data/X_test.npy");
    Tensor *y_test = tensor_load("data/y_test.npy");

    srand((unsigned)time(0));
    Tensor *Wc1 = tensor_requires_grad(tensor_normal((int[]){16, 1, 3, 3}, 4, 0, sqrtf(2 / 9.F)));
    Tensor *bc1 = tensor_requires_grad(tensor_zeros((int[]){16}, 1));
    Tensor *Wc2 =
        tensor_requires_grad(tensor_normal((int[]){32, 16, 3, 3}, 4, 0, sqrtf(2 / 144.F)));
    Tensor *bc2 = tensor_requires_grad(tensor_zeros((int[]){32}, 1));
    Tensor *W = tensor_requires_grad(tensor_normal((int[]){1568, 10}, 2, 0, sqrtf(1 / 1568.F)));
    Tensor *b = tensor_requires_grad(tensor_zeros((int[]){1, 10}, 2));

    Tensor *params[] = {Wc1, bc1, Wc2, bc2, W, b};
    Tensor *vels[] = {
        tensor_zeros(tensor_shape(Wc1), tensor_ndim(Wc1)),
        tensor_zeros(tensor_shape(bc1), tensor_ndim(bc1)),
        tensor_zeros(tensor_shape(Wc2), tensor_ndim(Wc2)),
        tensor_zeros(tensor_shape(bc2), tensor_ndim(bc2)),
        tensor_zeros(tensor_shape(W), tensor_ndim(W)),
        tensor_zeros(tensor_shape(b), tensor_ndim(b)),
    };

    int n_train = tensor_shape(X_train)[0];
    for (int epoch = 0; epoch < 10; epoch++) {
        tensor_shuffle((Tensor *[]){X_train, y_train}, 2, 0);

        int n_batches = 0;
        float total_loss = 0;
        for (int i = 0; i < n_train; i += BATCH) {
            tensor_frame_begin();

            int batch = (i + BATCH <= n_train) ? BATCH : (n_train - i);
            Tensor *X_b = tensor_slice(X_train, 0, i, i + batch, 1);
            Tensor *y_b = tensor_slice(y_train, 0, i, i + batch, 1);
            Tensor *loss = tensor_cross_entropy(forward(X_b, Wc1, bc1, Wc2, bc2, W, b), y_b);
            tensor_backward(loss, 0);

            tensor_no_grad_begin();
            for (int k = 0; k < 6; k++) {
                update(params[k], vels[k], LR, MOMENTUM);
            }
            tensor_no_grad_end();

            n_batches += 1;
            total_loss += tensor_data(loss)[0];

            tensor_frame_end();
        }

        printf("Epoch %2d  Loss: %.4f  Accuracy: %.1f%%\n", epoch + 1,
               total_loss / (float)n_batches, 100 * accuracy(X_test, y_test, params));
    }

    tensor_frame_end();
}
// NOLINTEND(readability-identifier-length)
