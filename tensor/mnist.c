// NOLINTBEGIN(readability-identifier-length)
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tensor.h"

#define BATCH 64
#define LR 0.01F

static Tensor *forward(Tensor *X, Tensor *W1, Tensor *b1, Tensor *W2, Tensor *b2)
{
    return tensor_add(tensor_matmul(tensor_relu(tensor_add(tensor_matmul(X, W1), b1)), W2), b2);
}

static Tensor *cross_entropy(Tensor *logit, const float *label, int batch)
{
    float one[batch * 10];
    memset(one, 0, sizeof(one));
    for (int i = 0; i < batch; i++) {
        one[(i * 10) + (int)label[i]] = 1;
    }
    Tensor *one_hot = tensor_from((int[]){batch, 10}, 2, one);
    Tensor *max = tensor_max(logit, 1, 1);  // subtract max per row for numerical stability
    Tensor *shifted = tensor_sub(logit, max);
    Tensor *log_softmax = tensor_sub(shifted, tensor_log(tensor_sum(tensor_exp(shifted), 1, 1)));
    return tensor_neg(tensor_mean(tensor_sum(tensor_mul(one_hot, log_softmax), 1, 0), INT_MAX, 0));
}

static float accuracy(Tensor *X, Tensor *y, Tensor *W1, Tensor *b1, Tensor *W2, Tensor *b2)
{
    tensor_no_grad_begin();
    Tensor *logit = forward(X, W1, b1, W2, b2);
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
    return (float)correct / (float)n;
}

static void update(Tensor *param, float lr)
{
    Tensor *grad = tensor_grad(param);
    if (!grad) {
        return;
    }
    float *pdata = tensor_data(param);
    float *gdata = tensor_data(grad);
    for (long i = 0; i < tensor_numel(param); i++) {
        pdata[i] -= lr * gdata[i];
    }
    tensor_zero_grad(param);
}

int main(void)
{
    tensor_frame_begin();

    Tensor *X_train = tensor_load("data/X_train.npy");
    Tensor *y_train = tensor_load("data/y_train.npy");
    Tensor *X_test = tensor_load("data/X_test.npy");
    Tensor *y_test = tensor_load("data/y_test.npy");

    srand((unsigned)time(0));
    Tensor *W1 = tensor_requires_grad(tensor_normal((int[]){784, 128}, 2, 0, sqrtf(2 / 784.F)));
    Tensor *b1 = tensor_requires_grad(tensor_zeros((int[]){1, 128}, 2));
    Tensor *W2 = tensor_requires_grad(tensor_normal((int[]){128, 10}, 2, 0, sqrtf(2 / 128.F)));
    Tensor *b2 = tensor_requires_grad(tensor_zeros((int[]){1, 10}, 2));

    Tensor *params[] = {W1, b1, W2, b2};
    float *y_data = tensor_data(y_train);

    int n_train = tensor_shape(X_train)[0];
    for (int epoch = 0; epoch < 10; epoch++) {
        tensor_shuffle((Tensor *[]){X_train, y_train}, 2, 0);

        int n_batches = 0;
        float total_loss = 0;
        for (int i = 0; i < n_train; i += BATCH) {
            tensor_frame_begin();

            int batch = (i + BATCH <= n_train) ? BATCH : (n_train - i);
            Tensor *X_b = tensor_slice(X_train, 0, i, i + batch, 1);
            Tensor *loss = cross_entropy(forward(X_b, W1, b1, W2, b2), y_data + i, batch);
            tensor_backward(loss, 0);

            tensor_no_grad_begin();
            for (int k = 0; k < 4; k++) {
                update(params[k], LR);
            }
            tensor_no_grad_end();

            n_batches += 1;
            total_loss += tensor_data(loss)[0];

            tensor_frame_end();
        }

        printf("Epoch %2d  Loss: %.4f\n", epoch + 1, total_loss / (float)n_batches);
    }

    printf("\nTest accuracy: %.1f%%\n", 100 * accuracy(X_test, y_test, W1, b1, W2, b2));

    tensor_frame_end();
}
// NOLINTEND(readability-identifier-length)
