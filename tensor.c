#include "tensor.h"
#include <stdlib.h>
#include <errno.h>
#include <string.h>

/* tensor creation and deletion */
tensor_f32_t *tensor_f32_empty(size_t *shape, size_t ndim) {
    if (shape == NULL || ndim == 0) {
        errno = EINVAL;
        return NULL;
    }

    tensor_f32_t *tensor = malloc(sizeof(tensor_f32_t));
    if (tensor == NULL) {
        return NULL;
    }

    tensor->shape = malloc(sizeof(size_t) * ndim);
    if (tensor->shape == NULL) {
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, sizeof(size_t) * ndim);
    tensor->ndim = ndim;

    tensor->nelem = 1;
    for (size_t i = 0; i < tensor->ndim; i++) {
        tensor->nelem *= tensor->shape[i];
    }

    tensor->data = malloc(sizeof(float) * tensor->nelem);
    if (tensor->data == NULL) {
        free(tensor->shape);
        free(tensor);
    }

    return tensor;
}

tensor_f32_t *tensor_f32_full(size_t *shape, size_t ndim, float x) {
    tensor_f32_t *tensor = tensor_f32_empty(shape, ndim);
    if (tensor == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < tensor->nelem; i++) {
        tensor->data[i] = x;
    }

    return tensor;
}

tensor_f32_t *tensor_f32_zeros(size_t *shape, size_t ndim) {
    tensor_f32_t *tensor = tensor_f32_full(shape, ndim, 0.0);
    if (tensor == NULL) {
        return NULL;
    }

    return tensor;
}

tensor_f32_t *tensor_f32_ones(size_t *shape, size_t ndim) {
    tensor_f32_t *tensor = tensor_f32_full(shape, ndim, 1.0);
    if (tensor == NULL) {
        return NULL;
    }

    return tensor;
}

void tensor_f32_free(tensor_f32_t *tensor) {
    if (tensor == NULL) {
        return;
    }
    if (tensor->data != NULL) {
        free(tensor->data);
    }
    if (tensor->shape != NULL) {
        free(tensor->shape);
    }
    free(tensor);
}

/* tensor movement */
void tensor_f32_reshape(tensor_f32_t *tensor, size_t *shape, size_t ndim) {
    if (tensor == NULL || shape == NULL || ndim == 0) {
        errno = EINVAL;
        return;
    }

    size_t nelem = 1;
    for (size_t i = 0; i < ndim; i++) {
        nelem *= shape[i];
    }
    if (nelem != tensor->nelem) {
        errno = EINVAL;
        return;
    }

    size_t *new_shape = malloc(sizeof(size_t) * ndim);
    if (new_shape == NULL) {
        return;
    }
    memcpy(new_shape, shape, sizeof(size_t) * ndim);

    free(tensor->shape);
    tensor->shape = new_shape;
    tensor->ndim = ndim;
}

/* element-wise ops */
void tensor_f32_add(tensor_f32_t *tensor, float x) {
    if (tensor == NULL) {
        errno = EINVAL;
        return;
    }

    for (size_t i = 0; i < tensor->nelem; i++) {
        tensor->data[i] += x;
    }
}

void tensor_f32_sub(tensor_f32_t *tensor, float x) {
    if (tensor == NULL) {
        errno = EINVAL;
        return;
    }

    for (size_t i = 0; i < tensor->nelem; i++) {
        tensor->data[i] -= x;
    }
}

void tensor_f32_mul(tensor_f32_t *tensor, float x) {
    if (tensor == NULL) {
        errno = EINVAL;
        return;
    }

    for (size_t i = 0; i < tensor->nelem; i++) {
        tensor->data[i] *= x;
    }
}

void tensor_f32_div(tensor_f32_t *tensor, float x) {
    if (tensor == NULL) {
        errno = EINVAL;
        return;
    }

    for (size_t i = 0; i < tensor->nelem; i++) {
        tensor->data[i] /= x;
    }
}

/* tensor ops                                      */
tensor_f32_t *tensor_f32_matmul(tensor_f32_t *tensor1, tensor_f32_t *tensor2) {
    // v1 only works with 2 dimensions for now
    // will be generalized later
    if (tensor1 == NULL || tensor2 == NULL || tensor1->ndim != 2 || tensor2->ndim != 2) {
        errno = EINVAL;
        return NULL;
    }

    if (tensor1->shape[tensor1->ndim - 1] != tensor2->shape[tensor2->ndim - 2]) {
        errno = EINVAL;
        return NULL;
    }

    size_t out_shape[2] = {tensor1->shape[0], tensor2->shape[1]};
    tensor_f32_t *out = tensor_f32_zeros(out_shape, 2);
    if (out == NULL) {
        return NULL;
    }

    // perform multiplication
    for (size_t i = 0; i < out->shape[0]; i++) {
        for (size_t j = 0; j < out->shape[1]; j++) {
            for (size_t k = 0; k < tensor1->shape[1]; k++) {
                out->data[i * out->shape[1] + j] += tensor1->data[i * tensor1->shape[1] + k] * tensor2->data[k * tensor2->shape[1] + j];
            }
        }
    }

    return out;
}

