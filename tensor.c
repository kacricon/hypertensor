#include "tensor.h"
#include <stdlib.h>
#include <errno.h>
#include <string.h>

/* helper functions */
static inline size_t tensor_f32_size(tensor_f32_t *tensor) {
    size_t size = 1;
    for (size_t i = 0; i < tensor->ndim; i++) {
        size *= tensor->shape[i];
    }
    return size;
}

/* tensor creation */
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

    tensor->size = tensor_f32_size(tensor);

    tensor->data = malloc(sizeof(float) * tensor->size);
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

    for (size_t i = 0; i < tensor->size; i++) {
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

/* element-wise ops */
void tensor_f32_add(tensor_f32_t *tensor, float x) {
    if (tensor == NULL) {
        errno = EINVAL;
        return;
    }

    for (size_t i = 0; i < tensor->size; i++) {
        tensor->data[i] += x;
    }
}

void tensor_f32_sub(tensor_f32_t *tensor, float x) {
    if (tensor == NULL) {
        errno = EINVAL;
        return;
    }

    for (size_t i = 0; i < tensor->size; i++) {
        tensor->data[i] -= x;
    }
}

void tensor_f32_mul(tensor_f32_t *tensor, float x) {
    if (tensor == NULL) {
        errno = EINVAL;
        return;
    }

    for (size_t i = 0; i < tensor->size; i++) {
        tensor->data[i] *= x;
    }
}

void tensor_f32_div(tensor_f32_t *tensor, float x) {
    if (tensor == NULL) {
        errno = EINVAL;
        return;
    }

    for (size_t i = 0; i < tensor->size; i++) {
        tensor->data[i] /= x;
    }
}

