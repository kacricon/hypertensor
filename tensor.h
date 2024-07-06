#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

/* types */
typedef struct {
    float  *data;
    size_t *shape; 
    size_t ndim;
    size_t nelem;
} tensor_f32_t;

/* tensor creation and deletion */
tensor_f32_t *tensor_f32_empty(size_t *shape, size_t ndim);
tensor_f32_t *tensor_f32_full(size_t *shape, size_t ndim, float x);
tensor_f32_t *tensor_f32_zeros(size_t *shape, size_t ndim);
tensor_f32_t *tensor_f32_ones(size_t *shape, size_t ndim);
// tensor_f32_t *tensor_f32_randn(size_t *shape, size_t ndim);
void tensor_f32_free(tensor_f32_t *tensor);

/* tensor movement */
void tensor_f32_reshape(tensor_f32_t *tensor, size_t *shape, size_t ndim);

/* element-wise ops */
void tensor_f32_add(tensor_f32_t *tensor, float x);
void tensor_f32_sub(tensor_f32_t *tensor, float x);
void tensor_f32_mul(tensor_f32_t *tensor, float x);
void tensor_f32_div(tensor_f32_t *tensor, float x);

/* tensor ops */
tensor_f32_t *tensor_f32_matmul(tensor_f32_t *tensor1, tensor_f32_t *tensor2);

#endif

