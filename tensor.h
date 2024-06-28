#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

/* types */
typedef struct {
    float  *data;
    size_t *shape; 
    size_t ndim;
    size_t size;
} tensor_f32_t;

/* tensor creation */
tensor_f32_t *tensor_f32_empty(size_t *shape, size_t ndim);
tensor_f32_t *tensor_f32_full(size_t *shape, size_t ndim);
tensor_f32_t *tensor_f32_zeros(size_t *shape, size_t ndim);
tensor_f32_t *tensor_f32_ones(size_t *shape, size_t ndim);
// tensor_f32_t *tensor_f32_randn(size_t *shape, size_t ndim);

/* element-wise ops */
void tensor_f32_add(tensor_f32_t *tensor, float x);
void tensor_f32_sub(tensor_f32_t *tensor, float x);
void tensor_f32_mul(tensor_f32_t *tensor, float x);
void tensor_f32_div(tensor_f32_t *tensor, float x);

/* tensor ops                                      */
/* (results are always stored in the first tensor) */
// void tensor_f32_dot(tensor_f32_t *tensor1, tensor_f32_t *tensor2);
// void tensor_f32_matmul(tensor_f32_t *tensor1, tensor_f32_t *tensor2);

#endif

