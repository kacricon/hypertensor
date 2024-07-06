#include "tensor.h"
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#define FLOAT_TOLERANCE 1e-6

bool float_cmp(float x, float y) {
    return fabs(x - y) < FLOAT_TOLERANCE;
} 

void test_tensor_f32_empty_valid(void) {
    size_t shape[] = {2, 3};
    tensor_f32_t *tensor = tensor_f32_empty(shape, 2);

    assert(tensor != NULL);
    assert(tensor->data != NULL);
    assert(tensor->shape != NULL);
    assert(tensor->ndim == 2);
    assert(tensor->nelem == 6);
    assert(tensor->shape[0] == 2);
    assert(tensor->shape[1] == 3);

    tensor_f32_free(tensor);

    printf("test_tensor_f32_empty_valid passed\n");
}

void test_tensor_f32_empty_invalid_shape(void) {
    tensor_f32_t *tensor = tensor_f32_empty(NULL, 2);

    assert(tensor == NULL);
    assert(errno == EINVAL);

    tensor_f32_free(tensor);

    printf("test_tensor_f32_empty_invalid_shape passed\n");
}

void test_tensor_f32_empty_zero_ndim(void) {
    size_t shape[] = {2, 3};
    tensor_f32_t *tensor = tensor_f32_empty(shape, 0);

    assert(tensor == NULL);
    assert(errno == EINVAL);

    tensor_f32_free(tensor);

    printf("test_tensor_f32_empty_zero_ndim passed\n");
}

void test_tensor_f32_empty_large_tensor(void) {
    size_t shape[] = {1000, 1000};
    tensor_f32_t *tensor = tensor_f32_empty(shape, 2);

    assert(tensor != NULL);
    assert(tensor->data != NULL);
    assert(tensor->shape != NULL);
    assert(tensor->ndim == 2);
    assert(tensor->nelem == 1000000);

    tensor_f32_free(tensor);

    printf("test_tensor_f32_empty_large_tensor passed\n");
}

void test_tensor_f32_matmul_valid() {
    size_t shape1[] = {2, 3};
    size_t shape2[] = {3, 2};
    tensor_f32_t *tensor1 = tensor_f32_zeros(shape1, 2);
    tensor_f32_t *tensor2 = tensor_f32_zeros(shape2, 2);

    // Fill tensor1 with some values
    // [1.0, 2.0, 3.0]
    // [4.0, 5.0, 6.0]
    tensor1->data[0] = 1.0;
    tensor1->data[1] = 2.0;
    tensor1->data[2] = 3.0;
    tensor1->data[3] = 4.0;
    tensor1->data[4] = 5.0;
    tensor1->data[5] = 6.0;

    // Fill tensor2 with some values
    // [7.0, 8.0]
    // [9.0, 10.0]
    // [11.0, 12.0]
    tensor2->data[0] = 7.0;
    tensor2->data[1] = 8.0;
    tensor2->data[2] = 9.0;
    tensor2->data[3] = 10.0;
    tensor2->data[4] = 11.0;
    tensor2->data[5] = 12.0;

    tensor_f32_t *result = tensor_f32_matmul(tensor1, tensor2);
    assert(result != NULL);
    assert(result->ndim == 2);
    assert(result->shape[0] == 2);
    assert(result->shape[1] == 2);

    // Expected result:
    // [58.0, 64.0]
    // [139.0, 154.0]
    assert(float_cmp(result->data[0], 58.0));
    assert(float_cmp(result->data[1], 64.0));
    assert(float_cmp(result->data[2], 139.0));
    assert(float_cmp(result->data[3], 154.0));

    tensor_f32_free(tensor1);
    tensor_f32_free(tensor2);
    tensor_f32_free(result);
}

void test_tensor_f32_matmul_invalid_shapes() {
    size_t shape1[] = {2, 3};
    size_t shape2[] = {4, 2};
    tensor_f32_t *tensor1 = tensor_f32_zeros(shape1, 2);
    tensor_f32_t *tensor2 = tensor_f32_zeros(shape2, 2);

    tensor_f32_t *result = tensor_f32_matmul(tensor1, tensor2);
    assert(result == NULL);
    assert(errno == EINVAL);

    tensor_f32_free(tensor1);
    tensor_f32_free(tensor2);
}

void test_tensor_f32_matmul_null_input() {
    tensor_f32_t *tensor1 = NULL;
    tensor_f32_t *tensor2 = NULL;

    tensor_f32_t *result = tensor_f32_matmul(tensor1, tensor2);
    assert(result == NULL);
    assert(errno == EINVAL);
}

int main() {
    test_tensor_f32_empty_valid();
    test_tensor_f32_empty_invalid_shape();
    test_tensor_f32_empty_zero_ndim();
    test_tensor_f32_empty_large_tensor();

    test_tensor_f32_matmul_valid();
    test_tensor_f32_matmul_invalid_shapes();
    test_tensor_f32_matmul_null_input();

    printf("All tests passed!\n");
    return 0;
}

