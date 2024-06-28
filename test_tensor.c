#include "tensor.h"
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
#include <stdio.h>

void test_tensor_f32_empty_valid(void) {
    size_t shape[] = {2, 3};
    tensor_f32_t *tensor = tensor_f32_empty(shape, 2);

    assert(tensor != NULL);
    assert(tensor->data != NULL);
    assert(tensor->shape != NULL);
    assert(tensor->ndim == 2);
    assert(tensor->size == 6);
    assert(tensor->shape[0] == 2);
    assert(tensor->shape[1] == 3);

    free(tensor->data);
    free(tensor->shape);
    free(tensor);

    printf("test_tensor_f32_empty_valid passed\n");
}

void test_tensor_f32_empty_invalid_shape(void) {
    tensor_f32_t *tensor = tensor_f32_empty(NULL, 2);

    assert(tensor == NULL);
    assert(errno == EINVAL);

    printf("test_tensor_f32_empty_invalid_shape passed\n");
}

void test_tensor_f32_empty_zero_ndim(void) {
    size_t shape[] = {2, 3};
    tensor_f32_t *tensor = tensor_f32_empty(shape, 0);

    assert(tensor == NULL);
    assert(errno == EINVAL);

    printf("test_tensor_f32_empty_zero_ndim passed\n");
}

void test_tensor_f32_empty_large_tensor(void) {
    size_t shape[] = {1000, 1000};
    tensor_f32_t *tensor = tensor_f32_empty(shape, 2);

    assert(tensor != NULL);
    assert(tensor->data != NULL);
    assert(tensor->shape != NULL);
    assert(tensor->ndim == 2);
    assert(tensor->size == 1000000);

    free(tensor->data);
    free(tensor->shape);
    free(tensor);

    printf("test_tensor_f32_empty_large_tensor passed\n");
}

int main() {
    test_tensor_f32_empty_valid();
    test_tensor_f32_empty_invalid_shape();
    test_tensor_f32_empty_zero_ndim();
    test_tensor_f32_empty_large_tensor();

    printf("All tests passed\n");
    return 0;
}

