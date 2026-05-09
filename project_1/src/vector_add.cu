#include "vector_add.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err));                                \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

namespace {

__global__ void addTwoVector(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

}  // namespace

namespace p1 {

void vector_add_device(const float* d_a, const float* d_b, float* d_c, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    addTwoVector<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
}

void vector_add(const float* a, const float* b, float* c, int n) {
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    vector_add_device(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

}  // namespace p1
