#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "vector_add.h"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err));                                \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

__global__ void hello_kernel() {
    std::printf("Hello from CUDA block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    hello_kernel<<<2, 4>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int size = 1000;
    std::vector<float> h_data_a(size);
    std::vector<float> h_data_b(size);
    std::vector<float> h_data_c(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < size; ++i) {
        h_data_a[i] = dist(gen);
        h_data_b[i] = dist(gen);
    }

    p1::vector_add(h_data_a.data(), h_data_b.data(), h_data_c.data(), size);

    for (int i = 0; i < size; ++i) {
        const float expected = h_data_a[i] + h_data_b[i];
        if (std::fabs(h_data_c[i] - expected) > 1.0e-5f) {
            std::fprintf(stderr, "Mismatch at %d: expected %f, got %f\n", i, expected, h_data_c[i]);
            return EXIT_FAILURE;
        }
    }

    for (int i = 0; i < 5; ++i) {
        std::cout << h_data_a[i] << " + " << h_data_b[i] << " = " << h_data_c[i] << std::endl;
    }

    return 0;
}
