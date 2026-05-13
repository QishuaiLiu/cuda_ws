#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

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
    std::printf("Hello from project_7 block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

__global__ void prefixSum(const float* __restrict__ input, float* result, int size) { return; }

int main() {
    hello_kernel<<<2, 4>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int size = 2050;
    std::vector<float> h_data(size);
    float result = 0;
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist(0.0f, 10.f);

    for (int i = 0; i < size; ++i) {
        h_data[i] = dist(gen);
    }

    float *d_data, *d_result;
    cudaMalloc(&d_data, size * sizeof(float));
    cudaMalloc(&d_result, 1 * sizeof(float));

    cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_dim(256);
    dim3 grid_dim(size + (block_dim.x - 1) / block_dim.x);

    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "project_7 initialized successfully" << std::endl;
    return 0;
}
