#include <cuda_runtime.h>

#include <algorithm>
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
    std::printf("Hello from project_3 block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

__global__ void arraySum(float* A, float* result, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && i + stride < size) {
            A[i] += A[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        result[blockIdx.x] = A[i];
    }
}

__global__ void sharedArraySum(float* A, float* result, int size) {
    __shared__ float sdata[256];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    // step 1:
    // sdata[tid] = (i < size) ? A[i] : 0.0f;
    float sum = 0.f;
    while (i < size) {
        sum += A[i];
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();

    // step 2:
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    // step 3:
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

int main() {
    hello_kernel<<<2, 4>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int size = 1 << 20;  // 1 million elements
    size_t bytes = size * sizeof(float);

    std::vector<float> h_data(size);
    float h_result = 0.0f;
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist(0.0f, 10.f);

    for (int i = 0; i < size; ++i) {
        h_data[i] = dist(gen);
    }

    float *d_data, *d_result, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float) * 1));  // Assuming max

    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    std::vector<float> h_partial(gridSize);
    CUDA_CHECK(cudaMalloc(&d_partial, sizeof(float) * gridSize));

    // arraySum<<<gridSize, blockSize>>>(d_data, d_result, size);

    // CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    sharedArraySum<<<gridSize, blockSize>>>(d_data, d_partial, size);
    sharedArraySum<<<1, blockSize>>>(d_partial, d_result, gridSize);

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_partial));
    printf("Sum of array: %f\n", h_result);

    std::cout << "project_3 CUDA starter finished." << std::endl;
    return 0;
}
