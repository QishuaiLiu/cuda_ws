#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
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
    // __shared__ float sdata[256];
    extern __shared__ float sdata[];

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

    int blockSize = 512;
    int gridSize = (size + blockSize - 1) / blockSize;

    float *d_data, *d_data_backup, *d_result, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_data_backup, bytes));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float) * 1));
    CUDA_CHECK(cudaMalloc(&d_partial, sizeof(float) * gridSize));

    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data_backup, d_data, bytes, cudaMemcpyDeviceToDevice));

    constexpr int WARMUP_ITERS = 5;
    constexpr int TIMED_ITERS = 100;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double gb = (double)bytes / (1024.0 * 1024.0 * 1024.0);

    auto bench = [&](const char* name, auto launch, bool destructive) {
        if (destructive)
            CUDA_CHECK(cudaMemcpy(d_data, d_data_backup, bytes, cudaMemcpyDeviceToDevice));
        for (int i = 0; i < WARMUP_ITERS; ++i) launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (destructive)
            CUDA_CHECK(cudaMemcpy(d_data, d_data_backup, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < TIMED_ITERS; ++i) launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        float avg_ms = total_ms / TIMED_ITERS;
        double bw = gb / (avg_ms / 1000.0);
        std::printf("%-20s avg %.3f ms   effective BW %.1f GiB/s\n", name, avg_ms, bw);
    };

    bench(
        "arraySum", [&] { arraySum<<<gridSize, blockSize>>>(d_data, d_partial, size); },
        /*destructive=*/true);

    size_t shared_bytes = blockSize * sizeof(float);
    bench(
        "sharedArraySum",
        [&] {
            sharedArraySum<<<gridSize, blockSize, shared_bytes>>>(d_data, d_partial, size);
        },
        /*destructive=*/false);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Final correctness run: full two-pass reduction on clean data
    CUDA_CHECK(cudaMemcpy(d_data, d_data_backup, bytes, cudaMemcpyDeviceToDevice));
    sharedArraySum<<<gridSize, blockSize, shared_bytes>>>(d_data, d_partial, size);
    sharedArraySum<<<1, blockSize, shared_bytes>>>(d_partial, d_result, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    float cpu_sum = std::accumulate(h_data.begin(), h_data.end(), 0.0f);
    std::printf("GPU sum: %f   CPU sum: %f\n", h_result, cpu_sum);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_data_backup));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_partial));

    std::cout << "project_3 CUDA starter finished." << std::endl;
    return 0;
}
