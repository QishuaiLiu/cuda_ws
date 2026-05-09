#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#define TILE 32

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
    std::printf("Hello from project_5 block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

__global__ void matrixMultiplication(const float* __restrict__ A, const float* __restrict__ B,
                                     float* result, int M, int K, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0;
    if (idx < N && idy < M) {
        for (int i = 0; i < K; ++i) {
            sum += A[idy * K + i] * B[i * N + idx];
        }
    }

    if (idy < M && idx < N) {
        result[idy * N + idx] = sum;
    }
}
__global__ void matrixMultiplicationTiled(const float* __restrict__ A, const float* __restrict__ B,
                                          float* result, int M, int K, int N) {
    __shared__ float s_A[TILE][TILE + 1];
    __shared__ float s_B[TILE][TILE + 1];
    float sum = 0;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        if (row < M && t * TILE + tx < K) {
            s_A[ty][tx] = A[row * K + t * TILE + tx];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        if (col < N && t * TILE + ty < K) {
            s_B[ty][tx] = B[(ty + t * TILE) * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        result[row * N + col] = sum;
    }
    return;
}

int main() {
    hello_kernel<<<2, 4>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist(0.0f, 10.f);

    int M = 745, K = 1024, N = 525;

    std::vector<float> h_first_matrix(M * K);
    std::vector<float> h_second_matrix(K * N);
    std::vector<float> h_result(M * N, 0.f);

    for (int i = 0; i < M * K; ++i) {
        h_first_matrix[i] = dist(gen);
    }

    for (int i = 0; i < K * N; ++i) {
        h_second_matrix[i] = dist(gen);
    }

    float *d_first, *d_second, *d_result;

    CUDA_CHECK(cudaMalloc(&d_first, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_second, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, M * N * sizeof(float)));

    CUDA_CHECK(
        cudaMemcpy(d_first, h_first_matrix.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_second, h_second_matrix.data(), N * K * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    constexpr int WARMUP_ITERS = 5;
    constexpr int TIMED_ITERS = 50;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto verify = [&](const char* name) {
        CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, M * N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        int row = 2, col = 3;
        float cpu_sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            cpu_sum += h_first_matrix[row * K + k] * h_second_matrix[k * N + col];
        }
        float gpu_val = h_result[row * N + col];
        std::printf("%-20s CPU %f GPU %f diff %f\n", name, cpu_sum, gpu_val,
                    std::fabs(cpu_sum - gpu_val));
    };

    double gflop = 2.0 * static_cast<double>(M) * N * K / 1e9;

    auto bench = [&](const char* name, auto launch) {
        for (int i = 0; i < WARMUP_ITERS; ++i) launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < TIMED_ITERS; ++i) launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        float avg_ms = total_ms / TIMED_ITERS;
        double gflops = gflop / (avg_ms / 1000.0);
        std::printf("%-20s avg %.3f ms   %.1f GFLOP/s\n", name, avg_ms, gflops);

        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        verify(name);
    };

    bench("naive matmul", [&] {
        matrixMultiplication<<<grid, block>>>(d_first, d_second, d_result, M, K, N);
    });
    bench("tiled matmul", [&] {
        matrixMultiplicationTiled<<<grid, block>>>(d_first, d_second, d_result, M, K, N);
    });

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_first));
    CUDA_CHECK(cudaFree(d_second));
    CUDA_CHECK(cudaFree(d_result));

    std::cout << "project_5 CUDA starter finished." << std::endl;
    return 0;
}
