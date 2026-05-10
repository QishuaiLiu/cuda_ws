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

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixMultiplicationRegTiled(const float* __restrict__ A,
                                             const float* __restrict__ B, float* C, int M, int K,
                                             int N) {
    constexpr int threadsPerBlock = (BM / TM) * (BN / TN);

    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Each thread owns a TM x TN tile of C inside the block tile.
    int threadRow = (tid / (BN / TN)) * TM;
    int threadCol = (tid % (BN / TN)) * TN;

    float c_reg[TM][TN] = {0.0f};
    float a_reg[TM];
    float b_reg[TN];

    constexpr int aLoadsPerThread = (BM * BK) / threadsPerBlock;
    constexpr int bLoadsPerThread = (BK * BN) / threadsPerBlock;

    for (int t = 0; t < (K + BK - 1) / BK; ++t) {
        // Cooperative load of A tile (BM rows x BK cols) into shared memory.
#pragma unroll
        for (int i = 0; i < aLoadsPerThread; ++i) {
            int idx = i * threadsPerBlock + tid;
            int r = idx / BK;
            int c = idx % BK;
            int gr = blockRow + r;
            int gc = t * BK + c;
            s_A[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        // Cooperative load of B tile (BK rows x BN cols).
#pragma unroll
        for (int i = 0; i < bLoadsPerThread; ++i) {
            int idx = i * threadsPerBlock + tid;
            int r = idx / BN;
            int c = idx % BN;
            int gr = t * BK + r;
            int gc = blockCol + c;
            s_B[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int i = 0; i < TM; ++i) a_reg[i] = s_A[threadRow + i][k];
#pragma unroll
            for (int j = 0; j < TN; ++j) b_reg[j] = s_B[k][threadCol + j];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    c_reg[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gr = blockRow + threadRow + i;
            int gc = blockCol + threadCol + j;
            if (gr < M && gc < N) C[gr * N + gc] = c_reg[i][j];
        }
    }
}

int main() {
    hello_kernel<<<2, 4>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist(0.0f, 10.f);

    int M = 2048, K = 2048, N = 2048;

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
        CUDA_CHECK(
            cudaMemcpy(h_result.data(), d_result, M * N * sizeof(float), cudaMemcpyDeviceToHost));
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

    bench("naive matmul",
          [&] { matrixMultiplication<<<grid, block>>>(d_first, d_second, d_result, M, K, N); });
    bench("tiled matmul", [&] {
        matrixMultiplicationTiled<<<grid, block>>>(d_first, d_second, d_result, M, K, N);
    });

    constexpr int RT_BM = 128, RT_BN = 128, RT_BK = 8, RT_TM = 8, RT_TN = 8;
    dim3 rtBlock((RT_BN / RT_TN), (RT_BM / RT_TM));  // 16 x 16 = 256 threads
    dim3 rtGrid((N + RT_BN - 1) / RT_BN, (M + RT_BM - 1) / RT_BM);
    bench("reg-tiled matmul", [&] {
        matrixMultiplicationRegTiled<RT_BM, RT_BN, RT_BK, RT_TM, RT_TN>
            <<<rtGrid, rtBlock>>>(d_first, d_second, d_result, M, K, N);
    });

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_first));
    CUDA_CHECK(cudaFree(d_second));
    CUDA_CHECK(cudaFree(d_result));

    std::cout << "project_5 CUDA starter finished." << std::endl;
    return 0;
}
