#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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
    std::printf("Hello from CUDA block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

__global__ void matrixTranspose(const float* A, float* B, int rows, int cols) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx < cols && row_idx < rows) {
        B[col_idx * rows + row_idx] = A[row_idx * cols + col_idx];
    }
}

template <int TILE_DIM, int BLOCK_ROWS>
__global__ void transposeTiled(const float* __restrict__ A, float* __restrict__ B, int rows,
                               int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < cols && y + j < rows) {
            tile[threadIdx.y + j][threadIdx.x] = A[(y + j) * cols + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < rows && y + j < cols) {
            B[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void printMatrixSample(const std::vector<float>& data, int rows, int cols) {
    int sample_rows = std::min(rows, 5);
    int sample_cols = std::min(cols, 5);

    for (int r = 0; r < sample_rows; ++r) {
        for (int c = 0; c < sample_cols; ++c) {
            std::cout << data[r * cols + c] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    hello_kernel<<<2, 4>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int rows = 1000, cols = 2000;
    int input_size = rows * cols;
    int output_size = cols * rows;
    size_t input_bytes = input_size * sizeof(float);
    size_t output_bytes = output_size * sizeof(float);

    std::vector<float> h_data(input_size);
    std::vector<float> h_transposed(output_size);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist(0.0f, 10.f);

    for (int i = 0; i < input_size; ++i) {
        h_data[i] = dist(gen);
    }

    float *d_A = nullptr, *d_B = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, output_bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_data.data(), input_bytes, cudaMemcpyHostToDevice));

    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;

    dim3 baseline_block(32, 32);
    dim3 baseline_grid((cols + baseline_block.x - 1) / baseline_block.x,
                       (rows + baseline_block.y - 1) / baseline_block.y);

    dim3 tiled_block(TILE_DIM, BLOCK_ROWS);
    dim3 tiled_grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);

    auto verify = [&](const char* name) {
        CUDA_CHECK(cudaMemcpy(h_transposed.data(), d_B, output_bytes, cudaMemcpyDeviceToHost));
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                float expected = h_data[r * cols + c];
                float actual = h_transposed[c * rows + r];
                if (std::fabs(expected - actual) > 1.0e-5f) {
                    std::fprintf(stderr, "%s: mismatch at A(%d, %d): expected %f, got %f\n", name,
                                 r, c, expected, actual);
                    std::exit(EXIT_FAILURE);
                }
            }
        }
    };

    constexpr int WARMUP_ITERS = 5;
    constexpr int TIMED_ITERS = 100;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

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
        // transpose moves each element once read + once written
        double gb = 2.0 * input_bytes / (1024.0 * 1024.0 * 1024.0);
        double bw = gb / (avg_ms / 1000.0);
        std::printf("%-20s avg %.3f ms   effective BW %.1f GiB/s\n", name, avg_ms, bw);

        verify(name);
    };

    bench("baseline", [&] {
        matrixTranspose<<<baseline_grid, baseline_block>>>(d_A, d_B, rows, cols);
    });
    bench("tiled (32x8)", [&] {
        transposeTiled<TILE_DIM, BLOCK_ROWS>
            <<<tiled_grid, tiled_block>>>(d_A, d_B, rows, cols);
    });

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::cout << "\noriginal A: " << rows << " x " << cols << std::endl;
    printMatrixSample(h_data, rows, cols);

    std::cout << "\nTransposed B: " << cols << " x " << rows << std::endl;
    printMatrixSample(h_transposed, cols, rows);
    std::cout << "\nTranspose check passed." << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    return 0;
}
