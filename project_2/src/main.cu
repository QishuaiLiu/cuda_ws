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

__global__ void matrixTranspose(const float* A, float* B, int cols, int rows) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (col_idx < cols && row_idx < rows) {
        B[col_idx * rows + row_idx] = A[row_idx * cols + col_idx];
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
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    matrixTranspose<<<grid, block>>>(d_A, d_B, cols, rows);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_transposed.data(), d_B, output_bytes, cudaMemcpyDeviceToHost));

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float expected = h_data[r * cols + c];
            float actual = h_transposed[c * rows + r];
            if (std::fabs(expected - actual) > 1.0e-5f) {
                std::fprintf(stderr, "Mismatch at A(%d, %d): expected %f, got %f\n", r, c, expected,
                             actual);
                return EXIT_FAILURE;
            }
        }
    }

    std::cout << "original A: " << rows << " x " << cols << std::endl;
    printMatrixSample(h_data, rows, cols);

    std::cout << "\n Transposed B: " << cols << " x " << rows << std::endl;
    printMatrixSample(h_transposed, cols, rows);
    std::cout << "\nTranspose check passed." << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    return 0;
}
