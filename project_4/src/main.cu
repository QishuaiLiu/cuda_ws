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

constexpr int NUM_BINS = 256;

__global__ void hello_kernel() {
    std::printf("Hello from project_4 block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

__global__ void histogramGlobal(const unsigned int* pixels, unsigned int* hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (idx < n) {
        atomicAdd(&hist[pixels[idx]], 1);
        idx += stride;
    }
}

__global__ void histogramShared(const unsigned int* pixels, unsigned int* hist, int n) {
    __shared__ unsigned int local_hist[NUM_BINS];

    int tid = threadIdx.x;
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        local_hist[i] = 0;
    }

    __syncthreads();

    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    while (idx < n) {
        int value = pixels[idx];
        atomicAdd(&local_hist[value], 1);
        idx += stride;
    }

    __syncthreads();

    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&hist[i], local_hist[i]);
    }
}

int main() {
    hello_kernel<<<2, 4>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    constexpr int width = 1024;
    constexpr int height = 768;
    const int num_pixels = width * height;
    std::vector<unsigned int> h_pixels(num_pixels);
    std::vector<unsigned int> h_result(NUM_BINS);
    std::vector<unsigned int> h_expected(NUM_BINS);
    std::mt19937 ng(123);
    std::uniform_int_distribution<int> dist(0, NUM_BINS - 1);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned int pixel = static_cast<unsigned int>(dist(ng));
            h_pixels[y * width + x] = pixel;
            ++h_expected[pixel];
        }
    }

    size_t pixel_bytes = num_pixels * sizeof(unsigned int);
    size_t hist_bytes = NUM_BINS * sizeof(unsigned int);

    unsigned int *d_pixels = nullptr, *d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pixels, pixel_bytes));
    CUDA_CHECK(cudaMalloc(&d_result, hist_bytes));
    CUDA_CHECK(cudaMemcpy(d_pixels, h_pixels.data(), pixel_bytes, cudaMemcpyHostToDevice));

    int block_size = 256;
    int full_grid_size = (num_pixels + (block_size - 1)) / block_size;
    int grid_size = std::min(full_grid_size, 256);

    constexpr int WARMUP_ITERS = 5;
    constexpr int TIMED_ITERS = 100;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto verify = [&](const char* name) {
        CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, hist_bytes, cudaMemcpyDeviceToHost));
        unsigned long long total = 0;
        for (int i = 0; i < NUM_BINS; ++i) {
            total += h_result[i];
            if (h_result[i] != h_expected[i]) {
                std::fprintf(stderr, "%s: mismatch at bin %d: expected %u, got %u\n", name, i,
                             h_expected[i], h_result[i]);
                std::exit(EXIT_FAILURE);
            }
        }
        std::printf("%s check passed. Total pixels: %llu\n", name, total);
    };

    double input_gb = static_cast<double>(pixel_bytes) / (1024.0 * 1024.0 * 1024.0);

    auto bench = [&](const char* name, auto launch) {
        CUDA_CHECK(cudaMemset(d_result, 0, hist_bytes));
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
        double bw = input_gb / (avg_ms / 1000.0);
        std::printf("%-20s avg %.3f ms   input BW %.1f GiB/s\n", name, avg_ms, bw);

        CUDA_CHECK(cudaMemset(d_result, 0, hist_bytes));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        verify(name);
    };

    bench("global atomics", [&] {
        histogramGlobal<<<grid_size, block_size>>>(d_pixels, d_result, num_pixels);
    });
    bench("shared histogram", [&] {
        histogramShared<<<grid_size, block_size>>>(d_pixels, d_result, num_pixels);
    });

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    for (size_t i = 0; i < h_result.size(); ++i) {
        std::cout << i << " : " << h_result[i] << "\n";
    }

    CUDA_CHECK(cudaFree(d_pixels));
    CUDA_CHECK(cudaFree(d_result));

    std::cout << "project_4 CUDA starter finished." << std::endl;
    return 0;
}
