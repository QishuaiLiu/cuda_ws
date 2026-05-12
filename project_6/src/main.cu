#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifndef PROJECT_6_ASSET_DIR
#define PROJECT_6_ASSET_DIR "."
#endif

#ifndef PROJECT_6_OUTPUT_DIR
#define PROJECT_6_OUTPUT_DIR "."
#endif

constexpr int TILE = 16;
constexpr int FILTER_R = 2;
constexpr int FILTER_SIZE = 2 * FILTER_R + 1;
[[maybe_unused]] constexpr int SHARED = TILE + 2 * FILTER_R;

__constant__ float d_filter[FILTER_SIZE][FILTER_SIZE];

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
    std::printf("Hello from project_6 block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

__global__ void convKernel(const float* input, float* output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width) return;

    float sum = 0.0f;

    for (int fr = -FILTER_R; fr <= FILTER_R; ++fr) {
        for (int fc = -FILTER_R; fc <= FILTER_R; ++fc) {
            int r = row + fr;
            int c = col + fc;

            r = max(0, min(r, height - 1));
            c = max(0, min(c, width - 1));

            sum += input[r * width + c] * d_filter[fr + FILTER_R][fc + FILTER_R];
        }
    }
    output[row * width + col] = sum;
}

__global__ void convKernelShared(const float* __restrict__ input, float* output, int width,
                                 int height) {
    __shared__ float sdata[SHARED][SHARED];

    for (int i = threadIdx.y; i < SHARED; i += blockDim.y) {
        for (int j = threadIdx.x; j < SHARED; j += blockDim.x) {
            int r = i + blockIdx.y * TILE - FILTER_R;
            int c = j + blockIdx.x * TILE - FILTER_R;

            r = max(0, min(r, height - 1));
            c = max(0, min(c, width - 1));
            sdata[i][j] = input[r * width + c];
        }
    }
    __syncthreads();

    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;

    if (col < width && row < height) {
        float sum = 0.0f;

        for (int fr = 0; fr < FILTER_SIZE; fr++) {
            for (int fc = 0; fc < FILTER_SIZE; fc++) {
                sum += sdata[threadIdx.y + fr][threadIdx.x + fc] * d_filter[fr][fc];
            }
        }
        output[row * width + col] = sum;
    }
}

int main() {
    const std::string input_path = std::string(PROJECT_6_ASSET_DIR) + "/sample.png";
    int width = 0;
    int height = 0;
    int source_channels = 0;
    constexpr int channels = 1;
    unsigned char* image =
        stbi_load(input_path.c_str(), &width, &height, &source_channels, channels);
    if (image == nullptr) {
        std::fprintf(stderr, "Failed to load %s: %s\n", input_path.c_str(), stbi_failure_reason());
        return EXIT_FAILURE;
    }

    std::cout << "Loaded grayscale image: " << input_path << " (" << width << "x" << height
              << ", source channels " << source_channels << ")" << std::endl;

    const std::string output_path = std::string(PROJECT_6_OUTPUT_DIR) + "/sample_copy.png";
    if (stbi_write_png(output_path.c_str(), width, height, channels, image, width * channels) ==
        0) {
        std::fprintf(stderr, "Failed to write %s\n", output_path.c_str());
        stbi_image_free(image);
        return EXIT_FAILURE;
    }
    hello_kernel<<<2, 4>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int size = width * height;
    std::vector<float> h_input(size), h_output(size);

    for (int i = 0; i < size; ++i) {
        h_input[i] = static_cast<float>(image[i]) / 255.0f;
    }
    stbi_image_free(image);

    float h_filter[FILTER_SIZE][FILTER_SIZE] = {{1, 4, 7, 4, 1},
                                                {4, 16, 26, 16, 4},
                                                {7, 26, 41, 26, 7},
                                                {4, 16, 26, 16, 4},
                                                {1, 4, 7, 4, 1}};

    // Normalize so weights sum to 1
    for (int i = 0; i < FILTER_SIZE; i++)
        for (int j = 0; j < FILTER_SIZE; j++) {
            h_filter[i][j] /= 273.0f;
        }

    CUDA_CHECK(cudaMemcpyToSymbol(d_filter, h_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float)));
    float *d_input, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(TILE, TILE);
    dim3 gridDim((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);

    constexpr int WARMUP_ITERS = 5;
    constexpr int TIMED_ITERS = 100;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto clamp_index = [](int value, int limit) {
        if (value < 0) return 0;
        if (value >= limit) return limit - 1;
        return value;
    };

    auto verify = [&](const char* name) {
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
        int row = height / 2;
        int col = width / 2;
        float cpu_sum = 0.0f;
        for (int fr = -FILTER_R; fr <= FILTER_R; ++fr) {
            for (int fc = -FILTER_R; fc <= FILTER_R; ++fc) {
                int r = clamp_index(row + fr, height);
                int c = clamp_index(col + fc, width);
                cpu_sum += h_input[r * width + c] * h_filter[fr + FILTER_R][fc + FILTER_R];
            }
        }
        float gpu_val = h_output[row * width + col];
        std::printf("%-20s CPU %.6f GPU %.6f diff %.6f\n", name, cpu_sum, gpu_val,
                    std::fabs(cpu_sum - gpu_val));
    };

    double gflop = 2.0 * static_cast<double>(size) * FILTER_SIZE * FILTER_SIZE / 1e9;

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

    bench("global conv",
          [&] { convKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height); });
    bench("shared conv", [&] {
        convKernelShared<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    });

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::vector<unsigned char> out_img(size);
    for (int i = 0; i < size; ++i) {
        float value = h_output[i];
        value = (value < 0.0f) ? 0.0f : ((value > 1.0f) ? 1.0f : value);
        out_img[i] = static_cast<unsigned char>(value * 255.0f + 0.5f);
    }

    const std::string conv_output_path = std::string(PROJECT_6_OUTPUT_DIR) + "/output.png";
    if (stbi_write_png(conv_output_path.c_str(), width, height, 1, out_img.data(), width) == 0) {
        std::fprintf(stderr, "Failed to write %s\n", conv_output_path.c_str());
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    std::cout << "project_6 CUDA starter finished." << std::endl;
    return 0;
}
