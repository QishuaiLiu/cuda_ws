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
    std::printf("Hello from project_4 block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

__global__ void histogramGlobal(const unsigned int* pixels, unsigned int* hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&hist[pixels[idx]], 1);
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
    std::vector<unsigned int> h_result(256);
    std::mt19937 ng(123);
    std::uniform_real_distribution<float> dist(0, 255);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            h_pixels[y * width + x] = static_cast<unsigned int>(dist(ng));
        }
    }

    unsigned int *d_pixels = nullptr, *d_result = nullptr;
    cudaMalloc(&d_pixels, num_pixels * sizeof(unsigned int));
    cudaMalloc(&d_result, 256 * sizeof(unsigned int));
    cudaMemset(d_result, 0, 256 * sizeof(unsigned int));
    cudaMemcpy(d_pixels, h_pixels.data(), num_pixels * sizeof(unsigned int),
               cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (num_pixels + (block_size - 1)) / block_size;

    histogramGlobal<<<grid_size, block_size>>>(d_pixels, d_result, num_pixels);
    cudaMemcpy(h_result.data(), d_result, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < h_result.size(); ++i) {
        std::cout << i << " : " << h_result[i] << "\n";
    }

    std::cout << "project_4 CUDA starter finished." << std::endl;
    return 0;
}
