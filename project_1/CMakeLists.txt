cmake_minimum_required(VERSION 3.18)

project(cuda_starter LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cuda_starter
    src/main.cu
)

set_target_properties(cuda_starter PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
