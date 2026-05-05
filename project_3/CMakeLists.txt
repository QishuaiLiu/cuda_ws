cmake_minimum_required(VERSION 3.18)

project(cuda_project_3 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(cuda_project_3
    src/main.cu
)

set_target_properties(cuda_project_3 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
