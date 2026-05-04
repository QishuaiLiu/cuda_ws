cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build for")
    else()
        set(CMAKE_CUDA_ARCHITECTURES 86-real CACHE STRING "CUDA architectures to build for")
    endif()
endif()

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
