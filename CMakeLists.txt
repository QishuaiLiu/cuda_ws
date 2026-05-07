cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build for")
    else()
        set(CMAKE_CUDA_ARCHITECTURES 86-real 89-real 120-real CACHE STRING "CUDA architectures to build for")
    endif()
endif()

project(cuda_ws LANGUAGES CXX CUDA)

add_subdirectory(project_1)
add_subdirectory(project_2)
add_subdirectory(project_3)
add_subdirectory(project_4)
add_subdirectory(project_5)
