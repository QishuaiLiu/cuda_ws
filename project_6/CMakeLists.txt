cmake_minimum_required(VERSION 3.18)

project(cuda_project_6 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(cuda_project_6
    include/stb_image.h
    include/stb_image_write.h
    src/main.cu
)

target_include_directories(cuda_project_6 PRIVATE include)
target_compile_definitions(cuda_project_6 PRIVATE
    PROJECT_6_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets"
    PROJECT_6_OUTPUT_DIR="${CMAKE_CURRENT_BINARY_DIR}"
)

configure_file(assets/sample.png assets/sample.png COPYONLY)

set_target_properties(cuda_project_6 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
