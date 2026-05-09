cmake_minimum_required(VERSION 3.18)

project(cuda_starter LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# Static lib carrying the kernel + launchers. Shared by the executable and
# the Python module so we have one source of truth.
add_library(project_1_cuda STATIC src/vector_add.cu)
target_include_directories(project_1_cuda PUBLIC include)
target_link_libraries(project_1_cuda PUBLIC CUDA::cudart)
set_target_properties(project_1_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_executable(cuda_starter src/main.cu)
target_link_libraries(cuda_starter PRIVATE project_1_cuda)
set_target_properties(cuda_starter PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# --- Python bindings via nanobind ---
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE NB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_project_1_py STABLE_ABI src/bindings.cpp)
target_link_libraries(_project_1_py PRIVATE project_1_cuda)
