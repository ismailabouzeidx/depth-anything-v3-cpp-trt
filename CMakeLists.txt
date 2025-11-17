cmake_minimum_required(VERSION 3.18)
project(depth_anything_3_cpp_trt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS
    ${TENSORRT_ROOT}
    ${TENSORRT_BUILD}
    ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(TENSORRT_LIBRARY_INFER nvinfer
    PATHS
    ${TENSORRT_ROOT}
    ${TENSORRT_BUILD}
    ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
    PATHS
    ${TENSORRT_ROOT}
    ${TENSORRT_BUILD}
    ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find cuDNN (optional)
find_library(CUDNN_LIBRARY cudnn
    PATHS
    ${CUDA_TOOLKIT_ROOT_DIR}
    /usr/lib/x86_64-linux-gnu
    PATH_SUFFIXES lib lib64 lib/x64)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/main.cpp
    src/tensorrt_engine.cpp
    src/preprocessing.cpp
    src/postprocessing.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})
# Set output name to snake_case
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "depth_anything_3_cpp_trt")

# Link libraries
set(LINK_LIBS
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_ONNXPARSER}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    cudart
    cublas
)

# Add cuDNN if found
if(CUDNN_LIBRARY)
    message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
    list(APPEND LINK_LIBS ${CUDNN_LIBRARY})
else()
    message(STATUS "cuDNN not found, skipping (may be included in TensorRT)")
endif()

target_link_libraries(${PROJECT_NAME} ${LINK_LIBS})

# Compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -Wextra
    -O3
)

