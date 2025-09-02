cmake_minimum_required(VERSION 3.18)

project(yolo11 LANGUAGES CXX CUDA)

# 设置路径
set(OpenCV_DIR "your OpenCV path, e.g. D:\\Opencv481\\opencv\\build\\x64\\vc16\\lib")
set(TRT_DIR "your tensorrt path, e.g. D:\\TensorRT-8.6.1.6")
set(Dirent_INCLUDE_DIRS "your current path ..\\include")

# 编译选项
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)

# 配置 CUDA
# 显式指定 CUDA 11.8 的路径
# set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
# set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/nvcc.exe")

# 重新定义 CUDA_INCLUDE_DIRS 和 CUDA_LIBRARIES
# set(CUDA_INCLUDE_DIRS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include")
# set(CUDA_LIBRARIES "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64")

# 检查 CUDA 是否正确配置
find_package(CUDA REQUIRED)
message(STATUS "CUDA found")
message(STATUS "    CUDA libraries: ${CUDA_LIBRARIES}")
message(STATUS "    CUDA include path: ${CUDA_INCLUDE_DIRS}")

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 75 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11 -g -G")

# 设置 include 路径
include_directories(your tensorrt include path, e.g. D:\\TensorRT-8.6.1.6\\include)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${Dirent_INCLUDE_DIRS}) 

# 配置 OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

# 链接路径
link_directories(your tensorrt lib path, e.g. D:\\TensorRT-8.6.1.6\\lib)

# 添加目标编译
add_library(yolo11 SHARED 
    "yolo11detection.h" 
    "yolo11detection.cpp" 
    "postprocess.h"
    "postprocess.cu"
    "preprocess.h"
    "preprocess.cu"
    "framework.h" 
    "dllmain.cpp" 
    "pch.h"
)

# 链接库
target_link_libraries(yolo11 "nvinfer" "nvinfer_plugin" "nvonnxparser")
target_link_libraries(yolo11 ${OpenCV_LIBS})
target_link_libraries(yolo11 ${CUDA_LIBRARIES})
target_link_libraries(yolo11 Threads::Threads)
