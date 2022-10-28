cmake_minimum_required(VERSION 2.6)
project(yolov7trt)
add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

# 请在这里设置ONNX文件名称，注意不要包含路径与后缀
set(ONNX_NAME "v7")

# 如果你是不同显卡，请设置为显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
set(CUDA_GEN_CODE "-gencode=arch=compute_86,code=sm_86")

# 如果你的cuda、cudnn、tensorrt不能被自动找到，请到path.cmake中设置路径
include(path.cmake)

message(INFO ${TENSORRT_DIR})


find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE cuda_srcs ${PROJECT_SOURCE_DIR}/src/*.cu)

add_definitions(-DONNX_PATH="${PROJECT_SOURCE_DIR}/engine/${ONNX_NAME}.onnx")
add_definitions(-DMODEL_PATH="${PROJECT_SOURCE_DIR}/engine/${ONNX_NAME}.trtmodel")

link_directories(
        ${TENSORRT_DIR}/lib
        ${CUDA_DIR}/lib64
        ${CUDNN_DIR}/lib
)

cuda_add_library(cucodes SHARED ${cuda_srcs})
add_executable(yolov7 ${cpp_srcs})

target_include_directories(
        cucodes PRIVATE
        include
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_DIR}/include
        ${TENSORRT_DIR}/include
        ${CUDNN_DIR}/include
)

target_include_directories(
        yolov7 PRIVATE
        include
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_DIR}/include
        $/home/stevegao/TensorRT-7.2.3.4/include
        ${CUDNN_DIR}/include
)




target_link_libraries(cucodes nvinfer nvonnxparser)
target_link_libraries(cucodes cuda cublas cudart cudnn)
target_link_libraries(yolov7 ${OpenCV_LIBS})
target_link_libraries(yolov7 cucodes)
