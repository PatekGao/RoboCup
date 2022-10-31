cmake_minimum_required(VERSION 3.0.2)
project(yolo_identify)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
add_definitions(-DENGINE_PATH="${PROJECT_SOURCE_DIR}")
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# 请在这里设置ONNX文件名称，注意不要包含路径与后缀
set(ONNX_NAME "v7")

# 如果你是不同显卡，请设置为显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
set(CUDA_GEN_CODE "-gencode=arch=compute_86,code=sm_86")

# 如果你的cuda、cudnn、tensorrt不能被自动找到，请到path.cmake中设置路径
include(path.cmake)

message(INFO ${TENSORRT_DIR})

find_package(catkin REQUIRED COMPONENTS
        cv_bridge
        image_transport
        rc_msgs
        roscpp
        std_msgs
        dynamic_reconfigure
        )
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
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
add_executable(${PROJECT_NAME}_node ${cpp_srcs})

target_include_directories(
        cucodes PRIVATE
        include
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_DIR}/include
        ${TENSORRT_DIR}/include
        ${CUDNN_DIR}/include
)

target_include_directories(
        ${PROJECT_NAME}_node PRIVATE
        $/home/stevegao/TensorRT-7.2.3.4/include
)

target_link_libraries(cucodes nvinfer nvonnxparser)
target_link_libraries(cucodes cuda cublas cudart cudnn)
target_link_libraries(${PROJECT_NAME}_node ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME}_node cucodes)
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES})

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
    include_directories(/home/stevegao/TensorRT-7.2.3.4/include)
    link_directories(/home/stevegao/TensorRT-7.2.3.4/lib)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
#cuda_add_library(myplugins_identify SHARED src/yololayer.cu)
#target_link_libraries(myplugins_identify nvinfer cudart)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES yolo_identify
  CATKIN_DEPENDS cv_bridge image_transport rc_msgs roscpp std_msgs
#  DEPENDS system_lib
)
include_directories(
         include
        ${catkin_INCLUDE_DIRS}
)

#cuda_add_executable(yolov5_identify src/calibrator.cpp src/yolov5.cpp src/preprocess.cu)
#add_dependencies(yolov5_identify rc_msgs_generate_messages)
#target_link_libraries(yolov5_identify nvinfer)
#target_link_libraries(yolov5_identify cudart)
#target_link_libraries(yolov5_identify myplugins_identify)
#target_link_libraries(yolov5_identify ${catkin_LIBRARIES})
#target_link_libraries(yolov5_identify ${OpenCV_LIBS})

if(UNIX)
    add_definitions(-O2 -pthread)
endif(UNIX)
