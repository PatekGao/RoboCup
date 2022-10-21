cmake_minimum_required(VERSION 3.0.2)
project(plane_extract)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)
set(CMAKE_BUILD_TYPE Debug)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    rc_msgs
    cv_bridge
    image_transport
    dynamic_reconfigure
    )
find_package(realsense2 REQUIRED)
find_package(OpenCV  REQUIRED)
find_package(PCL REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES plane_extract
#  CATKIN_DEPENDS other_catkin_pkg
 DEPENDS rc_msgs
)

include_directories(
${catkin_INCLUDE_DIRS}
/usr/local/include
/usr/include/pcl-1.8
/usr/include/eigen3
/usr/include/vtk-6.3
)

add_executable(${PROJECT_NAME}_node src/plane_extract.cpp)

add_dependencies(${PROJECT_NAME}_node rc_msgs_generate_messages)

target_link_libraries(${PROJECT_NAME}_node
${OpenCV_LIBRARIES}
${PCL_LIBRARIES}
${realsense2_LIBRARY}
${catkin_LIBRARIES}
)
