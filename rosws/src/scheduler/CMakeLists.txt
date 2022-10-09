cmake_minimum_required(VERSION 3.0.2)
project(scheduler)

add_definitions(-DOUTPUT_PATH="${PROJECT_SOURCE_DIR}/output/")
set(CMAKE_BUILD_TYPE Debug)
find_package(
        catkin REQUIRED COMPONENTS
        message_runtime
        rc_msgs
        roscpp
        std_msgs
        cv_bridge
        image_transport
        dynamic_reconfigure
)
find_package(OpenCV 4 REQUIRED)
find_package(Boost REQUIRED)

add_subdirectory(clipper)

catkin_package(
        #  INCLUDE_DIRS include
        #  LIBRARIES scheduler
        #  CATKIN_DEPENDS message_generation message_runtime rc_msgs roscpp std_msgs
        #  DEPENDS system_lib
)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node src/main.cpp src/measure.cpp src/identify.cpp src/interface.cpp src/tcpClient.cpp )


# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(
        ${PROJECT_NAME}_node
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        polyclipping
)
