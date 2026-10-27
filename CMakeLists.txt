cmake_minimum_required(VERSION 3.20)

project(PoolBallVision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV 4.12.0 EXACT REQUIRED CONFIG COMPONENTS
    core
    imgproc
    imgcodecs
    highgui
)

add_executable(PoolBallVision
    dev_main.cpp
    myApplication.cpp
    Utilities.cpp
    Geometric.cpp
    Histograms.cpp
    Region.cpp
    Utilities.h
)

target_include_directories(PoolBallVision PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(PoolBallVision PRIVATE
    ${OpenCV_LIBS}
)

target_compile_options(PoolBallVision PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV configuration: ${OpenCV_DIR}")