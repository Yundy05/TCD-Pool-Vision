cmake_minimum_required(VERSION 3.20)

project(PoolBallVision LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV 4.12.0 EXACT REQUIRED CONFIG COMPONENTS
    core
    imgproc
    imgcodecs
    highgui
)

# myApplication.cpp now includes main().
add_executable(PoolBallVision myApplication.cpp)

set_target_properties(PoolBallVision PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_include_directories(PoolBallVision PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(PoolBallVision PRIVATE
    ${OpenCV_LIBS}
)

if(MSVC)
    target_compile_options(PoolBallVision PRIVATE /W4)
else()
    target_compile_options(PoolBallVision PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV configuration: ${OpenCV_DIR}")