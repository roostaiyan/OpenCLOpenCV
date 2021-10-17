cmake_minimum_required(VERSION 3.0.0)

project(SobelOpenCL VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

message([=[
    1- Please Install Your GPU Driver (OpenCL Vendor).
    2- Make Sure that WITH_OPENCL=ON when Comipling Your OpenCV.
]=])

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})


add_executable(${PROJECT_NAME} "main.cpp" "sobel.cpp")
target_compile_definitions(${PROJECT_NAME} PUBLIC OPENCL_RESOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/opencl/")

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
