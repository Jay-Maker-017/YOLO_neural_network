cmake_minimum_required(VERSION 3.22)

project(YOLO_neural_network)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

# 导出 compile_commands.json（编辑器 clangd 靠它找到 OpenCV 等头文件路径）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui dnn)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
#find_package(nlohmann_json REQUIRED)

set(_openvino_candidate "$ENV{HOME}/.local/lib/python3.10/site-packages/openvino/cmake")
if(EXISTS "${_openvino_candidate}/OpenVINOConfig.cmake")
  set(OpenVINO_DIR "${_openvino_candidate}")
endif()
find_package(OpenVINO REQUIRED)

# NNdeployment 的 C++20/OpenVINO 设置限制在子目录作用域；先添加部署库，
# 使其目标可由后续 auto_aim 链接。
add_subdirectory(NNdeployment)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

#生成静态库 auto_aim、tools 和 NNdeployment_lib
add_subdirectory(tasks)
add_subdirectory(tools)

#生成主程序（统一处理图片与视频输入）
add_executable(main main.cpp)

target_link_libraries(main PRIVATE                 
    tools
    auto_aim
    fmt::fmt
    yaml-cpp
    spdlog::spdlog
    ${OpenCV_LIBS}
)

