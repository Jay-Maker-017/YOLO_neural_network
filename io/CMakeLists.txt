cmake_minimum_required(VERSION 3.16)

find_package(yaml-cpp REQUIRED)

# 创建目标 io
add_library(io STATIC 
    my_camera.cpp
)

# hikrobot
#io 自己以及依赖 io 的其他目标，都可以找到：#include "MvCameraControl.h"
target_include_directories(io PUBLIC hikrobot/include) 

#根据 CPU 架构选择动态库路径：
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(io PUBLIC hikrobot/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(io PUBLIC hikrobot/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

#最后链接：
target_link_libraries(io MvCameraControl usb-1.0 yaml-cpp)
