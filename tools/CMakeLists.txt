cmake_minimum_required(VERSION 3.16)

add_library(tools STATIC
    img_tools.cpp
    logger.cpp
)

target_link_libraries(tools PUBLIC ${OpenCV_LIBS} fmt::fmt yaml-cpp spdlog::spdlog)