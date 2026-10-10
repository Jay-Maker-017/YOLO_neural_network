cmake_minimum_required(VERSION 3.16)

set(_openvino_candidate "$ENV{HOME}/.local/lib/python3.10/site-packages/openvino/cmake")
if(EXISTS "${_openvino_candidate}/OpenVINOConfig.cmake")
  set(OpenVINO_DIR "${_openvino_candidate}")
endif()

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_library(auto_aim STATIC
    armor.cpp
    yolo.cpp
    yolos/yolov5.cpp
)

# 注意这里有 tools：
#   yolos/yolov5.cpp 里调用了 tools::logger() / tools::draw_points() / tools::draw_text()，
#   所以 auto_aim 依赖 tools 这个库。
#   原来漏写了这一项，链接 main 时会报 `undefined reference to tools::logger()`。
#   （静态库的链接顺序是有讲究的：main 的链接顺序是 tools → auto_aim，
#     tools 在 auto_aim 前面就被扫描完了，轮到 auto_aim 需要 logger 时已经找不到，
#     CMake 只有在这个依赖被显式声明时，才会在 auto_aim 后面再补一次 libtools.a。）
target_link_libraries(auto_aim PUBLIC tools openvino::runtime ${OpenCV_LIBS} fmt::fmt yaml-cpp)