//   图片/视频  →  识别器（tasks 模块 YOLO）  →  画图显示并保存结果
//
//   main.cpp 里只做"调度"，不实现任何算法：
//   图片与视频输入在本文件处理；神经网络推理与后处理在 tasks/ 里
//
//   运行方式：
//   ./build/main                            默认播放 video/video_combat.avi
//   ./build/main xxx.jpg                    识别单张图片
//   ./build/main xxx.mp4                    识别视频
//
//   绘制结果保存（参考 tradition 的本地保存做法）：
//   图片 → detect_result.jpg；视频 → detect_result.avi（MJPG，帧率沿用输入）
//
//   因为下面写的 "configs/yolo.yaml" 和 yaml 里的 "assets/yolov5.xml" 都是相对路径

#include <algorithm>
#include <filesystem>  
#include <iostream>  
#include <list>   
#include <string>   

#include <cmath>  // std::isfinite：校验视频帧率是否有效

#include <fmt/format.h>  // fmt::format，用来拼显示用的文字

#include <opencv2/opencv.hpp>

#include "tasks/armor.hpp"       // Armor / COLORS / ARMOR_NAMES / ARMOR_TYPES
#include "tasks/yolo.hpp"        // auto_aim::YOLO：现成的识别器
#include "tools/img_tools.hpp"   // tools::draw_points / draw_text：画图小工具


namespace   //用了“内部链接”写法，使得在main.cpp之外难以引用它
{
/**
 * 把识别到的装甲板画到图像上
 *
 * 这里画两种东西：
 *   1. 装甲板的 4 个角点（Armor::points，顺序固定为左上、左下、右下、右上），绿色
 *   2. 类别文字（颜色 / 图案编号 / 大小 / 置信度），黄色，在装甲板上方水平居中
 *
 * @param img    要画的图（会直接改到传进来的那张图上）
 * @param armors 识别结果
 */
void draw_armors(cv::Mat & img, const std::list<auto_aim::Armor> & armors)
{
  // 字体设置
  constexpr double font_scale = 0.6;
  constexpr int thickness = 2;
  const cv::Scalar text_color(0, 255, 255);

  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{} {} {:.2f}", auto_aim::COLORS[armor.color], auto_aim::ARMOR_NAMES[armor.name],
      armor.confidence);

    // 画 4 个绿色角点 
    tools::draw_points(img, armor.points, {0, 255, 0}, 2);

    // 由 4 个角点求装甲板最高点 y 与水平中心 x，用于把文字放在框上方居中
    float top_y = armor.points.front().y;
    float center_x = 0.f;
    for (const auto & point : armor.points) {
      top_y = std::min(top_y, point.y);
      center_x += point.x;
    }
    center_x /= static_cast<float>(armor.points.size());

    int base_line = 0;
    const cv::Size text_size =
      cv::getTextSize(info, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &base_line);

    // 水平居中；再往上留 8 像素间距，同时保证文字不超出图像上边界
    const cv::Point org(
      static_cast<int>(center_x - text_size.width / 2.f),
      std::max(text_size.height + 4, static_cast<int>(top_y) - 8));

    tools::draw_text(img, info, org, text_color, font_scale, thickness);
  }

}  // draw_armors 结束

}  // namespace 结束

int main(int argc, char ** argv)
{
  if (argc > 2) {
    std::cerr << "用法: " << argv[0] << " [图片或视频路径]" << std::endl;
    return 1;
  }

  // 从当前目录或可执行文件所在目录向上寻找项目根目录，
  // 确保 CMake Tools 从 build/ 启动时也能找到配置和模型。
  auto find_project_root = [](std::filesystem::path path) {
    path = std::filesystem::absolute(path);
    while (!path.empty()) {
      if (std::filesystem::exists(path / "configs/yolo.yaml")) return path;
      const auto parent = path.parent_path();
      if (parent == path) break;
      path = parent;
    }
    return std::filesystem::path{};
  };

  auto project_root = find_project_root(std::filesystem::current_path());
  if (project_root.empty()) {
    project_root = find_project_root(std::filesystem::absolute(argv[0]).parent_path());
  }
  if (project_root.empty()) {
    std::cerr << "无法定位项目根目录（未找到 configs/yolo.yaml）" << std::endl;
    return 1;
  }
  std::filesystem::current_path(project_root);

  // 绘制结果的保存路径（相对项目根目录，与 tradition 的本地保存一致）
  const std::string output_image = "detect_result.jpg";
  const std::string output_video = "detect_result.avi";

  const std::string source = argc == 2 ? argv[1] : "video/video_combat.avi";
  cv::VideoCapture cap;
  // imread 强制转成 8 位三通道 BGR，也会丢弃 PNG 的 alpha 通道。
  cv::Mat still_image = cv::imread(source, cv::IMREAD_COLOR);
  if (still_image.empty()) {
    cap.open(source);
    if (!cap.isOpened()) {
      std::cerr << "无法打开图片或视频: " << source << std::endl;
      return 1;
    }
    std::cout << "使用视频作为输入源: " << source << std::endl;
  } else {
    std::cout << "使用图片作为输入源: " << source << " (" << still_image.cols << "x"
              << still_image.rows << ", " << still_image.channels() << " 通道)" << std::endl;
  }

  // 单张图片用 waitKey(0) 暂停；视频用 waitKey(1) 连续播放。
  const bool single_image = !still_image.empty();

  // 视频模式：创建结果视频写入器（MJPG 编码，帧率与尺寸沿用输入视频）。
  // 与 tradition 一致：先尝试打开，失败时只警告，继续处理但不保存视频。
  cv::VideoWriter writer;
  if (!single_image) {
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (!(fps > 0.0) || !std::isfinite(fps)) fps = 30.0;
    const cv::Size frame_size(
      static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
      static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    writer.open(output_video, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frame_size, true);
    if (writer.isOpened()) {
      std::cout << "结果视频输出: " << output_video << " (" << fps << " FPS)" << std::endl;
    } else {
      std::cerr << "[警告] 无法创建结果视频: " << output_video
                << "；将继续处理但不保存视频。" << std::endl;
    }
  }

  // 构造识别器
  std::filesystem::create_directory("logs");

  // 构造参数只有"配置文件路径"，其余参数（模型路径、推理设备、阈值……）
  // 全部由 YOLO 自己从 yaml 里读。
  // 项目根目录已在上面设置为当前目录，配置与模型相对路径可稳定解析。
  auto_aim::YOLO yolo("configs/yolo.yaml");
  std::cout << "识别器初始化完成" << std::endl;


  // 主循环 —— 取图 -> 识别 -> 显示

  int frame_count = 0;  // 帧计数器

  while (true) {
    // 取图
    cv::Mat img;
    if (single_image) {
      //   静态图片：只交出去一次。
      //   下面立刻把 still_image 清空，于是下一轮 img 就是空的，
      //   正好触发"输入源已结束"那条分支正常退出
      img = still_image;
      still_image = cv::Mat();//标记"这张图已经交出去了"
    } else {
      cap >> img;  // 视频：读下一帧给到img
    }

    if (img.empty()) {
      // 图片已处理一次或视频文件已读完，正常退出。
      std::cout << "输入源已结束，共处理 " << frame_count << " 帧" << std::endl;
      break;
    }

    // 识别
    // detect 返回一个装甲板列表（可能一个都没有，size() 为 0）。
    // 第二个参数是帧号，仅用于日志和调试，不传也可以（默认 -1）。
    std::list<auto_aim::Armor> armors = yolo.detect(img, frame_count);

    // 打印结果到终端
    std::cout << "第 " << frame_count << " 帧: 检测到 " << armors.size() << " 个装甲板";
    for (const auto & armor : armors) {
      std::cout << " [" << auto_aim::COLORS[armor.color] << "-" << auto_aim::ARMOR_NAMES[armor.name]
                << "-" << auto_aim::ARMOR_TYPES[armor.type] << " "
                << fmt::format("{:.2f}", armor.confidence) << "]";
    }
    std::cout << std::endl;

    // 绘制结果（保持输入原始分辨率，用于保存）
    cv::Mat result = img.clone();
    draw_armors(result, armors);

    // 保存绘制结果：图片写一张结果图，视频逐帧写入结果文件；
    // 必须在缩放显示之前保存，保证保存的是原始分辨率画面。
    if (single_image) {
      if (cv::imwrite(output_image, result)) {
        std::cout << "结果图片已保存: " << output_image << std::endl;
      } else {
        std::cerr << "无法保存结果图片: " << output_image << std::endl;
      }
    } else if (writer.isOpened()) {
      writer.write(result);
    }

    // 显示：仅在显示副本上缩放，不影响保存的结果
    const int min_display_width = 460;    // 小于这个宽度就放大，不然看不清角点
    const int max_display_width = 1280;   // 大于这个宽度就缩小，不然超出屏幕

    cv::Mat display;
    if (result.cols < min_display_width) {
      // 太窄：等比放大到 min_display_width
      const double scale = static_cast<double>(min_display_width) / result.cols;
      cv::resize(result, display, {}, scale, scale);
    } else if (result.cols > max_display_width) {
      // 太宽：等比缩小到 max_display_width
      const double scale = static_cast<double>(max_display_width) / result.cols;
      cv::resize(result, display, {}, scale, scale);
    } else {
      // 宽度落在 [min, max] 区间内时不动，保持原始分辨率
      display = result;
    }

    cv::imshow("img", display);

    const int key = cv::waitKey(single_image ? 0 : 1);
    if (key == 'q') {
      // 按 q 退出
      std::cout << "收到退出指令，程序结束" << std::endl;
      break;
    }

    frame_count++;
  }

  //   循环结束：视频模式下释放写入器并报告结果文件
  if (!single_image) {
    const bool video_saved = writer.isOpened() && frame_count > 0;
    writer.release();
    if (video_saved) {
      std::cout << "结果视频已保存: " << output_video << std::endl;
    } else if (frame_count > 0) {
      std::cout << "结果视频未保存（VideoWriter 初始化失败）。" << std::endl;
    }
  }

  //   函数即将返回
  return 0;
}