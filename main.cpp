//   main.cpp 里只做"调度",处理,不实现任何算法
//   神经网络推理与后处理在 tasks/ 里
//
//   运行方式：
//   ./build/main                                      默认视频 + main 中配置的模型
//   ./build/main xxx.jpg                              识别单张图片
//   ./build/main xxx.mp4 infantry_v8                  指定输入和模型
//   可选模型：yolov5 / infantry_v5 / infantry_v8

#include <algorithm>
#include <cstdlib>
#include <filesystem>  
#include <iostream>  
#include <list>   
#include <memory>
#include <string>   
#include <utility>

#include <cmath> 

#include <fmt/format.h> 

#include <opencv2/opencv.hpp>

#include "tasks/armor.hpp" 
#include "tasks/pnp_solver.hpp"
#include "tasks/yolo.hpp"        
#include "tools/img_tools.hpp"   // 画图工具


namespace
{
//把识别到的装甲板画到图像上,画两种东西：
//   1. 装甲板的 4 个绿色角点
//   2. 黄色类别文字（颜色,图案编号,置信度）
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
  // 默认切换变量；设为空字符串时回退到 configs/yolo.yaml 的 yolo_name。
  // 可选：yolov5 / infantry_v5 / infantry_v8 。
  std::string model_name = "infantry_v8";
  if (argc > 3) {
    std::cerr << "用法: " << argv[0] << " [图片或视频路径] [模型名]" << std::endl;
    return 1;
  }
  if (argc == 3) model_name = argv[2];

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
  const bool show_window = std::getenv("DISPLAY") != nullptr ||
                           std::getenv("WAYLAND_DISPLAY") != nullptr;

  // 项目根目录已设置，配置与模型相对路径可稳定解析。
  std::filesystem::create_directories("logs");
  std::unique_ptr<auto_aim::YOLO> yolo;
  try {
    yolo = std::make_unique<auto_aim::YOLO>("configs/yolo.yaml", model_name);
  } catch (const std::exception & e) {
    std::cerr << "识别器初始化失败: " << e.what() << std::endl;
    return 1;
  }
  const std::string selected_model = yolo->model_name();
  std::cout << "使用模型: " << selected_model << std::endl;

  // 绘制结果的保存路径
  const std::string output_image = "detect_result_" + selected_model + ".jpg";
  const std::string output_video = "detect_result_" + selected_model + ".avi";

  const std::string source = argc >= 2 ? argv[1] : "video/video_combat.avi";
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
  std::unique_ptr<auto_aim::PnPSolver> pnp_solver;
  if (single_image) {
    try {
      auto_aim::PnPConfig pnp_config = auto_aim::PnPConfig::load("configs/pnp.yaml");
      if (!pnp_config.calibration_resolution_verified) {
        std::cerr << "[警告] configs/pnp.yaml 中的标定分辨率尚未核实；PnP 距离结果可能不准确。"
                  << std::endl;
      }
      pnp_solver = std::make_unique<auto_aim::PnPSolver>(std::move(pnp_config));
      std::cout << "照片 PnP 已启用，标定配置: configs/pnp.yaml" << std::endl;
    } catch (const std::exception & e) {
      std::cerr << "PnP 初始化失败: " << e.what() << std::endl;
      return 1;
    }
  }

  // 视频模式：创建结果视频写入器
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
    std::list<auto_aim::Armor> armors = yolo->detect(img, frame_count);
    std::vector<auto_aim::PoseResult> poses;
    if (single_image && pnp_solver) {
      poses.reserve(armors.size());
      for (const auto & armor : armors) {
        auto_aim::PoseResult pose = pnp_solver->solve(armor, img.size());
        if (pose.success) {
          std::cout << fmt::format(
            "  PnP: t_camera=({:.1f}, {:.1f}, {:.1f}) mm, 重投影 RMS={:.3f} px",
            pose.tvec_mm[0], pose.tvec_mm[1], pose.tvec_mm[2], pose.reprojection_error_px)
                    << std::endl;
        } else {
          std::cerr << "  PnP 解算失败: " << pose.error << std::endl;
        }
        poses.push_back(std::move(pose));
      }
    }

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
    for (const auto & pose : poses) {
      if (pose.success) pnp_solver->draw_axes(result, pose);
    }

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

    if (show_window) cv::imshow("img", display);

    const int key = show_window ? cv::waitKey(single_image ? 0 : 1) : -1;
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