//   相机（io 模块 MyCamera）  →  识别器（tasks 模块 YOLO）  →  画图显示
//
//   main.cpp 里只做"调度"，不实现任何算法：
//   相机怎么开、怎么取图在 io/my_camera.cpp 里；
//   神经网络怎么推理、怎么后处理在 tasks/ 里
//
//   运行方式（先 cd 到 homework/ 目录）：
//   ./build/main                 用海康相机
//   ./build/main xxx.jpg         用一张图片代替相机（没相机时调试用）
//   ./build/main xxx.mp4         用一段视频代替相机
//
//   因为下面写的 "configs/yolo.yaml" 和 yaml 里的 "assets/yolov5.xml" 都是相对路径

#include <filesystem>  
#include <iostream>  
#include <list>   
#include <string>   

#include <fmt/format.h>  // fmt::format，用来拼显示用的文字

//#include "opencv2/opencv.hpp"  // cv::Mat / cv::VideoCapture / cv::imread / cv::imshow 等

#include "io/my_camera.hpp"      // MyCamera：相机封装
#include "tasks/armor.hpp"       // Armor / COLORS / ARMOR_NAMES / ARMOR_TYPES
#include "tasks/yolo.hpp"        // auto_aim::YOLO：现成的识别器
#include "tools/img_tools.hpp"   // tools::draw_points / draw_text：画图小工具

// 上面这几个 include 用的是"以工程根目录为起点"的写法。
// 顶层 CMakeLists.txt 里有 include_directories(${PROJECT_SOURCE_DIR}),
// 把工程根目录加进了头文件搜索路径，所以这种写法一定能找到。

namespace   //用了“内部链接”写法，使得在main.cpp之外难以引用它
{
/**
 * 把识别到的装甲板画到图像上。
 *
 * 为什么要在 main 里自己画？不是识别器已经画了吗？
 *   因为 tasks 模块的 YOLOV5::draw_detections() 是把结果画在它内部 clone 出来的
 *   一份临时副本上，而且那里面最后的 cv::imshow 是被注释掉的。
 *   也就是说它画完就丢了，外面根本看不见。
 *   所以"显示"这件事必须由 main 自己负责。
 *
 * 这里画两种东西：
 *   1. 装甲板的 4 个角点（Armor::points，顺序固定为左上、左下、右下、右上）
 *   2. 类别文字（颜色 / 图案编号 / 大小 / 置信度），写在装甲板中心点位置
 *
 * @param img    要画的图（注意是"引用"，会直接改到调用者传进来的那张图上）
 * @param armors 识别结果
 */
void draw_armors(cv::Mat & img, const std::list<auto_aim::Armor> & armors)
{
  for (const auto & armor : armors) {
    // armor.color / armor.name / armor.type 都是枚举类型。
    // C++ 里枚举可以隐式转换成整数，所以能直接拿来当数组下标；
    // 而 COLORS / ARMOR_NAMES / ARMOR_TYPES 这三个字符串数组的顺序，
    // 恰好和枚举的定义顺序一一对应，所以下标取出来就是对应的名字。
    auto info = fmt::format(
      "{} {} {:.2f}", auto_aim::COLORS[armor.color], auto_aim::ARMOR_NAMES[armor.name],
      armor.confidence);

    // 画 4 个角点。颜色 {0, 255, 0} 在 OpenCV 的 BGR 顺序里是绿色。
    // armor.points 的类型正好是 std::vector<cv::Point2f>，和这个重载匹配。
    tools::draw_points(img, armor.points, {0, 255, 0}, 2);

    // 写类别文字。
    // 【易错点】draw_text 要的是 cv::Point（整数坐标），
    // 而 armor.center 是 cv::Point2f（浮点坐标），必须显式转换，
    // 否则编译会因类型不匹配而报错。
    tools::draw_text(img, info, cv::Point(armor.center), {0, 255, 0}, 0.6, 1);
  }

}  // draw_armors 结束

}  // namespace 结束
//   这个 } 闭合匿名命名空间。
//   只有当命名空间在"main 之前"就关掉main 才是全局函数
//   C++ 标准要求 main 必须有外部链接，否则找不到程序入口

int main(int argc, char ** argv)
{
  // 第 0 步：决定图像从哪里来
  // 命令行传了参数（例如 ./build/main test.jpg）就用文件
  // 没传参数就用真实相机

  const bool use_camera = (argc < 2);
  // shell 自动生成 argv,argc 。这两个是约定俗成放在main里的形参
  // argv[0] 是程序自身的路径，"命令行真正传了几个参数"= argc - 1。

  MyCamera camera;       // 相机对象。构造函数刻意不碰硬件，所以这里创建它是绝对安全的
  cv::VideoCapture cap;  // 从视频文件读帧时用的读取器
  cv::Mat still_image;   // 静态图片模式下：这唯一的一张图

  //   判断输入到底是"单张静态图片"还是"连续帧（相机 / 视频）"
  // 
  //  【踩过的坑】千万别用 cap.get(cv::CAP_PROP_FRAME_COUNT) == 1 来判断！
  //   实测 imgs 目录里的图：
  //     test_1.jpg（JPEG，3 通道）        FRAME_COUNT = 1         ✓ 判对了
  //     test_5.png（PNG，4 通道 RGBA）    FRAME_COUNT = -9.2e+18  ✗ 判错了
  //   那个 -9.22337e+18 就是 LLONG_MIN，是属性取不到时填进去的无效值。
  //   Windows 截屏保存的 PNG 大多是 RGBA 四通道，正好会踩到这个坑。
  //
  //   一旦被误判成"视频"，下面显示时就会用 waitKey(1)——
  //   窗口创建后只存在 1 毫秒就被销毁，看起来就是"图片没显示出来"。
  //   但终端里识别结果照样有，因为打印发生在 imshow 之前，
  //   所以表现就是"有结果、没窗口"。
  //
  //   正确做法：直接用 imread 试着读一次，读得出非空图像 → 就是静态图片。
  //   这个判断不依赖任何后端实现细节，对 JPEG / PNG / RGBA / BMP 全都成立。
  if (use_camera) {
    // 真实相机：显式打开，失败就退出
    if (!camera.open()) {
      std::cerr << "相机打开失败，请检查硬件连接。"
                   "（如果手边没有相机，可以用 ./build/main <图片或视频路径> 来调试）"
                << std::endl;
      return -1;
    }
  } else {
    const std::string source = argv[1];

    // 先按"静态图片"试读。
    //【易错点】这里必须传 IMREAD_COLOR（效果为强制统一成 3 通道 8 位 BGR，alpha 通道直接丢掉（默认值））
    // 它会强制输出 3 通道的 BGR，顺便把截屏图的 alpha 通道丢掉。
    // 若传 IMREAD_UNCHANGED，RGBA 图会带着第 4 个通道进来，
    // 后面 cvtColor / imshow / 神经网络都可能因通道数不对而报错。
    still_image = cv::imread(source, cv::IMREAD_COLOR);

    if (still_image.empty()) {
      // 读不出静态图，那就当视频处理
      cap.open(source);
      if (!cap.isOpened()) {
        std::cerr << "无法打开输入文件: " << source << std::endl;
        return -1;
      }
      std::cout << "使用视频作为输入源: " << source << std::endl;
    } else {
      std::cout << "使用图片作为输入源: " << source << "  (" << still_image.cols << "x"
                << still_image.rows << ", " << still_image.channels() << " 通道)" << std::endl;
    }
  }

  // single_image 为 true：显示时用 waitKey(0) 停住，让你慢慢看；
  //              为 false（相机/视频）：用 waitKey(1) 连续播放。
  const bool single_image = !still_image.empty();


  // 第 1 步：构造识别器
  // tools::logger() 会往 logs/ 目录写日志文件，但它自己不会创建目录，
  // 目录不存在时 spdlog 会直接抛异常。所以这里先确保目录存在。
  std::filesystem::create_directory("logs");

  // 构造参数只有"配置文件路径"，其余参数（模型路径、推理设备、阈值……）
  // 全部由 YOLO 自己从 yaml 里读。
  // 因为 yaml 里写的路径是相对的，所以必须在 homework/ 目录下运行本程序。
  auto_aim::YOLO yolo("configs/yolo.yaml");
  std::cout << "识别器初始化完成" << std::endl;


  // 第 2 步：主循环 —— 取图 → 识别 → 显示

  int frame_count = 0;  // 帧计数器，顺便当作传给识别器的帧号

  while (true) {
    // ---------------- 取图 ----------------
    cv::Mat img;
    if (use_camera) {
      img = camera.read();  // 相机：读一帧
    } else if (single_image) {
      //   静态图片：只交出去一次。
      //   下面立刻把 still_image 清空，于是下一轮 img 就是空的，
      //   正好触发"输入源已结束"那条分支正常退出。
      //  （因为显示时用的是 waitKey(0)，你要先按键才会走到第二轮，
      //   所以这张图会一直显示到你按键为止。）
      img = still_image;
      still_image = cv::Mat();//把一个"空矩阵"赋给它，等于显式地把它"清空"，作为"这张图已经交出去了"的标记。
    } else {
      cap >> img;  // 视频：读下一帧（这是重载运算符，相当于 cap.read(img)）流提取运算，给到img
    }

    if (img.empty()) {
      if (!use_camera) {
        // 文件读完了（或者单张图片已经读过一次），正常收尾退出
        std::cout << "输入源已结束，共处理 " << frame_count << " 帧" << std::endl;
        break;
      }
      // 相机偶尔会因为超时/丢帧取不到图，跳过这一帧继续下一轮即可，不必退出。
      //【重要】这里务必 continue，绝不能把空图送进 detect()！
      // 因为 YOLOV5::detect() 遇到空图会调用 tools::logger() 打警告，
      // 而 logger 要往 logs/ 目录写文件，万一目录不存在就会抛异常。
      std::cerr << "取图失败，跳过本帧" << std::endl;
      continue;
    }

    // ---------------- 识别 ----------------
    // detect 返回一个装甲板列表（可能一个都没有，size() 为 0）。
    // 第二个参数是帧号，仅用于日志和调试，不传也可以（默认 -1）。
    std::list<auto_aim::Armor> armors = yolo.detect(img, frame_count);

    // ---------------- 打印结果到终端 ----------------
    std::cout << "第 " << frame_count << " 帧: 检测到 " << armors.size() << " 个装甲板";
    for (const auto & armor : armors) {
      std::cout << " [" << auto_aim::COLORS[armor.color] << "-" << auto_aim::ARMOR_NAMES[armor.name]
                << "-" << auto_aim::ARMOR_TYPES[armor.type] << " "
                << fmt::format("{:.2f}", armor.confidence) << "]";
    }
    std::cout << std::endl;

    // ---------------- 显示 ----------------
    // 画在副本上，不直接改 img。
    cv::Mat display = img.clone();
    draw_armors(display, armors);

    // 原骨架里写的是 cv::resize(img, img, cv::Size(640, 480))，
    // 那样不管原图多大都硬拉到 640x480，长宽比会被压扁、画面变形，
    // 角点位置看起来也会很怪。
    // 这里改成"按宽度等比缩放"：太宽就缩小、太小就放大，
    // 两种情况都只乘同一个系数（fx = fy = scale），所以长宽比永远不变。
    // 注意两个阈值必须满足 min <= max，否则会互相打架。
    const int min_display_width = 460;    // 小于这个宽度就放大，不然看不清角点
    const int max_display_width = 1280;   // 大于这个宽度就缩小，不然超出屏幕

    if (display.cols < min_display_width) {
      // 太窄（画面太小）：等比放大到 min_display_width
      const double scale = static_cast<double>(min_display_width) / display.cols;
      cv::resize(display, display, {}, scale, scale);
    } else if (display.cols > max_display_width) {
      // 太宽：等比缩小到 max_display_width
      const double scale = static_cast<double>(max_display_width) / display.cols;
      cv::resize(display, display, {}, scale, scale);
    }
    // 宽度落在 [min, max] 区间内时不动，保持原始分辨率(避免无意义的插值损失)

    cv::imshow("img", display);

    // waitKey 有两个作用：
    //   1.处理窗口事件 —— 不调用它，OpenCV 的窗口会卡死、不刷新；
    //   2.等待键盘输入，返回被按下键的 ASCII 码（没按键则返回 -1）。
    // 参数是等待的毫秒数：0 表示无限等待直到按键，1 表示只等 1 毫秒。
    const int key = cv::waitKey(single_image ? 0 : 1);
    if (key == 'q') {
      // 按 q 退出。
      // 原骨架里这句 break 是被注释掉的，结果就是"按了 q 也退不出去"。
      std::cout << "收到退出指令，程序结束" << std::endl;
      break;
    }

    frame_count++;
  }

  //   循环结束，函数即将返回
  //   这里不需要写任何"关闭相机"的代码！
  //   camera 是局部对象，函数返回时会自动调用它的析构函数，
  //   析构函数里会依次执行 StopGrabbing → CloseDevice → DestroyHandle。
  return 0;
}