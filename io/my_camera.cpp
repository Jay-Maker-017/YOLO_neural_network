#include "my_camera.hpp"

#include <cstring>       // memset
#include <iostream>      // std::cerr（io 模块没有链接 spdlog，用标准错误流输出即可）
#include <string>
#include <unordered_map>

#include "MvCameraControl.h"  // 海康 SDK 主头文件
#include "opencv2/opencv.hpp" // cvtColor / Mat

// ============================================================================
// 匿名 namespace：里面的函数只在"本文件"可见，不会污染其他 .cpp 的全局命名空间。
// 适合放这种内部辅助函数。
// ============================================================================
namespace
{

/**
 * 打印 SDK 错误码。
 *
 * 为什么单独写一个函数：海康 SDK 的每个接口都用 int 返回值表示成功/失败，
 * MV_OK(0) 是成功，其他值是错误码（例如 0x80000007 表示枚举不到设备）。
 * 官方文档都用十六进制描述错误码，所以这里把十进制和十六进制一起打出来，
 * 方便对照文档查原因。
 */
void log_error(const std::string & what, int ret)
{
  std::cerr << "[MyCamera] " << what << " 失败，错误码: " << ret << " (0x" << std::hex << ret
            << std::dec << ")" << std::endl;
}

/**
 * 把 SDK 抓到的原始帧（MV_FRAME_OUT）转换成 OpenCV 的 BGR 图像。
 *
 * 【重要】海康工业相机为了速度和精度，输出的通常是 Bayer 格式（每个像素只有一种颜色），
 * 而 OpenCV / 神经网络 / 显示都需要 3 通道的 BGR，所以必须做一次"去马赛克"（demosaic）。
 *
 * 参考实现来自同目录的 io/example.cpp，这里做了两点改进：
 *   1. 删掉了原来那段没被使用的 MV_CC_PIXEL_CONVERT_PARAM 代码。它填完结构体却从来
 *      没有调用 MV_CC_ConvertPixelType，纯属死代码；而且里面的 nDstBufferSize 按单通道
 *      大小计算，对 3 通道的 BGR 来说其实会越界。
 *   2. 把 .at() 换成 .find()。.at() 在找不到 key 时会抛 std::out_of_range 异常，
 *      而某些相机会输出 Mono8 / BGR8 等非 Bayer 格式，那样程序会直接崩掉；
 *      换成 find() 后转换不了就原样返回，程序仍然能跑。
 *
 * @param raw  SDK 返回的原始帧结构体（包含图像指针 pBufAddr 和帧信息 stFrameInfo）
 * @return     BGR 三通道的 cv::Mat，与 SDK 的缓存无关（已经拷贝出来了）
 */
cv::Mat transfer(MV_FRAME_OUT & raw)
{
  // 第 1 步：先用 SDK 的内存地址创建一个"视图"。
  //   注意！这里 img 并不拥有这块内存，它只是指向 raw.pBufAddr。
  //   所以下面第 3 步一定要把结果 clone / cvtColor 出来，
  //   否则调用者一旦执行 MV_CC_FreeImageBuffer，这块内存就被回收了，img 立刻变成野指针。
  cv::Mat img(cv::Size(raw.stFrameInfo.nWidth, raw.stFrameInfo.nHeight), CV_8U, raw.pBufAddr);

  const auto pixel_type = raw.stFrameInfo.enPixelType;  // 相机实际的像素格式

  // 第 2 步：建立"海康像素格式 → OpenCV 转换码"的对照表。
  //
  //   这里有个 OpenCV 的经典陷阱：转换码名字叫 ...2RGB，实际输出却是按 BGR 排列的！
  //   例如 COLOR_BayerRG2RGB 和 COLOR_BayerBG2BGR 其实是同一个数值（历史遗留命名）。
  //   既然神经网络和 imshow 都要 BGR，就直接写 ...2BGR，含义最清楚。
  //   另外 BayerXX 里的 XX 表示相机传感器的滤色阵列排布，写错了会导致红蓝互换、
  //   整张图偏色，所以每个型号的正确格式最好以厂商手册为准。
  const std::unordered_map<MvGvspPixelType, cv::ColorConversionCodes> type_map = {
    {PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2BGR},
    {PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2BGR},
    {PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2BGR},
    {PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2BGR}};

  // 第 3 步：查表转换。
  const auto it = type_map.find(pixel_type);
  if (it == type_map.end()) {
    // 不认识的格式：不抛异常，直接拷贝一份原图对付过去（至少能跑，屏幕上会看到灰度/偏色）
    std::cerr << "[MyCamera] 遇到未知像素格式: 0x" << std::hex << pixel_type << std::dec
              << "，将直接拷贝原始数据" << std::endl;
    return img.clone();
  }

  // cvtColor 会新分配一块内存来存结果，所以返回值已经完全脱离 SDK 的缓冲区了。
  cv::Mat bgr;
  cv::cvtColor(img, bgr, it->second);
  return bgr;
}

}  // namespace

// ============================================================================
// MyCamera 的四个函数
// ============================================================================

/**
 * 构造函数：刻意"什么都不做"。
 *
 * 为什么不在这里直接打开相机？
 *   因为构造函数没法优雅地报告失败（要么抛异常，要么留下一个半死不活的对象）。
 *   所以把打开动作交给 open()，让调用者用 if (!camera.open()) 显式处理。
 *
 * 另外，成员 handle_ / is_open_ 已经在头文件里写了类内默认值，
 * 所以这里连初始化列表都不用写，它们一定是 nullptr / false。
 */
MyCamera::MyCamera()
{
  // 故意留空
}

/**
 * 析构函数：C++ RAII 的收尾环节——自动把相机资源还回去。
 *
 * 这是"封装"最重要的收益：调用者再也不用记得写一堆 CloseDevice，
 * 对象一离开作用域，这里就会被自动调用。
 *
 * 关闭顺序必须和打开顺序相反（后开的先关）：
 *   开始采集 → 关闭设备 → 销毁句柄
 *
 * 特别提醒：析构函数里绝对不能抛异常（会直接 std::terminate 结束进程），
 * 所以下面 SDK 的返回值都不检查，失败了也只是内部状态问题。
 */
MyCamera::~MyCamera()
{
  if (handle_ == nullptr) return;  // 从没创建过句柄，无事可做

  if (is_open_) {  // 正在采集，先停采集再关设备
    MV_CC_StopGrabbing(handle_);
    MV_CC_CloseDevice(handle_);
  }

  // 句柄本身无论什么状态都要销毁。
  // （open() 失败时会自己回滚并把 handle_ 置回 nullptr，所以能走到这里说明句柄是有效的）
  MV_CC_DestroyHandle(handle_);
  handle_ = nullptr;
  is_open_ = false;
}

/**
 * 打开相机并开始采集。
 *
 * 整个流程照搬 io/example.cpp 里已经验证过的顺序，只是拆进了类里：
 *   枚举设备 → 创建句柄 → 打开设备 → 设置参数 → 开始采集
 *
 * 每一行 SDK 调用都返回一个 int，MV_OK(0) 才代表成功，
 * 其他值都是错误码。所以每一步之后都要判断，失败就"回滚 + 返回 false"。
 */
bool MyCamera::open()
{
  // ------------------------------------------------------------------
  // 第 1 步：枚举（搜索）电脑上连着的海康 USB 相机
  // ------------------------------------------------------------------
  MV_CC_DEVICE_INFO_LIST device_list;

  // 【易错点】一定要先清零！
  // 这个结构体很大，不初始化的话里面全是栈上的随机垃圾数据，
  // SDK 会误以为 pDeviceInfo 数组里已经有内容，行为未定义。
  memset(&device_list, 0, sizeof(device_list));

  int ret = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
  if (ret != MV_OK) {
    log_error("枚举设备 MV_CC_EnumDevices", ret);
    return false;
  }

  if (device_list.nDeviceNum == 0) {
    // 走到这里通常是：相机没插好、被别的程序（如 MVS 客户端）占用、
    // 或者是 USB 3.0 线插在了 2.0 口上导致供电/带宽不足。
    std::cerr << "[MyCamera] 没有找到任何 USB 相机。请检查：相机是否插好、"
                 "是否被 MVS 客户端等其他程序占用"
              << std::endl;
    return false;
  }
  std::cout << "[MyCamera] 共发现 " << device_list.nDeviceNum << " 台相机，使用第 1 台"
            << std::endl;

  // ------------------------------------------------------------------
  // 第 2 步：创建句柄
  // ------------------------------------------------------------------
  // 这一句只是告诉 SDK"我要操作这台相机"，并把一个句柄写到 handle_ 里。
  // 注意：此时还没真正占用相机。
  ret = MV_CC_CreateHandle(&handle_, device_list.pDeviceInfo[0]);
  if (ret != MV_OK) {
    log_error("创建句柄 MV_CC_CreateHandle", ret);
    handle_ = nullptr;  // 保险起见清空，避免析构函数拿着无效句柄去操作
    return false;
  }

  // ------------------------------------------------------------------
  // 第 3 步：打开设备（这一步才真正占用相机）
  // ------------------------------------------------------------------
  ret = MV_CC_OpenDevice(handle_);
  if (ret != MV_OK) {
    log_error("打开设备 MV_CC_OpenDevice", ret);
    // 回滚：句柄已经创建了，必须销毁掉
    MV_CC_DestroyHandle(handle_);
    handle_ = nullptr;
    return false;
  }

  // ------------------------------------------------------------------
  // 第 4 步：设置相机参数
  // ------------------------------------------------------------------
  // 下面这些调用都不检查返回值，原因：
  //   这些参数并非所有型号都支持（例如部分相机没有白平衡自动），
  //   某个参数设置失败不该导致整个程序起不来。
  //   另外 SDK 里设置参数失败也不会留下半开的资源，所以可以放心忽略。
  MV_CC_SetEnumValue(handle_, "BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS);  // 白平衡自动
  MV_CC_SetEnumValue(handle_, "ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);  // 关闭自动曝光
  MV_CC_SetEnumValue(handle_, "GainAuto", MV_GAIN_MODE_OFF);               // 关闭自动增益

  // 曝光和增益手动固定下来，这样每帧亮度一致，
  // 识别算法的阈值门限才稳定（自动曝光会让同一目标忽明忽暗）。
  MV_CC_SetFloatValue(handle_, "ExposureTime", 10000);  // 曝光时间，单位微秒 → 10ms
  MV_CC_SetFloatValue(handle_, "Gain", 20);             // 模拟增益，单位 dB
  MV_CC_SetFrameRate(handle_, 60);                      // 目标帧率 60 fps

  // ------------------------------------------------------------------
  // 第 5 步：开始采集（启动相机往外吐图）
  // ------------------------------------------------------------------
  ret = MV_CC_StartGrabbing(handle_);
  if (ret != MV_OK) {
    log_error("开始采集 MV_CC_StartGrabbing", ret);
    // 回滚：设备已经打开了，必须先关设备再销毁句柄（和打开顺序相反）
    MV_CC_CloseDevice(handle_);
    MV_CC_DestroyHandle(handle_);
    handle_ = nullptr;
    return false;
  }

  is_open_ = true;
  // 建议：第一次接触工业相机的人，看到这句就可以确定"相机真的跑起来了"
  std::cout << "[MyCamera] 相机已打开并开始采集" << std::endl;
  return true;
}

/**
 * 取一帧图像。
 *
 * 取图是"两次配套调用"的：
 *   MV_CC_GetImageBuffer  →  从 SDK 的缓存队列里借出一帧（借来的内存归 SDK 管）
 *   ...自己处理（转换、识别、显示）...
 *   MV_CC_FreeImageBuffer →  用完必须还回去，否则 SDK 的缓存很快被占满，之后再也取不到图
 *
 * 所以这里用"先转换、后释放"的顺序，既保证数据安全，也不会有任何泄漏。
 * 哪怕中途 return，也一定先把缓存还掉。
 */
cv::Mat MyCamera::read()
{
  // 相机还没成功打开就直接返回空图。
  // 这样调用者只需要判断 img.empty() 一种情况，不用再关心相机状态。
  if (!is_open_ || handle_ == nullptr) {
    std::cerr << "[MyCamera] read() 被调用，但相机尚未打开" << std::endl;
    return cv::Mat();
  }

  MV_FRAME_OUT raw;
  memset(&raw, 0, sizeof(raw));  // 同样必须清零，避免结构体里的垃圾字段误导 SDK

  // 第 1 步：取一帧。
  //   第 3 个参数是超时时间（毫秒）：等超过这个时间还没拿到图就返回错误码。
  //   example.cpp 用的是 100ms，这里放宽到 1000ms——相机偶尔抖动时，
  //   宁可多等一会儿也不要疯狂返回空图让上层空转。
  int ret = MV_CC_GetImageBuffer(handle_, &raw, 1000);
  if (ret != MV_OK) {
    // 常见原因：相机掉线、USB 带宽不足丢帧、曝光时间设得过长
    log_error("取图 MV_CC_GetImageBuffer（可能是超时或相机掉线）", ret);
    return cv::Mat();
  }

  // 第 2 步：转成 BGR 的 cv::Mat。
  //   transfer() 内部会新分配内存，所以返回值不再依赖 raw.pBufAddr，
  //   下面第 3 步释放 SDK 缓存是安全的。
  cv::Mat img = transfer(raw);

  // 第 3 步：把 SDK 的缓存还回去（借的东西必须还）。
  //   注意：这里要用同一个 raw 变量，SDK 靠它里面的信息找到对应的那块缓存。
  ret = MV_CC_FreeImageBuffer(handle_, &raw);
  if (ret != MV_OK) {
    // 释放失败不致命，本帧图像仍然有效，打印出来提醒一下即可
    log_error("释放图像缓存 MV_CC_FreeImageBuffer", ret);
  }

  return img;
}
