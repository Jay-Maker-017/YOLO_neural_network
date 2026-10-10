#ifndef IO__MY_CAMERA_HPP
#define IO__MY_CAMERA_HPP

#include "opencv2/core.hpp"

/**
 * 海康（HIKROBOT）USB 工业相机的简易封装。
 *
 * 设计思路（RAII：Resource Acquisition Is Initialization，资源获取即初始化）：
 *   1. 构造函数什么都不做，打开相机要显式调用 open()。
 *      这样"创建对象"和"占用硬件"分开，调用者才有机会处理打开失败。
 *   2. 相机句柄、连接状态都藏在本类的私有成员里，外界拿不到，也不用手动释放。
 *   3. 对象销毁（离开作用域）时，析构函数自动停采集、关设备、销毁句柄。
 *      哪怕中间 return、抛异常，资源也不会泄漏。
 *
 * 典型用法：
 *     {
 *       MyCamera camera;                  // 1. 创建对象（不碰硬件）
 *       if (!camera.open()) return -1;    // 2. 打开相机，失败要自己处理
 *       while (true) {
 *         cv::Mat img = camera.read();    // 3. 反复取图
 *         if (img.empty()) continue;      //    取图可能失败，失败时返回空 Mat
 *       }
 *     }                                   // 4. 不需要手动关闭，析构时自动完成
 */
class MyCamera
{
public:
  MyCamera();
  ~MyCamera();

  /**
   * 枚举设备 → 打开相机 → 设置曝光/增益等参数 → 开始采集。
   * 任意一步失败都会把已经占用的资源回滚干净，并返回 false。
   */
  bool open();

  /**
   * 取一帧图像。返回 BGR 三通道的 cv::Mat（可以直接喂给识别器/显示）。
   * 失败时（相机没打开 / 取图超时 / 相机掉线）返回空 Mat，即 img.empty() == true。
   */
  cv::Mat read();

private:
  // 海康 SDK 里所有操作都要传这个句柄（void* 是 SDK 自己的类型写法）。
  // nullptr 表示"当前没有可用的相机句柄"。
  // 注意 = nullptr 是"类内默认初始化"，等价于在构造函数初始化列表里写 handle_(nullptr)，
  // 这样可以保证：就算构造函数体里什么都不做，成员也是确定的好值，不是栈上的垃圾数据。
  void * handle_ = nullptr;

  // 是否已经成功打开设备并开始了采集。
  // 用来区分两种状态：
  //   handle_ 非空 + is_open_ 为 false → 只创建了句柄，还没开始采集
  //   handle_ 非空 + is_open_ 为 true  → 正在采集，可以取图
  bool is_open_ = false;
};

#endif  // IO__MY_CAMERA_HPP