# YOLO_neural_network：纯神经网络装甲板识别


## 1. 工作流程

```
图片/视频帧
   │
   |-> [YOLO 推理] 整帧输入 -> 装甲板（4 角点 + 颜色 + 图案 + 尺寸 + 置信度）
   │       ├─ 同济的yolov5 ：自带 OpenVINO 前处理与解析
   │       └─ infantry_v5 / infantry_v8：用NNdeployment 的 ArmorModel
   │
   |-> [单张图片] PnP 解算 -> 相机系距离（m）+ 重投影 RMS + 坐标轴
   │
   |-> [可视化] 绿色 4 角点 + 黄色标签（颜色 图案 置信度）+ 浅蓝色距离
```

"main.cpp" 里只做调度，不实现任何算法。

可选模型：

| 模型名 | 后端 | 模型文件 |
| --- | --- | --- |
| "yolov5" | 自带 OpenVINO 管线 | "assets/yolov5.xml" |
| "infantry_v5" | NNdeployment ArmorModel | "models/openvino/Infantry-v5n-..." |
| "infantry_v8" | NNdeployment ArmorModel | "models/openvino/Infantry-v8n-..." |

类型 ID 对应关系：

| ID | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 类型 | 哨兵 | 英雄 | 工程 | 3号步兵 | 4号步兵 | 5号步兵 | 前哨站 | 基地(底部) | 基地(顶部) |

颜色 ID：0=蓝 1=红 2=白 3=紫

## 2. 环境依赖

| 依赖 | 版本要求 |
| --- | --- |
| Ubuntu | 22.04 |
| GCC | ≥ 11（C++17，NNdeployment 子目录用 C++20） |
| CMake | ≥ 3.22 |
| OpenCV | 4.x |
| Eigen3 / fmt / spdlog / yaml-cpp | 系统包（apt 安装） |
| OpenVINO | C++ Runtime |

### OpenVINO 安装方式（二选一）

**方式一：pip 安装**

```bash
pip install openvino
```

**方式二：apt 安装**

```bash
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo gpg --output /etc/apt/trusted.gpg.d/intel.gpg --dearmor GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/openvino ubuntu22 main" | sudo tee /etc/apt/sources.list.d/intel-openvino.list
sudo apt update
sudo apt install -y openvino-2025.4.1
```

> 注意：如果你像我一样使用的虚拟机，没有 Intel GPU，配置中的 "device" 必须为 "CPU"（已配置好）
> 只有物理机具备 Intel 核显/独显时才可改为 "GPU"

## 3. 构建

```bash
cd /home/jay/Desktop/auto_aim_homework/YOLO_neural_network
cmake -S . -B build
cmake --build build -j$(nproc)
```
> 注意：记得把路径改成自己的!

想跑 PnP 自测 ：

```bash
cd build && ctest --output-on-failure
```

## 4. 运行

```bash
./build/main                              # 默认视频 video/video_combat.avi + infantry_v8
./build/main imgs/test_1.jpg              # 指定图片（单图会额外做 PnP 测距）
./build/main video/video_combat.avi       # 指定视频
./build/main imgs/test_1.jpg yolov5       # 指定图片和模型
```

- 程序先尝试将输入按图片读取；读取失败时自动尝试作为视频打开，
- 因此图片（jpg/png）和视频（mp4/avi）都可以直接喂给它
- main 会从当前目录向上寻找 "configs/yolo.yaml" 来定位工程根目录，所以从 build/ 里启动
  也能稳定找到配置和模型
- 单张图片启用 PnP：终端额外打印 t_camera 和重投影 RMS，图上画出坐标轴；视频模式只做检测
- 绿点=装甲板四角点，黄字=颜色 图案 置信度，浅蓝字=PnP 距离(m)；
  显示窗口会自适应缩放，保存的是原始分辨率
- 按 q 退出

## 5. 部分文件说明

| 文件 | 说明 |
| --- | --- |
| main.cpp | 调度：定位工程根目录 -> 建识别器 -> 图片/视频判别 -> 逐帧识别、打印、绘制、保存 |
| tasks/ | 识别静态库  |
| tools/ | 画图 与 spdlog 日志 |
| tests/pnp_solver_test.cpp | PnP 合成数据自测 |
| NNdeployment/ | 部署库（完整复制自 深大开源工程） |
| models/openvino/ | Infantry-v5n / v8n 模型（完整复制自 深大开源工程） |
| assets/yolov5.xml | YOLOv5 模型 |
| configs/yolo.yaml | 模型选择、各模型阈值、设备 |
| configs/nndeployment.json | NNdeployment 模型节点配置（"armor_v5" / "armor_v8"） |
| configs/pnp.yaml | 相机内参与标定信息 |
| imgs/、video/ | 测试图片与视频 |
| test_imgs/ | 作业测试图 + 真值 md |
| logs/ | 运行日志 |

## 6. 常见参数调整

- **默认模型**："main.cpp" 里的 "model_name"（当前 "infantry_v8"）；置空则回退到 "configs/yolo.yaml" 的 "yolo_name"
- **各模型置信度阈值**："configs/yolo.yaml" 的 "model_thresholds"
- **NNdeployment 检测分数阈值**："configs/nndeployment.json" 中 "armor_v5" / "armor_v8" 节点的 "score_threshold"
- **颜色过滤**："configs/yolo.yaml" 的 "my_color"
- **相机内参**："configs/pnp.yaml"
- **设备**："configs/yolo.yaml" 的 "device"（无 Intel GPU 的虚拟机必须为 "CPU"）
- **换模型**：把新模型的 ".xml" / ".bin" 放到 "models/openvino/"，再改 "configs/nndeployment.json"
  中对应节点的 "xml" 相对路径；走 yolov5 管线的话则改 "yolo.yaml" 的 "yolov5_model_path"
