# Unitree G1 设备测试工具

用于 **2026-09-20 现场测试的特定 Unitree G1 29 轴设备**，覆盖硬件参数读取、传感器状态检查、电机受控测试、普通/深度摄像头采集及雷达数据读取。

这是现场测试程序的整理归档，不是通用 G1 SDK，也不是完整的机器人控制系统。代码依赖当时设备的 SDK、固件、网络与厂家服务配置；其他设备需要逐项核对适配。历史报告中的 `G1-01`、`G1-02` 是测试记录名称，不是整机序列号。

**开始使用： [简明使用指南](docs/usage.md)** · [传感器网页](docs/sensor-web.md) · [电机测试指南](docs/g1-cli-reuse.md)

## 功能板块

| 板块 | 能做什么 | 入口 |
| --- | --- | --- |
| 硬件参数 | 读取 CPU、内存、存储、USB、网卡、温度、功率模式与 CUDA/TensorRT 信息 | `scripts/read_compute_inventory.py` |
| 传感器与关节状态 | 订阅 LowState，记录关节角度、速度、估算力矩、温度、电压、错误码及原始状态快照 | `scripts/read_lowstate.py` |
| 电机测试 | 默认只读预检；显式执行原位保持、小角度往返与受保护的逐轴测试 | `scripts/lowlevel_joint_check.py`、`scripts/upper_body_check.py` |
| 普通摄像头 | 读取厂家视频 RPC，实时显示彩色图像或保存短时采样 | `scripts/camera_sample_check.py` |
| 深度摄像头 | 读取 D435i Z16 深度帧，显示伪彩图；独立工具可保存原始深度 PGM | `scripts/depth_frame_check.c` |
| 雷达 | 订阅 MID360 点云，读取设备信息和诊断告警，提供三维预览及采样导出 | `scripts/lidar_cloud_check.py`、`scripts/livox_readonly_probe.py` |
| 计算检查 | CPU/GPU/DLA 功能自检与有时限、温度检查的 INT8 吞吐短测 | `scripts/compute_smoke.py`、`scripts/compute_int8_bench.py` |

网页集中展示 **RGB、深度、雷达点云**，支持连接/停止、重新连接、快照与检查摘要下载。硬件清单、关节遥测和电机测试通过命令行运行。

## 适用环境

| 位置 | 已使用的环境 |
| --- | --- |
| 操作电脑 | macOS / Apple Silicon，Python 3、SSH、SCP、浏览器 |
| 机器人 | G1 29 轴，Jetson Orin NX 16 GB，Linux，Python 3.8+ |
| 机载依赖 | 已有 `unitree_sdk2py`、Cyclone DDS、NumPy、OpenCV；深度采集需 GCC 与 Linux V4L2 头文件 |
| 计算测试 | 当时机载 TensorRT 8.5、CUDA 11.4 环境 |
| 传感器 | Intel RealSense D435i、Livox MID360 |

程序复用机载环境，不自动安装机器人依赖、升级固件或修改校准。网页无 npm 构建流程、外部 CDN 或云服务依赖。操作电脑不需要安装机器人 SDK。

## 环境和连接

当时开发计算单元地址为 `192.168.123.164`，SSH 用户为 `unitree`，机载 DDS 网卡为 `eth0`。这些是现场配置示例，使用前应核对实物、网口与地址。密码由 SSH 交互输入，不写入脚本。

确认电脑已接入机器人网络且可以 SSH 登录后，在仓库根目录运行：

```sh
python3 scripts/start_sensor_web.py --host 192.168.123.164 --name g1-02
```

启动器检查机载依赖、将采集程序上传到临时目录并建立 SSH 转发。浏览器打开 <http://127.0.0.1:8765> 后，点击 **连接预览**。相机 USB 数据线也必须连接；只有网线不能代替相机 USB 连接。

预览可能临时开启原本停止的头部视频服务和雷达驱动；点击 **停止** 后应等待服务恢复确认，再在终端按 `Ctrl+C` 退出。网络中断时不能假定恢复已完成。具体连接、网络配置与排错见[使用指南](docs/usage.md)。

## 电机控制与结果边界

电机脚本默认只读；`--execute` 才发送动作命令，`--release-motion` 是独立的控制权释放操作。底层测试要求现场吊装、调试模式、身份核对、活动空间及独立停止手段，并检查控制权与关节禁用状态。运行前阅读[底层逐轴说明](docs/ssh-lowlevel-joints.md)。

历史实测完成了部分使能、运动反馈与退出禁用检查，但出现过保护中止和回位偏差超限；上肢 `rt/arm_sdk` 接管仍有未解决现象。**不能将这些程序描述为已通过完整电机验收或适用于任意 G1 的控制器。** 不要为完成动作而放宽保护阈值。

摄像头有画面、雷达有点云只证明当次采集链路可用，不代表标定、精度或长期可靠性合格。深度伪彩图不是已校准距离；GPU/DLA 短测不是持续 100 TOPS 认证。历史 MID360 窗口脏污告警仍需现场处理。

## 目录与文档

```text
scripts/    硬件清单、遥测、计算、电机和传感器测试入口
web/        本地传感器网页、机载 HTTP 服务和数据解析
tests/     离线单元测试（不连接机器人）
docs/       使用指南与专项操作说明
reports/    2026-09-20 历史测试记录与结论
```

- [简明使用指南](docs/usage.md)：连接、分模块运行、停止与排错。
- [实时传感器网页](docs/sensor-web.md)：画面、点云、导出与服务恢复。
- [电机命令行指南](docs/g1-cli-reuse.md)：身份核对、逐轴预检与执行。
- [底层逐轴说明](docs/ssh-lowlevel-joints.md) / [上肢控制说明](docs/ssh-upper-body.md)：两种控制接口的前提、保护及已知限制。
- [历史报告索引](reports/README.md)：硬件、算力、传感器与电机实测记录。

原始证据、图片/点云、PDF、采购验收表及本地 SSH 记录留在本地，不随仓库发布。历史报告引用的 `evidence/`、`output/` 是本地归档路径，公开仓库不包含这些文件。

## 本地验证

沿用标准库 `unittest`，无需连接设备：

```sh
python3 -m unittest discover -s tests -v
```

缺少 NumPy/OpenCV 时，相关解析与颜色测试会跳过。要覆盖这些测试，可使用 uv 的隔离临时环境，不改动系统 Python：

```sh
uv run --no-project --with numpy --with opencv-python-headless \
  python -m unittest discover -s tests -v
```

深度 C 程序使用 Linux V4L2，需在目标 Linux 环境编译；macOS 离线测试不能替代机载编译或实机验证。本项目没有独立版本号体系，此次整理不新增版本号。
