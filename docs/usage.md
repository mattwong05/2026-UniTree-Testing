# 简明使用指南

本指南针对当时测试的 G1 29 轴设备及其机载环境。先做只读检查，再按需要采集传感器或进行现场受控电机测试。所有电脑端命令均从仓库根目录执行。

首次获取仓库：

```sh
git clone https://github.com/mattwong05/2026-UniTree-Testing.git
cd 2026-UniTree-Testing
```

## 1. 连接设备

电脑用网线连接机器人开发网口，确认目标计算单元地址、SSH 用户和设备身份。示例使用 `unitree@192.168.123.164`：

```sh
ssh unitree@192.168.123.164
```

首次连接核对 SSH 指纹，密码交互输入。登录后用 `exit` 回到电脑终端。IP、SSH 主机密钥或开发板标识可能随克隆系统重复，不能单独作为整机身份凭据。

若 macOS 的有线网卡尚未配置同网段地址，先用 `networksetup -listallhardwareports` 确认接口。**只有确认接口为 `en10` 且地址未占用时**，才使用当时的临时配置示例：

```sh
sudo ifconfig en10 inet 192.168.123.100 netmask 255.255.255.0 alias
```

测试结束后移除本次添加的地址：

```sh
sudo ifconfig en10 inet 192.168.123.100 -alias
```

不要照抄其他电脑的网卡名称。机载依赖和已验证环境见 [README](../README.md#适用环境)；SDK 导入失败时先核对厂家 Python 环境，不自动升级 SDK 或固件。

## 2. 读取硬件参数与状态

在电脑创建本次记录目录；再次测试时换一个目录名，避免覆盖：

```sh
mkdir -p evidence/my-check

# 硬件清单：默认只枚举 CUDA，不运行计算负载
ssh unitree@192.168.123.164 'timeout 90 python3 -' \
  < scripts/read_compute_inventory.py > evidence/my-check/inventory.json

# 30 秒只读 LowState：关节状态、原始快照与统计
ssh unitree@192.168.123.164 'timeout 40 python3 - --seconds 30' \
  < scripts/read_lowstate.py > evidence/my-check/lowstate.jsonl
```

`timeout` 在机器人 Linux 上执行，电脑无需安装该命令。硬件清单中缺少工具、权限或超时属于证据缺口，不自动表示硬件损坏。LowState 的原始快照保留 SDK 提供的状态字段；本工具没有单独完成 IMU 精度、漂移或标定验证。

## 3. 看普通相机、深度图与雷达

确认头部相机的 USB 数据线连接到机载计算单元，然后在电脑运行：

```sh
python3 scripts/start_sensor_web.py --host 192.168.123.164 --name g1-02
```

1. 保留终端，打开启动器给出的地址，默认 <http://127.0.0.1:8765>。
2. 点击 **连接预览**，查看彩色画面、深度伪彩图和点云，核对相机身份、数据更新时间与雷达告警。
3. 按需要保存当前帧、预览点云 JSON 或检查摘要。
4. 点击 **停止**，等待采集停止与服务状态恢复；再回到终端按 `Ctrl+C`。

网页可临时启动头部视频服务和雷达驱动，停止时恢复原启停状态。不要同时运行其他占用相机或修改这些服务的测试。全部页面关闭后约 30 秒无心跳会自动停止采集；每次会话最长 15 分钟。断网时仍需核实机端实际状态。

深度下载保存的是伪彩图；点云下载是当前预览采样，不是完整录像。三路数据没有同步或标定。详见[网页说明](sensor-web.md)。

需要独立采集时，以下两个脚本从**已有运行中的厂家服务**读取，不负责启动服务；不要与网页同时运行。电脑端执行：

```sh
# RGB：图片保存在机器人 /tmp/g1-camera-check，统计回到电脑
ssh unitree@192.168.123.164 \
  'timeout 40 python3 - --seconds 20 --output-dir /tmp/g1-camera-check' \
  < scripts/camera_sample_check.py > evidence/my-check/camera.jsonl

# 点云：采样文件保存在机器人 /tmp/g1-lidar-check
ssh unitree@192.168.123.164 \
  'timeout 40 python3 - --seconds 20 --output-dir /tmp/g1-lidar-check' \
  < scripts/lidar_cloud_check.py > evidence/my-check/lidar.jsonl

# 取回各自输出；重复采集前更换远端和本地目录名
scp -r unitree@192.168.123.164:/tmp/g1-camera-check evidence/my-check/
scp -r unitree@192.168.123.164:/tmp/g1-lidar-check evidence/my-check/
```

这两个独立脚本使用当时的 `eth0` 接口。原始 Z16 深度采集需在 Linux 编译 `scripts/depth_frame_check.c`，并核对空闲的 D435i 节点；命令与边界见[传感器专项记录](../reports/sensors-2026-09-20.md#工具与复现)。不要猜测 `/dev/videoN` 编号。

## 4. 电机：先预检，再现场执行

先读[底层逐轴说明](ssh-lowlevel-joints.md)。底层工具适用于已吊装、调试模式且控制权满足要求的设备；上肢 `rt/arm_sdk` 是另一条路线，不能混用。

电脑端复制文件并登录：

```sh
ssh unitree@192.168.123.164 'umask 077; mkdir -p /tmp/g1-test-kit'
scp scripts/lowlevel_joint_check.py scripts/read_lowstate.py \
  unitree@192.168.123.164:/tmp/g1-test-kit/
ssh unitree@192.168.123.164
```

以下在**机器人终端**执行，均不发送动作：

```sh
cd /tmp/g1-test-kit
python3 lowlevel_joint_check.py --list-joints
python3 lowlevel_joint_check.py --device-id
python3 lowlevel_joint_check.py --joint left_elbow --delta-deg 1
```

主动操作需要将现场核对的开发板标识传给 `--expect-device-id`，并显式加 `--execute`。具体保持、往返、控制权释放及停止确认按[电机命令行指南](g1-cli-reuse.md)逐条执行，不能直接批量扫轴。`--hold-only --execute` 也会使能电机，不代表不会动。

出现 `blocked`、`abort` 或回位超限后保留日志，停止追加动作；不要删除身份、模式或保护检查。退出后核对新鲜遥测中的禁用状态和现场实物。历史主动控制并未完成整体验收，此次发布也没有重新运行实机动作。

## 5. 可选：计算功能短测

仅在机器人静止、散热正常且没有并发运动测试时，在电脑执行：

```sh
ssh unitree@192.168.123.164 'timeout 140 python3 - --seconds 10' \
  < scripts/compute_smoke.py > evidence/my-check/compute.jsonl

ssh unitree@192.168.123.164 'timeout --kill-after=5 690 python3 - --seconds 10' \
  < scripts/compute_int8_bench.py \
  > evidence/my-check/int8.jsonl 2> evidence/my-check/int8.stderr
```

这些命令会产生计算负载。程序有温度检查与时限；INT8 引擎构建还需要额外时间。计算结果只针对脚本中的固定负载，不能把依次运行的 GPU/DLA 分数相加，也不能当成持续峰值算力认证。详细解释见[计算报告](../reports/compute-capacity-2026-09-20.md)。

## 常见问题

| 现象 | 先检查什么 |
| --- | --- |
| SSH 无法连接 | 开发网口、当前 IP、电脑网卡及同网段地址 |
| SDK 无法导入 | 机载 Python 和 SDK 搜索路径；网页启动器会检查并传递已验证路径，独立脚本需自行确认环境 |
| 找不到深度相机 | USB 数据线、供电、是否被占用；不要只检查网线 |
| RGB 无画面 | 头部视频服务与实际 RGB 节点配置，不能用胸部视频代替 |
| 点云到达但有窗口告警 | 记录告警并现场检查；收到点云不表示告警消失 |
| 本地端口占用 | 启动器加 `--port 8766`；机上端口占用另加 `--remote-port 8766` |
| 网页显示旧数据或恢复失败 | 停止并核对网络、采集状态及厂家服务，不把冻结图像当作实时数据 |
| 电机预检拒绝执行 | 查看具体原因，核对吊装、模式、控制权与状态，不绕过检查 |

`evidence/` 已被 Git 忽略。记录可能包含硬件标识、现场图像和网络信息，分享前自行审阅；仓库中的历史报告只代表当时观察结果。
