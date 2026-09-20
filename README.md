# G1 二手设备现场测试

本目录保存 2026-09-20 的连接、计算功能和关节遥测测试工具。原始验收表保留不变，原始证据保存在 `evidence/2026-09-20/`，测试结论见 `reports/2026-09-20.md`。

当前已经验证网络连接、Jetson 计算功能和 29 轴遥测。主动手臂测试两次在接管阶段触发保护，未完成目标轨迹。**目前不能将主动控制或整机电机能力判定为验收通过。**

计算单元专项报告见 [100 TOPS 规格核验与 INT8 短测](reports/compute-capacity-2026-09-20.md)，现场 PDF 保存在 `output/pdf/G1_U2_计算单元能力报告_2026-09-20.pdf`。设备配置与原始 Orin NX 16 GB 标称 100 TOPS 相符；本次 GPU、双 DLA 的 INT8 短测及 GPU 稀疏测试通过有限自检，**未验证持续 100 TOPS**。

## 环境和连接

现场 Mac 为 Apple Silicon/macOS，USB 网卡为 AX88179B/en10。机器人计算模块为 `192.168.123.164`，用户 `unitree`。密码不放入脚本、文档或 Git。

仅当确认本次接口与网段仍一致时，可添加临时地址：

```sh
sudo ifconfig en10 inet 192.168.123.100 netmask 255.255.255.0 alias
ssh unitree@192.168.123.164
```

移除本次临时地址：

```sh
sudo ifconfig en10 inet 192.168.123.100 -alias
```

此操作不修改网络服务的持久 DHCP 配置、Wi-Fi 或默认网关。测试结束时保留临时地址供现场继续连接。

## 运行方式

脚本使用机器人已有 Python 3.8、Unitree SDK、Cyclone DDS、NumPy、TensorRT 8.5 和 CUDA 11.4。不在机器人安装依赖、不升级固件、不改变功率模式。脚本可经 SSH 标准输入运行，也可复制到机上临时目录执行。

只读关节记录，包含全采样统计和约 1 Hz 快照：

```sh
ssh unitree@192.168.123.164 'timeout 40 python3 - --seconds 30' \
  < scripts/read_lowstate.py > lowstate.jsonl
```

计算功能检查会产生短时负载，应在机器人静止、散热正常时运行：

```sh
ssh unitree@192.168.123.164 'timeout 140 python3 - --seconds 10' \
  < scripts/compute_smoke.py > compute.jsonl
```

CPU 检查是 8 个进程各校验 8 MiB 缓冲区；GPU/DLA 是固定小型 FP16 卷积与 NumPy 结果比较。它们不构成全内存检测、100 TOPS 测量或持续热稳定验收。计算脚本在读不到温度或温度达到保守的 75°C 测试阈值时退出。

INT8 吞吐短测使用已有 TensorRT，依次运行 GPU、DLA0、DLA1，每项约 10 秒（引擎构建另计）：

```sh
ssh unitree@192.168.123.164 'timeout --kill-after=5 690 python3 - --seconds 10' \
  < scripts/compute_int8_bench.py > int8-dense.jsonl 2> int8-dense.stderr

ssh unitree@192.168.123.164 'timeout --kill-after=5 230 python3 - --target gpu --sparse --seconds 10' \
  < scripts/compute_int8_bench.py > int8-sparse.jsonl 2> int8-sparse.stderr
```

保持机器人静止且散热正常，避免并发运动测试。`--target gpu|dla0|dla1` 可单独选目标，`--seconds` 限 1–30；`--sparse` 仅支持 GPU。程序每 0.5 秒检查可读温度，无有效读数或最高读数达到 75°C 时退出，个别不可读传感器会跳过；另有构建/运行时限和外部 timeout。Python 监控线程不是独立硬件保护。

JSONL 保存完整引擎信息、逐窗口计时和汇总，stderr 保存 TensorRT 提示。指标是 8 层固定 1×1 卷积的 dense-equivalent TOPS，一次乘加算 2 次运算，不额外对稀疏结果乘二。主指标排除构建、预热、传输及窗口间日志开销。正确性只覆盖常量输入特例；DLA 禁止 GPU fallback，但边界重排可用 GPU。各目标依次运行，不能将结果相加充当并发性能，也不能据此签署 100 TOPS 达标。详情、实际结果及官方来源见专项报告。

手臂脚本默认只做预检，不创建命令发布器：

```sh
ssh unitree@192.168.123.164 'timeout 12 python3 - --check-all' \
  < scripts/upper_body_check.py > arm-preflight.jsonl
```

原 `right_elbow_check.py` 已扩展并更名为 `upper_body_check.py`，可选择左右肩、肘、腕共 14 轴。默认相对偏移 +2°，允许正负 0.1–5°；`--check-all` 仅预检，不能与 `--execute` 合用。完整的 SSH 命令、关节名称和控制原理见 [SSH 上肢控制说明](docs/ssh-upper-body.md)。2026-09-20 的追加实机检查中，14 轴只读预检全部通过；未追加动作测试。

`--execute` 是主动控制入口，当前有未解决的实机接管现象，不能当作已通过验证的控制程序使用。脚本只发布 `rt/arm_sdk`，在接管期间跟随上身实测姿态，再保持接管结束时的姿态，使所选关节做一次小角度往返；不发布全身 `rt/lowcmd`，不切换运动模式。目标角度可按单关节指定，但接管权重作用于上身，其他关节仍参与保持，不能理解成只给一个电机上电。

该脚本绑定本机已观测的 `FSM 802 / mode_machine 5 / mode_pr 0` 与现有 `127.0.0.1:8080/status` 姿态服务。它不是通用 G1 控制器。原有姿态服务必须持续为 IDLE。低状态 CRC 错误、数据超过 100 ms、遥控按键、错误码、关节模式变化、温度达到 60°C、过大偏差/速度/估算力矩或下肢明显移动都会中止。阈值是本次短测的保守限制，不是厂家额定值。

退出时发送多条权重 0 释放消息并关闭本程序发布器。发布成功不等于机器人完成释放；必须用遥测与现场观察核实。硬件遥控停止仍是现场停止手段。

## 验证

```sh
python3 -m unittest discover -s tests -v
```

轨迹测试在本地运行且不需要 SDK，不连接机器人。设备端另做过只读采样、计算正确性检查、手臂预检和保护中止验证；详见现场报告。项目没有既有版本体系，本次不新增版本号，不发布或推送远端。
