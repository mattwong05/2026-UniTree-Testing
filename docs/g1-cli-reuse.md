# G1 吊装测试：命令行与换机复用

适用于已有 `unitree_sdk2py` 的 G1 29 轴机器，反馈必须为 `mode_machine=5 / mode_pr=0`。另一台需要重新确认型号、实物关节、吊装与活动空间，并在该机调试/底层模式验证现场停止。不得照搬上一台的起始角度、控制权状态或验收结论。

本套工具不安装依赖，不升级固件，不写校准值，不切换 PR/AB，也不提供自动扫全身动作。左右侧按机器人自身方向定义；同一个正负号在镜像关节上的实际空间方向应先小幅核对。

## 1. 准备与连接

复用包包含 `scripts/lowlevel_joint_check.py`（底层命令行）、`scripts/read_lowstate.py`（只读采样）、`scripts/compute_smoke.py`（短时计算检查）及本指南。解压后，从包根目录把脚本复制到目标机器人：

```sh
ssh unitree@192.168.123.164 'mkdir -p /tmp/g1-test-kit'
scp scripts/lowlevel_joint_check.py scripts/read_lowstate.py scripts/compute_smoke.py \
  unitree@192.168.123.164:/tmp/g1-test-kit/
ssh unitree@192.168.123.164
cd /tmp/g1-test-kit
python3 -c 'import unitree_sdk2py; print(unitree_sdk2py.__file__)'
ip -br addr
```

SDK 导入失败时先按该机厂家环境配置 Python；不要通过删除保护检查或更换消息类型来绕过。命令默认从机器人 `eth0` 收发 DDS，需要其他接口时使用 `--interface 接口名`。SSH 的 IP 是开发板地址，不能替代整机型号/序列号确认。

```sh
python3 lowlevel_joint_check.py --list-joints
python3 lowlevel_joint_check.py --device-id
```

`--device-id` 是当前开发板 `/etc/machine-id` 的 SHA256 前 16 位，不是整机序列号，也不保证克隆系统之间唯一。核对目标实物与 SSH 主机后，手动复制输出：

```sh
G1_DEVICE_ID='粘贴当前开发板标识'
mkdir -p logs/robot-b
```

换机器后重新核对并更新这个值，不要把上一台标识、IP 相同或 SSH 登录成功当成设备已确认。执行和释放命令都要求 `--expect-device-id "$G1_DEVICE_ID"` 匹配；普通只读预检不需要此参数。

## 2. 每台先做只读检查

```sh
python3 read_lowstate.py --seconds 10 > logs/robot-b/lowstate.jsonl
python3 lowlevel_joint_check.py --joint left_elbow --hold-shoulder --delta-deg 1
```

预检输出实际起始角度、目标、峰值目标速度、预计时长、需要使能的轴和开发板标识。默认不创建命令发布器。若显示 `motion ownership is not confirmed released`，先由现场操作员确认调试模式、全身禁用及吊装状态；随后可显式释放一次控制权：

```sh
python3 lowlevel_joint_check.py --release-motion --expect-device-id "$G1_DEVICE_ID"
```

释放命令自己仍会检查全身禁用、CRC、稳定状态、错误码和是否有其他命令流。它不发布电机动作、不自动重复释放。报错时查看 `blocked`/`abort` 原因，不能直接在运动模式强行运行；释放成功后重新预检。

如果机上存在本站姿态服务，必须返回 IDLE。没有该服务时允许明确的“连接被拒绝”；超时、损坏响应和非 IDLE 均拒绝启动。仍需保证没有其他 SDK/自动恢复程序抢写；启动时的无命令流检查不能保证未来独占。

## 3. 调整角度、方向、速度

| 参数 | 含义 |
| --- | --- |
| `--joint left_elbow` | 按名称选机器人左肘；右肘用 `right_elbow` |
| `--delta-deg -15` | 相对本次实测起点减少 15°，随后返回该起点；不是绝对角度 |
| `--speed-deg-s 1` | 目标速度上限，单位度/秒；允许 0.25–5，默认 1 |
| `--hold-shoulder` | 肘部测试时保持同侧三个肩轴，共四轴使能 |
| `--hold-only` | 只保持起点 3 秒，再禁用 |
| `--execute` | 真正发送动作命令；省略时只读 |

每次起点都会重新采样，保护中止后再次给同样的相对角度会得到不同绝对终点，必须重看预检。较大行程开始前由现场操作员核对整段路径与吊带/线缆间隙。

轨迹使用平滑起停。单程时间 `T=max(3, 1.5×abs(角度)/速度上限)` 秒；行程超过 2° 时，总时长为 `2T+5` 秒。15° / 1°/秒约 50 秒；60° / 5°/秒约 41 秒。小角度受到最短 3 秒单程限制，实际峰值目标速度可能低于设置值。目标速度不是实际速度保证：实际速度超过 0.3 rad/s（约 17.2°/秒）仍会停止。

本机反向 60° / 5°/秒已经在约 -6.12° 时因实际速度约 18.2°/秒中止，尚未完成行程。不要把这个参数当作已验证的稳定工作点，也不要通过提高增益或放宽保护来凑通过。以下仅演示该计划的**只读预检**：

```sh
python3 lowlevel_joint_check.py --joint left_elbow --hold-shoulder \
  --delta-deg -60 --speed-deg-s 5 --expect-device-id "$G1_DEVICE_ID"
```

## 4. 左右关节分别测试

同一时间只运行一个主动程序，完成一条后查看结果与现场观察，再决定下一条。新机器先用原位保持和小行程，不直接复现上一台的大行程。

```sh
# 左肩与左肘原位保持 3 秒
python3 lowlevel_joint_check.py --joint left_elbow --hold-shoulder --hold-only \
  --expect-device-id "$G1_DEVICE_ID" --execute

# 左肘反向 2°，目标速度上限 0.5°/秒
python3 lowlevel_joint_check.py --joint left_elbow --hold-shoulder \
  --delta-deg -2 --speed-deg-s 0.5 --expect-device-id "$G1_DEVICE_ID" --execute \
  > logs/robot-b/left-elbow.jsonl

# 右肘：同侧肩部保持，先原位检查
python3 lowlevel_joint_check.py --joint right_elbow --hold-shoulder --hold-only \
  --expect-device-id "$G1_DEVICE_ID" --execute
python3 lowlevel_joint_check.py --joint right_elbow --hold-shoulder \
  --delta-deg -2 --speed-deg-s 0.5 --expect-device-id "$G1_DEVICE_ID" --execute \
  > logs/robot-b/right-elbow.jsonl
```

可选主要关节共 16 个，具体索引与参考范围用 `--list-joints` 查看：

| 部位 | 左侧名称 | 右侧名称 | 本工具相对角度上限 |
| --- | --- | --- | --- |
| 肘 | `left_elbow` | `right_elbow` | 60° |
| 肩俯仰 | `left_shoulder_pitch` | `right_shoulder_pitch` | 2° |
| 肩侧摆 | `left_shoulder_roll` | `right_shoulder_roll` | 2° |
| 肩旋转 | `left_shoulder_yaw` | `right_shoulder_yaw` | 2° |
| 髋俯仰 | `left_hip_pitch` | `right_hip_pitch` | 2° |
| 髋侧摆 | `left_hip_roll` | `right_hip_roll` | 2° |
| 髋旋转 | `left_hip_yaw` | `right_hip_yaw` | 2° |
| 膝 | `left_knee` | `right_knee` | 2° |

肩、髋、膝每次仅使能目标轴，不能加 `--hold-shoulder`；其他关节没有保持力矩，可能机械随动。先预检，例如：

```sh
python3 lowlevel_joint_check.py --joint right_knee --delta-deg 1 --speed-deg-s 0.5
# 现场核对、预检通过后才加 --expect-device-id "$G1_DEVICE_ID" --execute
```

所有轴仍须满足绝对参考限位及 5° 裕量；工具不包含碰撞模型。踝、腰、腕没有纳入本次可复用的主要关节执行列表。软件支持不等于实机验证：目前现场明确确认的较大往返仅为本机左肩保持下的左肘 15° 方案，回位精度仍未达到本次标准。

## 5. 停止、记录与计算复用

现场停止为第一停止手段；程序检测任意遥控按键或 Ctrl+C 后尝试退出，并发送 20 帧全槽禁用，再确认 29 轴 mode=0。强制杀进程、系统崩溃或断网不能靠 Python 退出逻辑保证停机。不要用反复重跑来覆盖失败；按设备和尝试编号保留 JSONL。

- `HOLD_SEQUENCE_COMPLETE`：保持序列完成，不代表位移验收通过。
- `RESPONSE_WITHIN_TEST_LIMITS`：本次响应比 50%–150%、回位偏差不超过 0.5°，且退出禁用确认；仍需现场观察。
- `RESPONSE_OUTSIDE_TEST_LIMITS`：完成序列但响应指标未达标，退出码为 1。
- `abort`：保护中止，未完成动作。查看后续 `disable_feedback`；没有全轴禁用确认时由现场处理。

关节停止、机器人静止后，可复用既有计算功能短测，不要与主动关节测试并发：

```sh
timeout -k 5 140 python3 compute_smoke.py --seconds 10 > logs/robot-b/compute.jsonl
```

它需要该机现有 NumPy/CUDA/TensorRT 环境，只覆盖有限 CPU/GPU/DLA 自检，不能作为额定 TOPS 或长期热稳定结论。关节索引/增益参考[官方 SDK 示例](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/g1/low_level/g1_low_level_example.py)，机械范围参考[官方 29 轴 URDF](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof.urdf)，现场必须核对实际配置。
