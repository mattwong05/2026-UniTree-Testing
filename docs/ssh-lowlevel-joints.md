# 吊装状态下的底层逐轴检查

本工具适用于本次 G1 29 轴设备的调试模式。使用 `rt/lowcmd`，与 `upper_body_check.py` 的 `rt/arm_sdk` 路线不同。机器人必须稳固吊装、活动范围无人接触，现场操作员持有已在此底层模式验证有效的停止手段。

## 控制范围

`scripts/lowlevel_joint_check.py` 默认每次只使能一个指定关节，其他 28 个实际关节及协议空槽均明确设为 `mode=0`，不提供姿态保持力矩。默认仅预检；只有 `--execute` 才创建运动命令发布器。

可选 `--hold-left-shoulder` 仅用于左肘，会额外使能左肩 Pitch/Roll/Yaw（15/16/17），把三轴目标固定在预检时的实际角度，使用 Kp/Kd=40/1、估算力矩停止线 4 Nm。此时共有 4 轴使能，其余 25 个实际关节仍禁用，不能称为只有一个电机使能。三轴目标不会追随实时反馈漂移；它们同样必须在 500 ms 内反馈使能，位移/跟随偏差不超过 0.05 rad，温度、速度和退出保护保持有效。该方案用于评估肩部保持能否减少自由随动，不能承诺不会发生其他关节位移。

| 关节名称 | 索引 | Kp / Kd | 估算力矩停止线 |
| --- | --- | --- | --- |
| `left_elbow` / `right_elbow` | 18 / 25 | 40 / 1 | 4 Nm |
| `left_shoulder_pitch` / `right_shoulder_pitch` | 15 / 22 | 40 / 1 | 4 Nm |
| `left_knee` / `right_knee` | 3 / 9 | 100 / 2 | 6 Nm |

增益及命令字段参考[官方 G1 底层示例](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/g1/low_level/g1_low_level_example.py)。范围参考[官方 29 轴 URDF](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof.urdf)。这些不代替实物配置、碰撞检查或额定负载数据。

默认偏移 +1°，通常允许正负 0.25–2°。从实测角度开始保持 1 秒，3 秒移动到目标、保持 1 秒、3 秒返回、保持 1 秒，再发送全槽禁用命令。默认目标最大速度 0.5°/秒。`--hold-only` 则只保持原位 3 秒后禁用。

用户在两次 +1° 后明确要求左肘做 15° 可视动作，因此仅 `left_elbow` 允许最大正负 15°，其余轴仍限制为 2°。较大角度自动延长平滑轨迹，最高目标速度为 1°/秒，返回后保持 3 秒；15° 总时长约 50 秒。增益、估算力矩停止线、跟随误差及其他关节移动限制均保持不变。单轴方案在左肘约 +5.44° 时因未使能左肩随动越限中止；后续肩部保持方案完整执行约 50 秒，反馈增量约 13.65°，但回位偏差约 0.96° 超过判定线。详见 [实测报告](../reports/lowlevel-joints-2026-09-20.md)。

## 控制权与前置检查

调试模式下 `MotionSwitcher.CheckMode()` 仍可能显示 `name=ai`。官方底层示例会释放该运控程序；本轮在确认 29 轴均已禁用后，只调用一次 `ReleaseMode()`，随后另行确认 `name` 为空。机上 SDK 的返回值是 `(0, None)`，不能将整个元组与整数 0 比较来判断失败，也不能因此重复发送释放请求。

逐轴脚本本身不会释放或启动运控。它要求：

- 控制管理器查询成功且运控名称为空；原有姿态服务为 IDLE。
- `mode_machine=5 / mode_pr=0`，29 轴初始均为 mode=0、无错误、温度低于 60°C。
- CRC 正确、状态不超过 100 ms、遥控没有按键、全部关节稳定。
- 启动前没有观察到其他 `rt/lowcmd` 或 `rt/arm_sdk` 命令。

执行过程中，任何遥控按键会触发本程序中止；这只是软件保护，不能替代现场独立停止。全身速度超过 0.3 rad/s、未测轴移动超过 0.05 rad、目标轴相对起点移动超过 `max(0.08 rad, 指令偏移绝对值 + 0.05 rad)`、目标跟随偏差超过 0.05 rad、超过估算力矩限制或控制循环延迟超过 50 ms 都会中止。目标轴须在 500 ms 内反馈 mode=1，其余轴必须持续为 mode=0。收到其他上身控制器命令也会中止。

文件锁只防止本工具多开；启动时的无其他命令检查不能保证后续没有其他底层程序抢写。现场不得同时运行其他 SDK、姿态动作或自动恢复运控的程序。

## SSH 使用

文件放在机器人 `/tmp/g1-acceptance-01a0bdf8/lowlevel_joint_check.py`。临时目录可能在重启后消失；可从 Mac 的项目目录复制：

```sh
scp scripts/lowlevel_joint_check.py unitree@192.168.123.164:/tmp/g1-acceptance-01a0bdf8/
ssh unitree@192.168.123.164
cd /tmp/g1-acceptance-01a0bdf8

# 只读预检
python3 lowlevel_joint_check.py --joint left_elbow --hold-only

# 以下会实际使能一个电机
python3 lowlevel_joint_check.py --joint left_elbow --hold-only --execute
python3 lowlevel_joint_check.py --joint left_elbow --delta-deg 1 --execute
```

一次执行退出后再评估是否进入下一轴，不提供自动扫全身或失败重试功能。不要在未吊装时运行，也不要让另一人扶着正在使能的肢体“辅助通过”。

可选的左肩保持方案应先预检、再做 3 秒原位保持；只有保持与禁用检查完成且现场无异常，才进入慢速肘部往返：

```sh
python3 lowlevel_joint_check.py --joint left_elbow --hold-left-shoulder --delta-deg 15
python3 lowlevel_joint_check.py --joint left_elbow --hold-left-shoulder --hold-only --execute
python3 lowlevel_joint_check.py --joint left_elbow --hold-left-shoulder --delta-deg 15 --execute
```

最后一条计划约 50 秒、最高目标速度 1°/秒。手动辨识之后必须先让人员离开活动范围，不能在有人扶着手臂时开始保持或运动。

## 停止与判定边界

正常结束、保护中止及可处理的终止信号都会尝试发送 20 帧全槽 `mode=0、Kp=Kd=tau=dq=0`，再关闭发布器。随后须读到新鲜状态，确认 29 轴 mode=0；没有确认会以失败退出。网络中断、进程强制终止或系统崩溃时不能靠这个退出代码保证停机。测试完成后不自动启动原来的 `ai` 运控程序。

`RESPONSE_WITHIN_TEST_LIMITS` 只表示本次小角度遥测响应满足：实际往返平台差为指令的 50%–150%，返回保持位置与起始保持位置相差不超过 0.5°，且保护没有触发、退出禁用反馈得到确认。结果仍需现场确认异响、抖动、卡滞及实际动作。它不是电机额定扭矩、负载能力、完整活动范围或采购验收通过的证明。

测量平台使用使能后的保持反馈作为比较基准：起始 0.75–1 秒、目标保持最后 0.5 秒、回位保持最后 0.5 秒；窗口随轨迹时长计算。`HOLD_SEQUENCE_COMPLETE` 仅记录保持与退出完成，不算位移能力测试。出现 `abort` 或 `RESPONSE_OUTSIDE_TEST_LIMITS` 后保存日志并停止追加动作，评估日志及现场观察后再决定后续检查。明确要求的重复测试仍独立记录，不能覆盖先前失败，也不通过放宽判定线来凑通过结果。

单轴使能不等于只会有该轴发生机械运动：其他轴禁用后没有保持力矩，肘部移动可能带动肩部。此前单轴 15° 尝试没有放宽旁轴保护，也没有给肩部上电。可选的肩部保持方案必须单列记录，不能沿用单电机使能结论；若仍触发保护，应保存数据并停止，不能扩大限制来凑通过。

如果反馈角度变化而现场明确没有观察到动作，应先核对实物与反馈的对应关系。机器人左侧按机器人自身方向定义，正面对着机器人时位于观察者右侧。确认全部关节禁用后，使用只读 `read_lowstate.py` 记录，由现场操作员轻缓手动屈伸指定关节、观察真实动作；比较全轴变化与官方索引，并记录操作员确实执行的时间和动作。有明显阻力时停止手动辨识。仅有静止数据或未收到操作员完成反馈，不能宣称关节编号、反馈来源或机械动作已确认，也不能靠继续扩大命令角度排除这些疑点。
