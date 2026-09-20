"""G1 arm_sdk: one selected arm joint, a bounded excursion, then return.

Dry run by default. Physical execution requires a supported suspended robot,
clear arm space, a trained observer and a previously verified remote stop.
Never publishes rt/lowcmd or changes the locomotion mode.
"""
import argparse
import json
import math
import signal
import threading
import time
import urllib.request

UPPER = tuple(range(12, 29))
DELTA = math.radians(2)
DURATION = 16.0
HOLD_DURATION = 9.0
KP = 40.0
KD = 1.5
# Reference bounds from Unitree's g1_29dof.urdf; not a measured collision envelope.
# https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof.urdf
JOINTS = {
    "left_shoulder_pitch": (15, -3.0892, 2.6704),
    "left_shoulder_roll": (16, -1.5882, 2.2515),
    "left_shoulder_yaw": (17, -2.618, 2.618),
    "left_elbow": (18, -1.0472, 2.0944),
    "left_wrist_roll": (19, -1.972222054, 1.972222054),
    "left_wrist_pitch": (20, -1.614429558, 1.614429558),
    "left_wrist_yaw": (21, -1.614429558, 1.614429558),
    "right_shoulder_pitch": (22, -3.0892, 2.6704),
    "right_shoulder_roll": (23, -2.2515, 1.5882),
    "right_shoulder_yaw": (24, -2.618, 2.618),
    "right_elbow": (25, -1.0472, 2.0944),
    "right_wrist_roll": (26, -1.972222054, 1.972222054),
    "right_wrist_pitch": (27, -1.614429558, 1.614429558),
    "right_wrist_yaw": (28, -1.614429558, 1.614429558),
}


def validate_target(name, q, delta):
    if not math.isfinite(delta) or not math.radians(0.1) <= abs(delta) <= math.radians(5):
        raise ValueError("offset must have magnitude 0.1 to 5 degrees")
    _, lower, upper = JOINTS[name]
    margin = math.radians(5)
    if not all(math.isfinite(v) and lower+margin <= v <= upper-margin for v in [q, q+delta]):
        raise ValueError("start/target outside reference joint bounds with 5-degree margin")
    return q + delta


def trajectory(t, delta=DELTA, hold_only=False):
    """Return arm weight, joint offset, phase; <=2.5 deg/s for allowed offsets."""
    if t < 0:
        raise ValueError("negative time")
    if t < 5:
        return t / 5, 0.0, "engage"
    if hold_only:
        if t < 6:
            return 1.0, 0.0, "hold_baseline"
        if t < HOLD_DURATION:
            return (HOLD_DURATION-t) / 3, 0.0, "release"
        return 0.0, 0.0, "released"
    if t < 8:
        u = (t - 5) / 3
        return 1.0, delta * (3*u*u - 2*u*u*u), "outbound"
    if t < 9:
        return 1.0, delta, "hold"
    if t < 12:
        u = (t - 9) / 3
        return 1.0, delta * (1 - 3*u*u + 2*u*u*u), "return"
    if t < 13:
        return 1.0, 0.0, "settle"
    if t < DURATION:
        return (DURATION - t) / 3, 0.0, "release"
    return 0.0, 0.0, "released"


def command_targets(anchor, selected_index, offset):
    """Keep non-selected joints at the fixed pre-takeover reference."""
    return {i: anchor[i] + (offset if i == selected_index else 0.0) for i in UPPER}


def motion_violations(q, dq, desired, initial):
    """Keep the same guard limits, but distinguish their causes in the log."""
    checks = (("tracking_error_rad", abs(q-desired), 0.15),
              ("velocity_rad_s", abs(dq), 0.6),
              ("displacement_rad", abs(q-initial), 0.15))
    return [dict(metric=name, observed=value, limit=limit)
            for name, value, limit in checks if value > limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--joint", choices=JOINTS, default="right_elbow")
    parser.add_argument("--delta-deg", type=float, default=2.0)
    parser.add_argument("--list-joints", action="store_true")
    parser.add_argument("--check-all", action="store_true", help="read-only preflight of all 14 arm joints")
    parser.add_argument("--hold-only", action="store_true", help="with --execute, take over and release without a joint excursion")
    args = parser.parse_args()
    if args.list_joints:
        print(json.dumps({name: spec[0] for name, spec in JOINTS.items()}, indent=2))
        return
    if args.check_all and args.execute:
        parser.error("--check-all cannot be combined with --execute")
    if args.check_all and args.hold_only:
        parser.error("--check-all cannot be combined with --hold-only")
    delta = math.radians(args.delta_deg)
    if not math.isfinite(delta) or not math.radians(0.1) <= abs(delta) <= math.radians(5):
        parser.error("--delta-deg magnitude must be 0.1 to 5")
    joint_index = JOINTS[args.joint][0]
    duration = HOLD_DURATION if args.hold_only else DURATION
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    from unitree_sdk2py.g1.loco.g1_loco_api import ROBOT_API_ID_LOCO_GET_FSM_ID

    ChannelFactoryInitialize(0, "eth0")
    # On this installed SDK, querying after stream startup timed out. Query first;
    # a failed query still blocks control, without changing the robot mode.
    client = LocoClient()
    client.SetTimeout(3)
    client.Init()
    code, data = client._Call(ROBOT_API_ID_LOCO_GET_FSM_ID, "{}")
    if code != 0 or json.loads(data)["data"] != 802:
        raise RuntimeError("expected existing FSM 802; no mode change attempted; RPC=%r data=%r" % (code, data))
    crc = CRC()
    lock = threading.Lock()
    state = dict(msg=None, received=0.0, fault=None)

    def latch(reason):
        with lock:
            if state["fault"] is None:
                state["fault"] = reason

    def receive(msg):
        try:
            if crc.Crc(msg) != msg.crc:
                latch("LowState CRC mismatch")
                return
            if msg.mode_machine != 5 or msg.mode_pr != 0:
                latch("unexpected machine/PR mode")
            if len(msg.wireless_remote) < 4 or any(msg.wireless_remote[2:4]):
                latch("remote button pressed or remote data missing")
            for i, motor in enumerate(msg.motor_state[:29]):
                if motor.motorstate != 0 or motor.mode != 1:
                    latch("joint %d error or mode changed" % i)
                if max(motor.temperature) >= 60:
                    latch("joint %d reached conservative 60 C test stop" % i)
                if not all(math.isfinite(v) for v in [motor.q, motor.dq, motor.tau_est, motor.vol]):
                    latch("nonfinite telemetry")
            with lock:
                state["msg"], state["received"] = msg, time.monotonic()
        except Exception as exc:
            latch("telemetry callback failed: " + str(exc))

    def check_pose_server():
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open("http://127.0.0.1:8080/status", timeout=0.3) as response:
            status = json.load(response)
        if status.get("state") != "IDLE":
            raise RuntimeError("existing pose server is not IDLE")

    check_pose_server()
    stop_observer = threading.Event()

    def observe_pose_server():
        while not stop_observer.wait(0.5):
            try:
                check_pose_server()
            except Exception as exc:
                latch("pose server guard: " + str(exc))

    observer = threading.Thread(target=observe_pose_server, daemon=True)
    observer.start()

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(receive, 1)
    publisher = None
    command = None
    samples = []
    aborted = None
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, lambda *_: latch("termination signal"))

    def current():
        with lock:
            msg, received, fault = state["msg"], state["received"], state["fault"]
        if fault:
            raise RuntimeError(fault)
        if msg is None or time.monotonic() - received > 0.1:
            raise RuntimeError("LowState absent or older than 100 ms")
        return msg

    def publish(weight):
        command.motor_cmd[29].q = weight
        command.crc = crc.Crc(command)
        if publisher.Write(command) is False:
            raise RuntimeError("DDS write failed")

    try:
        deadline = time.monotonic() + 3
        while state["msg"] is None and time.monotonic() < deadline:
            time.sleep(0.02)
        baseline = current()
        initial = [motor.q for motor in baseline.motor_state[:29]]
        stable_until = time.monotonic() + 1
        while time.monotonic() < stable_until:
            msg = current()
            if any(abs(msg.motor_state[i].q-initial[i]) > 0.025 or
                   abs(msg.motor_state[i].dq) > 0.15 for i in UPPER):
                raise RuntimeError("upper body must be still before starting")
            time.sleep(0.02)
        if args.check_all:
            results = []
            for name, (index, _, _) in JOINTS.items():
                reason = None
                try:
                    target = validate_target(name, initial[index], delta)
                except ValueError as exc:
                    reason, target = str(exc), None
                motor = msg.motor_state[index]
                results.append(dict(joint=name, index=index, q_deg=math.degrees(initial[index]),
                               dq_rad_s=motor.dq, temperature=list(motor.temperature),
                               motorstate=motor.motorstate, planned_q_deg=math.degrees(target) if target is not None else None,
                               preflight="PASS" if reason is None else "BLOCKED", reason=reason))
            print(json.dumps(dict(kind="all_joint_preflight", execute=False, delta_deg=args.delta_deg,
                                  fsm=802, joints=results)), flush=True)
            if any(row["reason"] for row in results):
                raise RuntimeError("one or more joint target preflights blocked")
            return
        validate_target(args.joint, initial[joint_index], delta)
        print(json.dumps(dict(kind="preflight", execute=args.execute,
              fsm=802, initial_q=initial, target_joint=joint_index, joint=args.joint,
              delta_deg=0.0 if args.hold_only else args.delta_deg, hold_only=args.hold_only,
              kp=KP, kd=KD, reference="fixed_pre_takeover", duration_s=duration)), flush=True)
        if not args.execute:
            return
        baseline = current()  # Refresh the fixed reference before creating a writer.
        initial = [motor.q for motor in baseline.motor_state[:29]]
        validate_target(args.joint, initial[joint_index], delta)
        publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        publisher.Init()
        current()
        command = unitree_hg_msg_dds__LowCmd_()
        # Retain the metadata defaults used by the installed official arm example.
        for i in UPPER:
            command.motor_cmd[i].q = initial[i]
            command.motor_cmd[i].kp = KP
            command.motor_cmd[i].kd = KD
            command.motor_cmd[i].dq = 0.0
            command.motor_cmd[i].tau = 0.0
        start = previous = time.monotonic()
        held = list(initial)
        last_sent = None
        while True:
            now = time.monotonic()
            if now - previous > 0.1:
                raise RuntimeError("control loop delayed more than 100 ms")
            previous = now
            elapsed = now - start
            weight, offset, phase = trajectory(elapsed, delta, args.hold_only)
            msg = current()
            targets = command_targets(held, joint_index, offset)
            target = targets[joint_index]
            for i in UPPER:
                desired = targets[i]
                motor = msg.motor_state[i]
                violations = motion_violations(motor.q, motor.dq, desired, initial[i])
                if violations:
                    name = next((name for name, spec in JOINTS.items() if spec[0] == i), "waist_%d" % i)
                    print(json.dumps(dict(kind="guard_snapshot", index=i,
                          joint=name, target_joint=args.joint, violations=violations,
                          desired=desired, q=motor.q, dq=motor.dq,
                          tau_est=motor.tau_est, initial=initial[i],
                          elapsed_s=elapsed, phase=phase, last_sent=last_sent)), flush=True)
                    raise RuntimeError("%s (%d): %s during %s; requested joint %s" %
                                       (name, i, ", ".join(v["metric"] for v in violations), phase, args.joint))
                torque_limit = 2 if i in (20, 21, 27, 28) else (6 if i == joint_index else 12)
                if abs(motor.tau_est) > torque_limit:
                    raise RuntimeError("joint %d measured torque test stop" % i)
                command.motor_cmd[i].q = desired
                # The SDK weight handles the transition. Do not also fade the
                # posture-holding gains to zero or chase the measured position.
                command.motor_cmd[i].kp = KP
                command.motor_cmd[i].kd = KD
            if any(abs(msg.motor_state[i].q-initial[i]) > math.radians(10) for i in range(12)):
                raise RuntimeError("leg changed by more than 10 degrees")
            command.motor_cmd[joint_index].q = target
            publish(weight)
            last_sent = dict(elapsed_s=elapsed, weight=weight, kp=KP, kd=KD,
                             upper_q=[targets[i] for i in UPPER])
            row = dict(kind="control", elapsed_s=elapsed, phase=phase,
                       hold_only=args.hold_only, kp=KP, kd=KD,
                       commanded_upper_q=last_sent["upper_q"],
                       weight=weight, desired=target, measured=msg.motor_state[joint_index].q,
                       q=[m.q for m in msg.motor_state[:29]],
                       tau_est=msg.motor_state[joint_index].tau_est,
                       temperature=msg.motor_state[joint_index].temperature)
            samples.append(row)
            print(json.dumps(row), flush=True)
            if elapsed >= duration:
                break
            time.sleep(max(0, 0.02-(time.monotonic()-now)))
    except Exception as exc:
        aborted = str(exc)
        print(json.dumps(dict(kind="abort", reason=aborted)), flush=True)
    finally:
        if publisher is not None:
            if command is not None:
                for _ in range(15):
                    try:
                        publish(0.0)
                    except Exception as exc:
                        print(json.dumps(dict(kind="release_error", reason=str(exc))), flush=True)
                        aborted = aborted or "arm release write failed"
                    time.sleep(0.02)
                print(json.dumps(dict(kind="release_sent", arm_weight=0.0)), flush=True)
            publisher.Close()
        stop_observer.set()
        observer.join(timeout=1)
        sub.Close()
    if aborted:
        raise SystemExit(1)
    if samples:
        print(json.dumps(dict(kind="completed", control_samples=len(samples),
              hold_only=args.hold_only, physical_acceptance="not_established",
              measured_range_deg=math.degrees(max(r["measured"] for r in samples)-
                                               min(r["measured"] for r in samples)),
              final_error_deg=math.degrees(samples[-1]["measured"]-held[joint_index]),
              displacement_from_initial_deg=math.degrees(samples[-1]["measured"]-initial[joint_index]))), flush=True)


if __name__ == "__main__":
    main()
