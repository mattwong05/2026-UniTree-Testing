"""G1 arm_sdk: one bounded +5 degree right-elbow excursion and return.

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
ELBOW = 25
DELTA = math.radians(5)
DURATION = 16.0


def trajectory(t):
    """Return arm weight, elbow offset, phase. Max requested speed 2.5 deg/s."""
    if t < 0:
        raise ValueError("negative time")
    if t < 5:
        return t / 5, 0.0, "engage"
    if t < 8:
        u = (t - 5) / 3
        return 1.0, DELTA * (3*u*u - 2*u*u*u), "outbound"
    if t < 9:
        return 1.0, DELTA, "hold"
    if t < 12:
        u = (t - 9) / 3
        return 1.0, DELTA * (1 - 3*u*u + 2*u*u*u), "return"
    if t < 13:
        return 1.0, 0.0, "settle"
    if t < DURATION:
        return (DURATION - t) / 3, 0.0, "release"
    return 0.0, 0.0, "released"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
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
        if not 0.2 <= initial[ELBOW] <= 1.8:
            raise RuntimeError("right elbow outside conservative start range")
        print(json.dumps(dict(kind="preflight", execute=args.execute,
              fsm=802, initial_q=initial, target_joint=ELBOW,
              delta_deg=5, duration_s=DURATION)), flush=True)
        if not args.execute:
            return
        current()  # Recheck freshness after the read-only RPC.
        publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        publisher.Init()
        current()
        command = unitree_hg_msg_dds__LowCmd_()
        # Retain the metadata defaults used by the installed official arm example.
        for i in UPPER:
            command.motor_cmd[i].q = initial[i]
            command.motor_cmd[i].kp = 40.0
            command.motor_cmd[i].kd = 1.5
            command.motor_cmd[i].dq = 0.0
            command.motor_cmd[i].tau = 0.0
        start = previous = time.monotonic()
        held = list(initial)
        settled = False
        while True:
            now = time.monotonic()
            if now - previous > 0.1:
                raise RuntimeError("control loop delayed more than 100 ms")
            previous = now
            elapsed = now - start
            weight, offset, phase = trajectory(elapsed)
            msg = current()
            if phase == "engage":
                held = [motor.q for motor in msg.motor_state[:29]]
            elif not settled:
                held = [motor.q for motor in msg.motor_state[:29]]
                settled = True
            target = held[ELBOW] + offset
            for i in UPPER:
                desired = target if i == ELBOW else held[i]
                motor = msg.motor_state[i]
                if (abs(motor.q-desired) > 0.15 or abs(motor.dq) > 0.6 or
                        abs(motor.q-initial[i]) > 0.15):
                    print(json.dumps(dict(kind="guard_snapshot", index=i,
                          desired=desired, q=motor.q, dq=motor.dq,
                          tau_est=motor.tau_est, initial=initial[i],
                          elapsed_s=elapsed, phase=phase)), flush=True)
                    raise RuntimeError("joint %d tracking/velocity stop" % i)
                if abs(motor.tau_est) > (6 if i == ELBOW else 12):
                    raise RuntimeError("joint %d measured torque test stop" % i)
                command.motor_cmd[i].q = desired
                gain = (3*weight*weight - 2*weight*weight*weight) if phase == "engage" else 1.0
                command.motor_cmd[i].kp = 40.0 * gain
                command.motor_cmd[i].kd = 1.5 * gain
            if any(abs(msg.motor_state[i].q-initial[i]) > math.radians(10) for i in range(12)):
                raise RuntimeError("leg changed by more than 10 degrees")
            command.motor_cmd[ELBOW].q = target
            publish(weight)
            row = dict(kind="control", elapsed_s=elapsed, phase=phase,
                       weight=weight, desired=target, measured=msg.motor_state[ELBOW].q,
                       q=[m.q for m in msg.motor_state[:29]],
                       tau_est=msg.motor_state[ELBOW].tau_est,
                       temperature=msg.motor_state[ELBOW].temperature)
            samples.append(row)
            print(json.dumps(row), flush=True)
            if elapsed >= DURATION:
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
              measured_range_deg=math.degrees(max(r["measured"] for r in samples)-
                                               min(r["measured"] for r in samples)),
              final_error_deg=math.degrees(samples[-1]["measured"]-held[ELBOW]),
              displacement_from_initial_deg=math.degrees(samples[-1]["measured"]-initial[ELBOW]))), flush=True)


if __name__ == "__main__":
    main()
