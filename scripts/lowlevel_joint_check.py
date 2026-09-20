"""Suspended G1: move one joint on rt/lowcmd, then disable all commands.

Read-only unless --execute is supplied. Requires released motion ownership,
all 29 joints initially disabled, a trained observer and a verified stop for
this low-level mode. Only --release-motion releases ownership; it never moves.
Never changes PR/AB mode or walks.
The optional --hold-shoulder enables the elbow's three shoulder joints at fixed angles.
"""
import argparse
import fcntl
import hashlib
import json
import math
import signal
import threading
import time
from pathlib import Path
import urllib.error
import urllib.request

# Index, reference URDF limits, official example gains, conservative test torque.
# Limits are not a collision model or a proof of installed mechanical limits.
JOINTS = {
    "left_elbow": (18, -1.0472, 2.0944, 40., 1., 4.),
    "right_elbow": (25, -1.0472, 2.0944, 40., 1., 4.),
    "left_shoulder_pitch": (15, -3.0892, 2.6704, 40., 1., 4.),
    "right_shoulder_pitch": (22, -3.0892, 2.6704, 40., 1., 4.),
    "left_shoulder_roll": (16, -1.5882, 2.2515, 40., 1., 4.),
    "right_shoulder_roll": (23, -2.2515, 1.5882, 40., 1., 4.),
    "left_shoulder_yaw": (17, -2.618, 2.618, 40., 1., 4.),
    "right_shoulder_yaw": (24, -2.618, 2.618, 40., 1., 4.),
    "left_hip_pitch": (0, -2.5307, 2.8798, 60., 1., 6.),
    "right_hip_pitch": (6, -2.5307, 2.8798, 60., 1., 6.),
    "left_hip_roll": (1, -0.5236, 2.9671, 60., 1., 6.),
    "right_hip_roll": (7, -2.9671, 0.5236, 60., 1., 6.),
    "left_hip_yaw": (2, -2.7576, 2.7576, 60., 1., 6.),
    "right_hip_yaw": (8, -2.7576, 2.7576, 60., 1., 6.),
    "left_knee": (3, -0.087267, 2.8798, 100., 2., 6.),
    "right_knee": (9, -0.087267, 2.8798, 100., 2., 6.),
}
LEFT_SHOULDER_LIMITS = {15: (-3.0892, 2.6704), 16: (-1.5882, 2.2515), 17: (-2.618, 2.618)}
RIGHT_SHOULDER_LIMITS = {22: (-3.0892, 2.6704), 23: (-2.2515, 1.5882), 24: (-2.618, 2.618)}
PERIOD = .002
HOLD_DURATION = 3.


def emit(kind, **fields):
    print(json.dumps(dict(kind=kind, **fields)), flush=True)


def validate_speed(speed_deg_s):
    if not math.isfinite(speed_deg_s) or not .25 <= speed_deg_s <= 5.:
        raise ValueError('speed must be 0.25 to 5 degrees/second')


def ramp_duration(delta, speed_deg_s=1.):
    # The speed option is a ceiling; retain a minimum 3-second ramp.
    validate_speed(speed_deg_s)
    return max(3., 1.5*abs(math.degrees(delta))/speed_deg_s)


def trajectory_duration(delta, hold_only=False, speed_deg_s=1.):
    if hold_only:
        return HOLD_DURATION
    return 2.+2.*ramp_duration(delta, speed_deg_s)+(3. if abs(delta) > math.radians(2) else 1.)


def target_offset(t, delta, hold_only=False, speed_deg_s=1.):
    if not math.isfinite(t) or t < 0:
        raise ValueError("invalid trajectory time")
    if hold_only:
        return 0., "hold_initial"
    if t < 1:
        return 0., "hold_initial"
    ramp = ramp_duration(delta, speed_deg_s)
    if t < 1+ramp:
        u = (t-1)/ramp
        return delta*(3*u*u-2*u*u*u), "outbound"
    if t < 2+ramp:
        return delta, "hold_target"
    if t < 2+2*ramp:
        u = (t-2-ramp)/ramp
        return delta*(1-3*u*u+2*u*u*u), "return"
    return 0., "hold_return"


def max_delta_deg(name):
    return 60 if name in ('left_elbow', 'right_elbow') else 2


def validate_delta(name, delta):
    maximum = max_delta_deg(name)
    if not math.isfinite(delta) or not math.radians(.25) <= abs(delta) <= math.radians(maximum):
        raise ValueError("%s offset magnitude must be 0.25 to %d degrees" % (name, maximum))


def validate_plan(name, q, delta, hold_only=False):
    validate_delta(name, delta)
    _, lower, upper, *_ = JOINTS[name]
    end = q if hold_only else q+delta
    margin = math.radians(5)
    if not all(math.isfinite(v) and lower+margin <= v <= upper-margin for v in (q, end)):
        raise ValueError("start/target exceeds reference bounds with 5-degree margin")


def validate_holds(holds):
    if not holds:
        return
    limits = next((group for group in (LEFT_SHOULDER_LIMITS, RIGHT_SHOULDER_LIMITS)
                   if set(holds) == set(group)), None)
    if limits is None:
        raise ValueError('shoulder hold must contain exactly one side: 15/16/17 or 22/23/24')
    margin = math.radians(5)
    for index, q in holds.items():
        lower, upper = limits[index]
        if not math.isfinite(q) or not lower+margin <= q <= upper-margin:
            raise ValueError('shoulder hold target outside reference bounds')


def set_command(command, name, q, enabled, holds=None):
    """Clear every slot before enabling the selected joint and optional holds."""
    for motor in command.motor_cmd:
        motor.mode = 0
        motor.q = motor.dq = motor.tau = motor.kp = motor.kd = 0.
    command.mode_machine = 5
    command.mode_pr = 0
    if enabled:
        index, _, _, kp, kd, _ = JOINTS[name]
        motor = command.motor_cmd[index]
        motor.mode, motor.q, motor.kp, motor.kd = 1, q, kp, kd
        if holds:
            if set(holds) != set(shoulder_indices(name)):
                raise ValueError('shoulder hold must match the selected elbow side')
            validate_holds(holds)
            for held, target in holds.items():
                motor = command.motor_cmd[held]
                motor.mode, motor.q, motor.kp, motor.kd = 1, target, 40., 1.


def validate_state(msg, selected=None, ignore_remote=False, held_indices=()):
    if msg.mode_machine != 5 or msg.mode_pr != 0 or len(msg.motor_state) < 29:
        raise RuntimeError("unexpected robot/PR mode or missing axes")
    if not ignore_remote and (len(msg.wireless_remote) < 4 or any(msg.wireless_remote[2:4])):
        raise RuntimeError("remote button pressed or remote data unavailable")
    for index, motor in enumerate(msg.motor_state[:29]):
        allowed = (0, 1) if index == selected or index in held_indices else (0,)
        if motor.mode not in allowed:
            raise RuntimeError("joint %d unexpectedly enabled or changed mode" % index)
        if motor.motorstate != 0:
            raise RuntimeError("joint %d motor error %s" % (index, motor.motorstate))
        if not all(math.isfinite(v) for v in (motor.q, motor.dq, motor.tau_est, motor.vol)):
            raise RuntimeError("joint %d nonfinite state" % index)
        if not motor.temperature or max(motor.temperature) >= 60 or not 40 <= motor.vol <= 60:
            raise RuntimeError("joint %d temperature/voltage test limit" % index)


class MotionStop(RuntimeError):
    """Keep the triggering sample; periodic control logs may miss it."""
    def __init__(self, reason, **snapshot):
        super().__init__(reason)
        self.snapshot = snapshot


def validate_motion(msg, anchor, selected, desired, torque_limit, delta=0., holds=None):
    for index, motor in enumerate(msg.motor_state[:29]):
        travel_limit = max(.08, abs(delta)+.05) if index == selected else .05
        violations = []
        if abs(motor.q-anchor[index]) > travel_limit:
            violations.append('displacement')
        if abs(motor.dq) > .3:
            violations.append('velocity')
        if violations:
            raise MotionStop('joint %d %s stop' % (index, '/'.join(violations)),
                index=index, violations=violations, q_rad=motor.q,
                travel_deg=math.degrees(motor.q-anchor[index]),
                travel_limit_deg=math.degrees(travel_limit), dq_rad_s=motor.dq,
                velocity_limit_rad_s=.3, tau_est_nm=motor.tau_est)
    targets = [(selected, desired, torque_limit, 'selected joint')]
    targets += [(held, target, 4., 'held shoulder joint') for held, target in (holds or {}).items()]
    for index, target, limit, scope in targets:
        motor = msg.motor_state[index]
        error = motor.q-target
        violations = []
        if abs(error) > .05:
            violations.append('tracking')
        if abs(motor.tau_est) > limit:
            violations.append('estimated_torque')
        if violations:
            raise MotionStop('%s %d %s stop' % (scope, index, '/'.join(violations)),
                index=index, violations=violations, q_rad=motor.q, desired_rad=target,
                tracking_error_deg=math.degrees(error), tracking_limit_deg=math.degrees(.05),
                travel_deg=math.degrees(motor.q-anchor[index]), dq_rad_s=motor.dq,
                tau_est_nm=motor.tau_est, estimated_torque_limit_nm=limit)


def response_metrics(samples, delta, hold_only, speed_deg_s=1.):
    if hold_only:
        return dict(scope="hold_only", verdict="HOLD_SEQUENCE_COMPLETE")
    def average(start, end):
        values = [q for t, q in samples if start <= t <= end]
        if not values:
            raise RuntimeError("insufficient plateau feedback for response assessment")
        return sum(values)/len(values)
    target_end = 2.+ramp_duration(delta, speed_deg_s)
    end = trajectory_duration(delta, speed_deg_s=speed_deg_s)
    initial, target, returned = average(.75, 1.), average(target_end-.5, target_end), average(end-.5, end)
    ratio = (target-initial)/delta
    return_error = math.degrees(returned-initial)
    passed = .5 <= ratio <= 1.5 and abs(return_error) <= .5
    return dict(scope="unloaded_joint_excursion", measured_excursion_deg=math.degrees(target-initial),
                requested_excursion_deg=math.degrees(delta), response_ratio=ratio,
                return_error_deg=return_error, verdict="RESPONSE_WITHIN_TEST_LIMITS" if passed else "RESPONSE_OUTSIDE_TEST_LIMITS")


def shoulder_indices(name):
    if name == 'left_elbow':
        return tuple(LEFT_SHOULDER_LIMITS)
    if name == 'right_elbow':
        return tuple(RIGHT_SHOULDER_LIMITS)
    raise ValueError('shoulder hold requires left_elbow or right_elbow')


def device_fingerprint():
    raw = Path('/etc/machine-id').read_text().strip()
    if not raw:
        raise RuntimeError('computer identity unavailable')
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def probe_pose_service():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open('http://127.0.0.1:8080/status', timeout=.3) as response:
            if json.load(response).get('state') != 'IDLE':
                raise RuntimeError('existing pose server is not IDLE')
        return 'IDLE'
    except urllib.error.URLError as exc:
        # A missing site-specific server is portable; an unresponsive one is not.
        if isinstance(exc.reason, ConnectionRefusedError):
            return 'not_listening'
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", choices=JOINTS, default="left_elbow")
    parser.add_argument("--delta-deg", type=float, default=1.)
    parser.add_argument("--speed-deg-s", type=float, default=1., help="target speed ceiling, 0.25 to 5 degrees/s")
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--list-joints", action="store_true")
    parser.add_argument("--device-id", action="store_true", help="print computer fingerprint without SDK initialization")
    parser.add_argument("--expect-device-id", help="required identity match before execute/release")
    parser.add_argument("--release-motion", action="store_true", help="release motion owner once after disabled-state checks; no movement")
    parser.add_argument("--hold-only", action="store_true")
    parser.add_argument("--hold-left-shoulder", action="store_true",
                        help="legacy left-elbow-only spelling of --hold-shoulder")
    parser.add_argument("--hold-shoulder", action="store_true", help="hold the selected elbow's three shoulder axes")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.list_joints:
        print('joint                      index  max_offset_deg  reference_min_deg  reference_max_deg')
        for name, (idx, lower, upper, *_) in JOINTS.items():
            print('%-27s %3d %15g %18.2f %18.2f' % (name, idx, max_delta_deg(name), math.degrees(lower), math.degrees(upper)))
        return
    if args.device_id:
        print(device_fingerprint())
        return
    if args.execute and args.release_motion:
        parser.error('--release-motion and --execute are separate operations')
    if (args.execute or args.release_motion) and not args.expect_device_id:
        parser.error('--expect-device-id is required; inspect this device with --device-id first')
    if args.hold_left_shoulder and args.joint != 'left_elbow':
        parser.error('--hold-left-shoulder requires --joint left_elbow')
    delta = math.radians(args.delta_deg)
    try:
        held_indices = shoulder_indices(args.joint) if args.hold_shoulder or args.hold_left_shoulder else ()
        validate_delta(args.joint, delta)
        validate_speed(args.speed_deg_s)
    except ValueError as exc:
        parser.error(str(exc))
    index, _, _, _, _, torque_limit = JOINTS[args.joint]
    duration = trajectory_duration(delta, args.hold_only, args.speed_deg_s)
    device_id = device_fingerprint()
    if args.expect_device_id and device_id != args.expect_device_id:
        emit('blocked', reason='computer identity mismatch', device_id=device_id)
        raise SystemExit(1)
    lease = open('/tmp/g1-single-joint-test.lock', 'a')
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another instance of this joint test is running")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

    ChannelFactoryInitialize(0, args.interface)
    manager = MotionSwitcherClient()
    manager.SetTimeout(3)
    manager.Init()
    code, mode = manager.CheckMode()
    if (code != 0 or not isinstance(mode, dict) or not isinstance(mode.get('name'), str)
            or (mode['name'] and not args.release_motion)):
        emit('blocked', reason='motion ownership is not confirmed released', code=code, mode=mode)
        raise SystemExit(1)

    pose_service = probe_pose_service()
    crc = CRC()
    # Signal handlers may latch a stop while this same thread holds the lock.
    lock = threading.RLock()
    state = dict(msg=None, received=0., fault=None)
    active = False
    def latch(reason):
        with lock:
            state['fault'] = state['fault'] or str(reason)
    def receive(msg):
        try:
            if crc.Crc(msg) != msg.crc:
                raise RuntimeError('LowState CRC mismatch')
            with lock:
                state['msg'], state['received'] = msg, time.monotonic()
            validate_state(msg, index if active else None, held_indices=held_indices if active else ())
        except Exception as exc:
            latch(exc)
    def other_command(_):
        if not active:
            latch('another low-level/arm command publisher is active')
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, lambda *_: latch('termination signal'))
    sub = ChannelSubscriber('rt/lowstate', LowState_)
    sub.Init(receive, 1)
    command_readers = []
    publisher = None
    command = None
    aborted = None
    guard_snapshot = None
    samples = []
    result = None
    stop_verified = False
    def current(check_fault=True):
        with lock:
            msg, stamp, fault = state['msg'], state['received'], state['fault']
        if check_fault and fault:
            raise RuntimeError(fault)
        if msg is None or time.monotonic()-stamp > .1:
            raise RuntimeError('LowState unavailable or older than 100 ms')
        return msg
    def write():
        command.crc = crc.Crc(command)
        if publisher.Write(command) is False:
            raise RuntimeError('DDS command write failed')
    try:
        for topic in ('rt/lowcmd', 'rt/arm_sdk'):
            reader = ChannelSubscriber(topic, LowCmd_)
            handler = other_command if topic == 'rt/lowcmd' else lambda _: latch('another arm controller published a command')
            reader.Init(handler, 1)
            command_readers.append(reader)
        deadline = time.monotonic()+3
        while state['msg'] is None and time.monotonic() < deadline:
            time.sleep(.01)
        msg = current()
        anchor = [m.q for m in msg.motor_state[:29]]
        deadline = time.monotonic()+1
        while time.monotonic() < deadline:
            msg = current()
            if any(abs(m.q-anchor[i]) > .01 or abs(m.dq) > .15 for i, m in enumerate(msg.motor_state[:29])):
                raise RuntimeError('all joints must be still before enabling test joints')
            time.sleep(.01)
        anchor = [m.q for m in current().motor_state[:29]]
        validate_plan(args.joint, anchor[index], delta, args.hold_only or args.release_motion)
        holds = {i: anchor[i] for i in held_indices}
        validate_holds(holds)
        emit('preflight', execute=args.execute, release_motion=args.release_motion,
             device_id=device_id, interface=args.interface, pose_service=pose_service, joint=args.joint, index=index,
             hold_only=args.hold_only, anchor_rad=anchor[index], delta_deg=0 if args.hold_only or args.release_motion else args.delta_deg,
             duration_s=0 if args.release_motion else duration,
             target_rad=anchor[index]+(0 if args.hold_only or args.release_motion else delta),
             speed_limit_deg_s=args.speed_deg_s,
             peak_target_speed_deg_s=0 if args.hold_only or args.release_motion else 1.5*abs(args.delta_deg)/ramp_duration(delta, args.speed_deg_s),
             enabled_axes=1+len(holds), held_targets_rad=holds, initial_angles_rad=anchor,
             other_axes='mode 0; no holding torque', motion_owner=mode)
        if args.release_motion:
            if probe_pose_service() != pose_service:
                raise RuntimeError('pose service state changed before release')
            validate_state(current())
            if mode['name']:
                release = manager.ReleaseMode()
                emit('release_requested', result=release)
                if not isinstance(release, (tuple, list)) or not release or release[0] != 0:
                    raise RuntimeError('release result unconfirmed; inspect ownership before retrying')
            code, released = manager.CheckMode()
            if code != 0 or not isinstance(released, dict) or released.get('name') != '':
                raise RuntimeError('motion ownership release not confirmed')
            emit('motion_released', mode=released, device_id=device_id)
            return
        if not args.execute:
            return
        if probe_pose_service() != pose_service:
            raise RuntimeError('pose service state changed before enabling')
        current()
        command = unitree_hg_msg_dds__LowCmd_()
        set_command(command, args.joint, anchor[index], False)
        publisher = ChannelPublisher('rt/lowcmd', LowCmd_)
        publisher.Init()
        current()
        active = True
        start = previous = time.monotonic()
        next_log = 0.
        while True:
            now = time.monotonic()
            elapsed = now-start
            if now-previous > .05:
                raise RuntimeError('control loop delayed more than 50 ms')
            previous = now
            msg = current()
            validate_state(msg, index, held_indices=held_indices)
            offset, phase = target_offset(elapsed, delta, args.hold_only, args.speed_deg_s)
            desired = anchor[index]+offset
            validate_motion(msg, anchor, index, desired, torque_limit, 0 if args.hold_only else delta, holds)
            if elapsed > .5:
                for required in (index, *held_indices):
                    if msg.motor_state[required].mode != 1:
                        raise RuntimeError('joint %d enable not confirmed within 500 ms' % required)
            set_command(command, args.joint, desired, True, holds)
            write()
            motor = msg.motor_state[index]
            samples.append((elapsed, motor.q))
            if elapsed >= next_log:
                emit('control', elapsed_s=elapsed, phase=phase, joint=args.joint,
                     desired=desired, q=motor.q, dq=motor.dq, tau_est=motor.tau_est,
                     mode=motor.mode, temperature=list(motor.temperature),
                     held_joints=[dict(index=i, desired=holds[i], q=msg.motor_state[i].q,
                                       dq=msg.motor_state[i].dq, tau_est=msg.motor_state[i].tau_est,
                                       mode=msg.motor_state[i].mode) for i in held_indices])
                next_log = elapsed+.1
            if elapsed >= duration:
                break
            # Blocking HTTP is deliberately kept outside the command loop.
            time.sleep(max(0., PERIOD-(time.monotonic()-now)))
        result = response_metrics(samples, delta, args.hold_only, args.speed_deg_s)
    except Exception as exc:
        aborted = str(exc)
        if isinstance(exc, MotionStop):
            guard_snapshot = dict(exc.snapshot, elapsed_s=elapsed, phase=phase)
        emit('abort', reason=aborted)
    finally:
        if publisher is not None:
            set_command(command, args.joint, 0., False)
            sent = 0
            for _ in range(20):
                try:
                    write()
                    sent += 1
                except Exception as exc:
                    aborted = aborted or str(exc)
                time.sleep(.005)
            emit('disable_sent', frames=sent, all_command_modes=0)
            publisher.Close()
            deadline = time.monotonic()+1
            while time.monotonic() < deadline:
                try:
                    msg = current(check_fault=False)
                    validate_state(msg, ignore_remote=True)
                    stop_verified = True
                    break
                except Exception:
                    time.sleep(.01)
            # Report the exact failure sample after the disable sequence, so
            # formatting extra diagnostics cannot postpone disable commands.
            if guard_snapshot is not None:
                emit('guard_snapshot', **guard_snapshot)
            emit('disable_feedback', all_29_modes_zero=stop_verified)
        for reader in command_readers:
            reader.Close()
        sub.Close()
        lease.close()
    if aborted or (args.execute and not stop_verified):
        raise SystemExit(1)
    if result:
        emit('result', joint=args.joint, **result, stop_feedback_verified=stop_verified,
             enabled_axes=1+len(held_indices), held_axes=list(held_indices),
             physical_observation='required', rated_torque_acceptance=False)
        if result['verdict'] == 'RESPONSE_OUTSIDE_TEST_LIMITS':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
