"""No SDK/network required: one enabled axis, bounded trajectory and stop paths."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from copy import deepcopy
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('lowlevel', Path(__file__).resolve().parents[1]/'scripts/lowlevel_joint_check.py')
ll = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ll)


def command():
    return SimpleNamespace(crc=0, motor_cmd=[SimpleNamespace(
        mode=1, q=1., dq=1., tau=1., kp=100., kd=10.) for _ in range(35)])


def state():
    return SimpleNamespace(crc=0, mode_machine=5, mode_pr=0, wireless_remote=[0]*40,
        motor_state=[SimpleNamespace(mode=0, q=.1, dq=0., tau_est=0., vol=48.,
            motorstate=0, temperature=[40, 40]) for _ in range(35)])


class LowlevelPlanTests(unittest.TestCase):
    def test_exactly_one_axis_enabled_and_unused_weight_slot_zero(self):
        for name, (index, *_) in ll.JOINTS.items():
            msg = command()
            ll.set_command(msg, name, .123, True)
            self.assertEqual([i for i, m in enumerate(msg.motor_cmd) if m.mode], [index])
            self.assertEqual(msg.motor_cmd[index].q, .123)
            for i, m in enumerate(msg.motor_cmd):
                self.assertEqual(m.tau, 0)
                self.assertEqual(m.dq, 0)
                if i != index:
                    self.assertEqual((m.q, m.kp, m.kd), (0, 0, 0))
            self.assertEqual((msg.mode_machine, msg.mode_pr), (5, 0))
            ll.set_command(msg, name, .123, False)
            self.assertTrue(all((m.mode, m.q, m.kp, m.kd, m.tau) == (0, 0, 0, 0, 0) for m in msg.motor_cmd))

    def test_signed_trajectory_envelope_and_return(self):
        for deg in (-2, -.25, .25, 1, 2):
            delta = math.radians(deg)
            previous = 0
            for i in range(10001):
                offset, _ = ll.target_offset(i/1000, delta)
                self.assertGreaterEqual(offset, min(0, delta)-1e-12)
                self.assertLessEqual(offset, max(0, delta)+1e-12)
                self.assertLessEqual(abs(offset-previous)/.001, math.radians(1.001))
                previous = offset
            self.assertEqual(ll.target_offset(9, delta)[0], 0)
        for t in (0, 1, 2, 3):
            self.assertEqual(ll.target_offset(t, 1, True)[0], 0)

    def test_bad_plan_and_state_fail_closed(self):
        for delta in (0, math.nan, math.inf, math.radians(61)):
            with self.assertRaises(ValueError):
                ll.validate_plan('left_elbow', .1, delta)
        with self.assertRaises(ValueError):
            ll.validate_plan('left_knee', -.05, math.radians(1))
        mutations = [lambda s: setattr(s.motor_state[0], 'mode', 1),
                     lambda s: setattr(s.motor_state[18], 'motorstate', 1),
                     lambda s: setattr(s.motor_state[18], 'q', math.nan),
                     lambda s: setattr(s, 'mode_pr', 1),
                     lambda s: s.wireless_remote.__setitem__(2, 1),
                     lambda s: setattr(s.motor_state[18], 'temperature', [60, 40])]
        for mutate in mutations:
            msg = state()
            mutate(msg)
            with self.assertRaises(RuntimeError):
                ll.validate_state(msg, 18)

    def test_response_requires_movement_and_return(self):
        delta = math.radians(1)
        samples = [(i/100, .1+ll.target_offset(i/100, delta)[0]) for i in range(901)]
        self.assertEqual(ll.response_metrics(samples, delta, False)['verdict'], 'RESPONSE_WITHIN_TEST_LIMITS')
        stalled = [(t, .1) for t, _ in samples]
        self.assertEqual(ll.response_metrics(stalled, delta, False)['verdict'], 'RESPONSE_OUTSIDE_TEST_LIMITS')

    def test_larger_left_elbow_excursion_keeps_speed_bounds_and_other_axis_limits(self):
        for deg in (-15, 15):
            delta = math.radians(deg)
            ll.validate_plan('left_elbow', 1., delta)
            end = ll.trajectory_duration(delta)
            self.assertAlmostEqual(end, 50.)
            samples = [(i/100, ll.target_offset(i/100, delta)[0]) for i in range(int(end*100)+1)]
            self.assertAlmostEqual(samples[-1][1], 0.)
            self.assertAlmostEqual(max(abs(q) for _, q in samples), abs(delta))
            self.assertLessEqual(max(abs(b[1]-a[1])/.01 for a, b in zip(samples, samples[1:])), math.radians(1.001))
            self.assertEqual(ll.response_metrics(samples, delta, False)['verdict'], 'RESPONSE_WITHIN_TEST_LIMITS')
        for name in ll.JOINTS:
            if not name.endswith('_elbow'):
                with self.assertRaises(ValueError):
                    ll.validate_plan(name, 1., math.radians(15))
        with self.assertRaises(ValueError):
            ll.validate_plan('left_elbow', 1.9, math.radians(15))

    def test_large_excursion_does_not_relax_tracking_or_unselected_motion_stops(self):
        msg = state()
        anchors = [m.q for m in msg.motor_state[:29]]
        msg.motor_state[18].q += math.radians(15)
        ll.validate_motion(msg, anchors, 18, msg.motor_state[18].q, 4., math.radians(15))
        with self.assertRaises(RuntimeError):
            ll.validate_motion(msg, anchors, 18, msg.motor_state[18].q+.051, 4., math.radians(15))
        msg.motor_state[0].q += .051
        with self.assertRaises(RuntimeError):
            ll.validate_motion(msg, anchors, 18, msg.motor_state[18].q, 4., math.radians(15))

    def test_shoulder_hold_targets_and_torque_guards(self):
        holds = {15: .12, 16: .13, 17: .14}
        msg = command()
        ll.set_command(msg, 'left_elbow', 1., True, holds)
        self.assertEqual([i for i, m in enumerate(msg.motor_cmd) if m.mode], [15, 16, 17, 18])
        for i, q in holds.items():
            self.assertEqual((msg.motor_cmd[i].q, msg.motor_cmd[i].kp, msg.motor_cmd[i].kd), (q, 40., 1.))
        ll.set_command(msg, 'left_elbow', 1., False, holds)
        self.assertTrue(all((m.mode, m.kp, m.kd, m.tau) == (0, 0, 0, 0) for m in msg.motor_cmd))
        for bad in ({15: .1}, {15: math.nan, 16: .1, 17: .1}, {15: .1, 16: .1, 17: 3.}):
            with self.assertRaises(ValueError):
                ll.validate_holds(bad)
        with self.assertRaises(ValueError):
            ll.set_command(msg, 'right_elbow', 1., True, holds)
        robot = state()
        for i, q in holds.items():
            robot.motor_state[i].q = q
            robot.motor_state[i].mode = 1
        ll.validate_state(robot, 18, held_indices=holds)
        with self.assertRaises(RuntimeError):
            ll.validate_state(robot, 18)
        anchors = [m.q for m in robot.motor_state[:29]]
        robot.motor_state[16].tau_est = 4.01
        with self.assertRaises(RuntimeError):
            ll.validate_motion(robot, anchors, 18, .1, 4., math.radians(15), holds)

    def test_motion_stop_identifies_each_limit_and_exact_sample(self):
        for field, value, expected in (('q', .151, ['tracking']),
                                       ('tau_est', 4.01, ['estimated_torque']),
                                       ('dq', -.301, ['velocity'])):
            msg = state()
            anchors = [m.q for m in msg.motor_state[:29]]
            setattr(msg.motor_state[18], field, value)
            with self.assertRaises(ll.MotionStop) as caught:
                ll.validate_motion(msg, anchors, 18, .1, 4., math.radians(60))
            details = caught.exception.snapshot
            self.assertEqual(details['index'], 18)
            self.assertEqual(details['violations'], expected)
            self.assertEqual(details['q_rad'], msg.motor_state[18].q)
            self.assertEqual(details['tau_est_nm'], msg.motor_state[18].tau_est)
            if field == 'q':
                self.assertAlmostEqual(details['tracking_error_deg'], math.degrees(.051))
                self.assertAlmostEqual(details['tracking_limit_deg'], math.degrees(.05))
        msg.motor_state[18].dq = 0
        msg.motor_state[18].q = .151
        msg.motor_state[18].tau_est = 4.01
        with self.assertRaises(ll.MotionStop) as caught:
            ll.validate_motion(msg, anchors, 18, .1, 4., math.radians(60))
        self.assertEqual(caught.exception.snapshot['violations'], ['tracking', 'estimated_torque'])

    def test_sixty_degree_reverse_path_speed_limits_and_metric_windows(self):
        delta = math.radians(-60)
        for name in ('left_elbow', 'right_elbow'):
            ll.validate_plan(name, math.radians(76), delta)
            with self.assertRaises(ValueError):
                ll.validate_plan(name, math.radians(76), -delta)
        for speed in (.25, 1., 3., 5.):
            end = ll.trajectory_duration(delta, speed_deg_s=speed)
            samples = [(i/100, 1.3+ll.target_offset(i/100, delta, speed_deg_s=speed)[0]) for i in range(round(end*100)+1)]
            self.assertLessEqual(max(abs(b[1]-a[1])/.01 for a,b in zip(samples,samples[1:])), math.radians(speed+.001))
            self.assertAlmostEqual(min(q for _,q in samples), 1.3+delta)
            self.assertAlmostEqual(samples[-1][1], 1.3)
            self.assertEqual(ll.response_metrics(samples, delta, False, speed)['verdict'], 'RESPONSE_WITHIN_TEST_LIMITS')
        self.assertAlmostEqual(ll.trajectory_duration(delta, speed_deg_s=5.), 41.)
        for speed in (0., -1., .1, 6., math.nan, math.inf):
            with self.assertRaises(ValueError):
                ll.ramp_duration(delta, speed)

    def test_missing_pose_service_is_distinct_from_timeout_or_active_service(self):
        for error, allowed in ((ConnectionRefusedError(), True), (TimeoutError(), False)):
            opener = SimpleNamespace(open=lambda *a, **k: (_ for _ in ()).throw(ll.urllib.error.URLError(error)))
            with patch.object(ll.urllib.request, 'build_opener', return_value=opener):
                if allowed:
                    self.assertEqual(ll.probe_pose_service(), 'not_listening')
                else:
                    with self.assertRaises(ll.urllib.error.URLError):
                        ll.probe_pose_service()
        with patch.object(ll.urllib.request, 'build_opener', return_value=SimpleNamespace(
                open=lambda *a, **k: io.BytesIO(b'{"state":"RUNNING"}'))):
            with self.assertRaises(RuntimeError):
                ll.probe_pose_service()


class FakeRobot:
    def __init__(self, owner='', remote_stop=False, enable_works=True, disable_works=True, passive_motion=False,
                 missing_axis=None, shoulder_torque_fault=False, motion_fault=None):
        self.owner = owner
        self.remote_stop = remote_stop
        self.enable_works = enable_works
        self.disable_works = disable_works
        self.passive_motion = passive_motion
        self.missing_axis = missing_axis
        self.shoulder_torque_fault = shoulder_torque_fault
        self.motion_fault = motion_fault
        self.now = 0.
        self.msg = state()
        self.callback = None
        self.writes = []
        self.publishers = 0
        self.control_started = None
        self.release_calls = 0
    def release_mode(self):
        self.release_calls += 1
        self.owner = ''
        return (0, None)
    def sleep(self, seconds):
        self.now += seconds
        if self.writes:
            for motor, sent in zip(self.msg.motor_state, self.writes[-1].motor_cmd):
                if sent.mode or self.disable_works:
                    motor.mode = sent.mode if self.enable_works else 0
                if sent.mode:
                    motor.q = sent.q
            if self.remote_stop and self.now-self.control_started > .2:
                self.msg.wireless_remote[2] = 1
            if self.passive_motion and self.now-self.control_started > 4:
                self.msg.motor_state[15].q = .151
            if self.missing_axis is not None:
                self.msg.motor_state[self.missing_axis].mode = 0
            if self.shoulder_torque_fault and self.now-self.control_started > .8:
                self.msg.motor_state[16].tau_est = 4.01
            if self.motion_fault and self.now-self.control_started > .8:
                axis, field, value = self.motion_fault
                setattr(self.msg.motor_state[axis], field, value)
        if self.callback:
            self.callback(deepcopy(self.msg))
    def subscriber(self, topic, _type):
        def init(callback, _depth):
            if topic == 'rt/lowstate':
                self.callback = callback
                callback(deepcopy(self.msg))
        return SimpleNamespace(Init=init, Close=lambda: None)
    def publisher(self, topic, _type):
        assert topic == 'rt/lowcmd'
        self.publishers += 1
        def write(cmd):
            if self.control_started is None:
                self.control_started = self.now
            self.writes.append(deepcopy(cmd))
            return True
        return SimpleNamespace(Init=lambda: None, Close=lambda: None, Write=write)
    def run(self, args, provide_identity=True):
        args = list(args)
        if provide_identity and ('--execute' in args or '--release-motion' in args) and '--expect-device-id' not in args:
            args += ['--expect-device-id', 'a'*16]
        modules = {}
        def module(name, **members):
            parts = name.split('.')
            for i in range(1, len(parts)+1):
                prefix = '.'.join(parts[:i])
                if prefix not in modules:
                    modules[prefix] = ModuleType(prefix)
                    modules[prefix].__path__ = []
            for key, value in members.items():
                setattr(modules[name], key, value)
        module('unitree_sdk2py.core.channel', ChannelFactoryInitialize=lambda *_: None,
               ChannelSubscriber=self.subscriber, ChannelPublisher=self.publisher)
        module('unitree_sdk2py.idl.unitree_hg.msg.dds_', LowState_=object, LowCmd_=object)
        module('unitree_sdk2py.idl.default', unitree_hg_msg_dds__LowCmd_=command)
        module('unitree_sdk2py.utils.crc', CRC=lambda: SimpleNamespace(Crc=lambda _: 0))
        module('unitree_sdk2py.comm.motion_switcher.motion_switcher_client', MotionSwitcherClient=lambda: SimpleNamespace(
            SetTimeout=lambda _: None, Init=lambda: None, CheckMode=lambda: (0, {'name':self.owner}), ReleaseMode=self.release_mode))
        output = io.StringIO()
        code = 0
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.object(sys, 'argv', ['lowlevel_joint_check.py', *args]))
            stack.enter_context(patch.object(ll.time, 'monotonic', lambda: self.now))
            stack.enter_context(patch.object(ll.time, 'sleep', self.sleep))
            stack.enter_context(patch.object(ll.signal, 'signal', lambda *_: None))
            stack.enter_context(patch.object(ll, 'device_fingerprint', return_value='a'*16))
            stack.enter_context(patch.object(ll, 'open', lambda *_: io.StringIO(), create=True))
            stack.enter_context(patch.object(ll.fcntl, 'flock', lambda *_: None))
            stack.enter_context(patch.object(ll.urllib.request, 'build_opener', lambda *_: SimpleNamespace(
                open=lambda *_, **__: io.BytesIO(b'{"state":"IDLE"}'))))
            stack.enter_context(redirect_stdout(output))
            stack.enter_context(redirect_stderr(io.StringIO()))
            try:
                ll.main()
            except SystemExit as exc:
                code = exc.code
        return code, [json.loads(line) for line in output.getvalue().splitlines()]


class LowlevelRuntimeTests(unittest.TestCase):
    def test_preflight_does_not_create_a_writer(self):
        robot = FakeRobot()
        code, _ = robot.run([])
        self.assertEqual(code, 0)
        self.assertEqual(robot.publishers, 0)
    def test_motion_owner_blocks_before_any_command(self):
        robot = FakeRobot(owner='ai')
        code, rows = robot.run(['--execute'])
        self.assertEqual(code, 1)
        self.assertEqual(robot.publishers, 0)
        self.assertEqual(rows[0]['kind'], 'blocked')
    def test_one_joint_trajectory_and_verified_disable(self):
        robot = FakeRobot()
        code, rows = robot.run(['--joint', 'right_elbow', '--execute'])
        self.assertEqual(code, 0)
        self.assertEqual(rows[-1]['verdict'], 'RESPONSE_WITHIN_TEST_LIMITS')
        self.assertTrue(rows[-1]['stop_feedback_verified'])
        for cmd in robot.writes:
            self.assertTrue(all(m.mode == 0 for i, m in enumerate(cmd.motor_cmd) if i != 25))
        self.assertTrue(all(all(m.mode == 0 for m in cmd.motor_cmd) for cmd in robot.writes[-20:]))
    def test_remote_stop_disables_without_resuming(self):
        robot = FakeRobot(remote_stop=True)
        code, rows = robot.run(['--execute'])
        self.assertEqual(code, 1)
        self.assertTrue(any(r['kind']=='abort' and 'remote' in r['reason'] for r in rows))
        self.assertTrue(rows[-1]['all_29_modes_zero'])
        self.assertTrue(all(all(m.mode == 0 for m in cmd.motor_cmd) for cmd in robot.writes[-20:]))
    def test_large_left_elbow_trajectory_and_verified_disable(self):
        robot = FakeRobot()
        with patch.object(ll, 'PERIOD', .01):
            code, rows = robot.run(['--joint', 'left_elbow', '--delta-deg', '15', '--execute'])
        self.assertEqual(code, 0)
        self.assertEqual(rows[-1]['verdict'], 'RESPONSE_WITHIN_TEST_LIMITS')
        self.assertTrue(rows[-1]['stop_feedback_verified'])
        self.assertAlmostEqual(rows[-1]['measured_excursion_deg'], 15.)
        for cmd in robot.writes:
            self.assertTrue(all(m.mode == 0 for i, m in enumerate(cmd.motor_cmd) if i != 18))
        self.assertTrue(all(all(m.mode == 0 for m in cmd.motor_cmd) for cmd in robot.writes[-20:]))
    def test_missing_enable_feedback_stops(self):
        robot = FakeRobot(enable_works=False)
        code, rows = robot.run(['--execute'])
        self.assertEqual(code, 1)
        self.assertTrue(any(r['kind']=='abort' and 'enable not confirmed' in r['reason'] for r in rows))
    def test_passive_shoulder_motion_aborts_elbow_and_verifies_disable(self):
        robot = FakeRobot(passive_motion=True)
        code, rows = robot.run(['--joint', 'left_elbow', '--delta-deg', '15', '--execute'])
        self.assertEqual(code, 1)
        self.assertTrue(any(r['kind']=='abort' and 'joint 15 displacement stop' in r['reason'] for r in rows))
        self.assertTrue(rows[-1]['all_29_modes_zero'])
        self.assertFalse(any(r['kind']=='result' for r in rows))
        self.assertTrue(all(all(m.mode == 0 for m in cmd.motor_cmd) for cmd in robot.writes[-20:]))
    def test_failed_disable_feedback_cannot_report_success(self):
        robot = FakeRobot(disable_works=False)
        code, rows = robot.run(['--hold-only', '--execute'])
        self.assertEqual(code, 1)
        self.assertFalse(rows[-1]['all_29_modes_zero'])
        self.assertFalse(any(r['kind']=='result' for r in rows))
    def test_shoulder_hold_preflight_is_readonly(self):
        robot = FakeRobot()
        code, rows = robot.run(['--hold-left-shoulder'])
        self.assertEqual(code, 0)
        self.assertEqual(robot.publishers, 0)
        self.assertEqual(rows[0]['enabled_axes'], 4)
    def test_shoulder_hold_is_fixed_through_elbow_excursion(self):
        robot = FakeRobot()
        with patch.object(ll, 'PERIOD', .01):
            code, rows = robot.run(['--delta-deg', '15', '--hold-left-shoulder', '--execute'])
        self.assertEqual(code, 0)
        self.assertEqual(rows[-1]['enabled_axes'], 4)
        self.assertTrue(rows[-1]['stop_feedback_verified'])
        for cmd in robot.writes:
            active = [i for i, m in enumerate(cmd.motor_cmd) if m.mode]
            self.assertIn(active, ([], [15, 16, 17, 18]))
            if active:
                self.assertTrue(all(cmd.motor_cmd[i].q == .1 for i in (15, 16, 17)))
        self.assertTrue(all(all(m.mode == 0 for m in cmd.motor_cmd) for cmd in robot.writes[-20:]))
    def test_shoulder_faults_and_remote_stop_disable_all_four_axes(self):
        for kwargs in ({'missing_axis':16}, {'shoulder_torque_fault':True}, {'remote_stop':True}):
            robot = FakeRobot(**kwargs)
            code, rows = robot.run(['--hold-left-shoulder', '--hold-only', '--execute'])
            self.assertEqual(code, 1)
            self.assertTrue(any(r['kind']=='abort' for r in rows))
            self.assertTrue(rows[-1]['all_29_modes_zero'])
            self.assertTrue(all(all(m.mode == 0 for m in cmd.motor_cmd) for cmd in robot.writes[-20:]))

    def test_exact_guard_snapshot_is_emitted_after_disable_and_keeps_failure_exit(self):
        for axis, field, value, violation in ((18, 'q', .151, 'tracking'),
                                               (18, 'tau_est', 4.01, 'estimated_torque'),
                                               (18, 'dq', -.301, 'velocity'),
                                               (16, 'tau_est', 4.01, 'estimated_torque')):
            robot = FakeRobot(motion_fault=(axis, field, value))
            code, rows = robot.run(['--hold-shoulder', '--hold-only', '--execute'])
            self.assertEqual(code, 1)
            details = next(r for r in rows if r['kind'] == 'guard_snapshot')
            self.assertEqual(details['index'], axis)
            self.assertEqual(details['violations'], [violation])
            self.assertIn('elapsed_s', details)
            kinds = [r['kind'] for r in rows]
            self.assertLess(kinds.index('disable_sent'), kinds.index('guard_snapshot'))
            self.assertTrue(rows[-1]['all_29_modes_zero'])
            self.assertTrue(all(all(m.mode == 0 for m in c.motor_cmd) for c in robot.writes[-20:]))
    def test_identity_required_and_mismatch_never_publishes(self):
        for args in (['--execute'], ['--release-motion']):
            robot = FakeRobot()
            code, _ = robot.run(args, provide_identity=False)
            self.assertEqual(code, 2)
            self.assertEqual(robot.publishers, 0)
            code, rows = robot.run(args+['--expect-device-id', 'b'*16])
            self.assertEqual(code, 1)
            self.assertEqual(rows[0]['reason'], 'computer identity mismatch')
            self.assertEqual(robot.publishers, 0)
            self.assertEqual(robot.release_calls, 0)
    def test_release_once_checks_disabled_feedback_and_never_publishes(self):
        robot = FakeRobot(owner='ai')
        code, rows = robot.run(['--release-motion'])
        self.assertEqual(code, 0)
        self.assertEqual(robot.release_calls, 1)
        self.assertEqual(robot.publishers, 0)
        self.assertEqual(rows[-1]['kind'], 'motion_released')
        robot = FakeRobot(owner='ai')
        robot.msg.motor_state[0].mode = 1
        code, _ = robot.run(['--release-motion'])
        self.assertEqual(code, 1)
        self.assertEqual(robot.release_calls, 0)
        self.assertEqual(robot.publishers, 0)
    def test_reverse_sixty_degree_right_elbow_holds_correct_side(self):
        robot = FakeRobot()
        robot.msg.motor_state[25].q = math.radians(76)
        with patch.object(ll, 'PERIOD', .01):
            code, rows = robot.run(['--joint', 'right_elbow', '--delta-deg', '-60', '--speed-deg-s', '5', '--hold-shoulder', '--execute'])
        self.assertEqual(code, 0)
        self.assertAlmostEqual(rows[-1]['measured_excursion_deg'], -60.)
        self.assertEqual(rows[-1]['held_axes'], [22,23,24])
        for cmd in robot.writes:
            active = [i for i,m in enumerate(cmd.motor_cmd) if m.mode]
            self.assertIn(active, ([], [22,23,24,25]))
            if active:
                self.assertTrue(all(cmd.motor_cmd[i].q == .1 for i in (22,23,24)))
        self.assertTrue(rows[-1]['stop_feedback_verified'])


if __name__ == '__main__':
    unittest.main()
