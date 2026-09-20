"""No SDK/network required: one enabled axis, bounded trajectory and stop paths."""
from contextlib import ExitStack, redirect_stdout
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
        for delta in (0, math.nan, math.inf, math.radians(16)):
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
            if name != 'left_elbow':
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


class FakeRobot:
    def __init__(self, owner='', remote_stop=False, enable_works=True, disable_works=True, passive_motion=False):
        self.owner = owner
        self.remote_stop = remote_stop
        self.enable_works = enable_works
        self.disable_works = disable_works
        self.passive_motion = passive_motion
        self.now = 0.
        self.msg = state()
        self.callback = None
        self.writes = []
        self.publishers = 0
        self.control_started = None
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
    def run(self, args):
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
            SetTimeout=lambda _: None, Init=lambda: None, CheckMode=lambda: (0, {'name':self.owner})))
        output = io.StringIO()
        code = 0
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.object(sys, 'argv', ['lowlevel_joint_check.py', *args]))
            stack.enter_context(patch.object(ll.time, 'monotonic', lambda: self.now))
            stack.enter_context(patch.object(ll.time, 'sleep', self.sleep))
            stack.enter_context(patch.object(ll.signal, 'signal', lambda *_: None))
            stack.enter_context(patch.object(ll, 'open', lambda *_: io.StringIO(), create=True))
            stack.enter_context(patch.object(ll.fcntl, 'flock', lambda *_: None))
            stack.enter_context(patch.object(ll.urllib.request, 'build_opener', lambda *_: SimpleNamespace(
                open=lambda *_, **__: io.BytesIO(b'{"state":"IDLE"}'))))
            stack.enter_context(redirect_stdout(output))
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
        self.assertTrue(any(r['kind']=='abort' and 'joint 15 displacement/velocity stop' in r['reason'] for r in rows))
        self.assertTrue(rows[-1]['all_29_modes_zero'])
        self.assertFalse(any(r['kind']=='result' for r in rows))
        self.assertTrue(all(all(m.mode == 0 for m in cmd.motor_cmd) for cmd in robot.writes[-20:]))
    def test_failed_disable_feedback_cannot_report_success(self):
        robot = FakeRobot(disable_works=False)
        code, rows = robot.run(['--hold-only', '--execute'])
        self.assertEqual(code, 1)
        self.assertFalse(rows[-1]['all_29_modes_zero'])
        self.assertFalse(any(r['kind']=='result' for r in rows))


if __name__ == '__main__':
    unittest.main()
