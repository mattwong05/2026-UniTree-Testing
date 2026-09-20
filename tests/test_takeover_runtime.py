"""Exercise command generation and abort/release without SDK, network or motors."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from test_upper_body import arm


class FakeRobot:
    def __init__(self, fault=False):
        self.now = 0.0
        self.callback = None
        self.fault = fault
        self.writes = []
        self.publishers = 0
        self.state = SimpleNamespace(crc=0, mode_machine=5, mode_pr=0,
            wireless_remote=[0] * 40, motor_state=[SimpleNamespace(
                q=0.1, dq=0.0, tau_est=0.0, vol=48, motorstate=0,
                mode=1, temperature=[40, 40]) for _ in range(35)])

    def sleep(self, seconds):
        self.now += seconds
        if self.writes:
            # Measured shoulder drifts slightly: targets must not follow it.
            self.state.motor_state[15].q = 0.105
        if self.fault and len(self.writes) >= 2:
            self.state.motor_state[15].dq = 0.8
        if self.callback:
            self.callback(deepcopy(self.state))

    def subscriber(self, *_):
        def init(callback, _depth):
            self.callback = callback
            callback(deepcopy(self.state))
        return SimpleNamespace(Init=init, Close=lambda: None)

    def publisher(self, *_):
        self.publishers += 1
        def write(command):
            self.writes.append(deepcopy(command))
            return True
        return SimpleNamespace(Init=lambda: None, Close=lambda: None, Write=write)

    @staticmethod
    def command():
        return SimpleNamespace(crc=0, motor_cmd=[SimpleNamespace(
            q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0) for _ in range(35)])

    def run(self, args):
        modules = {}
        def module(name, **members):
            parts = name.split('.')
            for count in range(1, len(parts)+1):
                prefix = '.'.join(parts[:count])
                if prefix not in modules:
                    modules[prefix] = ModuleType(prefix)
                    modules[prefix].__path__ = []
            for key, value in members.items():
                setattr(modules[name], key, value)
        module('unitree_sdk2py.core.channel', ChannelFactoryInitialize=lambda *_: None,
               ChannelSubscriber=self.subscriber, ChannelPublisher=self.publisher)
        module('unitree_sdk2py.idl.unitree_hg.msg.dds_', LowState_=object, LowCmd_=object)
        module('unitree_sdk2py.idl.default', unitree_hg_msg_dds__LowCmd_=self.command)
        module('unitree_sdk2py.utils.crc', CRC=lambda: SimpleNamespace(Crc=lambda _: 0))
        module('unitree_sdk2py.g1.loco.g1_loco_client', LocoClient=lambda: SimpleNamespace(
            SetTimeout=lambda _: None, Init=lambda: None, _Call=lambda *_: (0, '{"data":802}')))
        module('unitree_sdk2py.g1.loco.g1_loco_api', ROBOT_API_ID_LOCO_GET_FSM_ID=7001)
        output = io.StringIO()
        exit_code = 0
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.object(sys, 'argv', ['upper_body_check.py', *args]))
            stack.enter_context(patch.object(arm.time, 'monotonic', lambda: self.now))
            stack.enter_context(patch.object(arm.time, 'sleep', self.sleep))
            stack.enter_context(patch.object(arm.signal, 'signal', lambda *_: None))
            stack.enter_context(patch.object(arm.threading, 'Thread', lambda **_: SimpleNamespace(
                start=lambda: None, join=lambda **_: None)))
            stack.enter_context(patch.object(arm.urllib.request, 'build_opener', lambda *_: SimpleNamespace(
                open=lambda *_, **__: io.BytesIO(b'{"state":"IDLE"}'))))
            stack.enter_context(redirect_stdout(output))
            try:
                arm.main()
            except SystemExit as exc:
                exit_code = exc.code
        return exit_code, [json.loads(line) for line in output.getvalue().splitlines()]


class TakeoverRuntimeTests(unittest.TestCase):
    def test_read_only_preflight_cannot_create_a_publisher(self):
        robot = FakeRobot()
        code, rows = robot.run(['--check-all'])
        self.assertEqual(code, 0)
        self.assertEqual(robot.publishers, 0)
        self.assertEqual(robot.writes, [])
        self.assertEqual(len(rows[0]['joints']), 14)

    def test_hold_keeps_reference_and_gains_through_takeover(self):
        robot = FakeRobot()
        code, rows = robot.run(['--hold-only', '--execute'])
        self.assertEqual(code, 0)
        self.assertEqual(rows[-1]['kind'], 'completed')
        self.assertTrue(any(command.motor_cmd[29].q == 1 for command in robot.writes))
        for command in robot.writes:
            for index in arm.UPPER:
                self.assertEqual(command.motor_cmd[index].q, 0.1)
                self.assertEqual(command.motor_cmd[index].kp, 40)
                self.assertEqual(command.motor_cmd[index].kd, 1.5)
        self.assertTrue(all(command.motor_cmd[29].q == 0 for command in robot.writes[-15:]))

    def test_velocity_abort_names_joint_and_sends_release(self):
        robot = FakeRobot(fault=True)
        code, rows = robot.run(['--joint', 'left_wrist_pitch', '--execute'])
        self.assertEqual(code, 1)
        guard = next(row for row in rows if row['kind'] == 'guard_snapshot')
        self.assertEqual(guard['joint'], 'left_shoulder_pitch')
        self.assertEqual(guard['target_joint'], 'left_wrist_pitch')
        self.assertEqual(guard['violations'][0]['metric'], 'velocity_rad_s')
        self.assertEqual(guard['last_sent']['kp'], 40)
        self.assertTrue(all(command.motor_cmd[29].q == 0 for command in robot.writes[-15:]))
        self.assertFalse(any(row['kind'] == 'completed' for row in rows))


if __name__ == '__main__':
    unittest.main()
