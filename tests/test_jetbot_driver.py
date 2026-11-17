from __future__ import annotations

import pytest
import requests

from src.following.controller import MotionCommand
from src.following.jetbot_driver import DriveConfig, JetBotHttpDriver, mix

CFG = DriveConfig()


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class FakeResponse:
    def __init__(self, data) -> None:
        self._data = data

    def json(self):
        return self._data


def make_driver(post=None, get=None, cfg=CFG):
    calls = []

    def record(path, payload):
        calls.append((path, payload))

    clock = FakeClock()
    driver = JetBotHttpDriver("http://bot:8100", cfg, post=post or record, get=get,
                              clock=clock, start_thread=False)
    return driver, calls, clock


def test_mix_zero():
    assert mix(MotionCommand(), CFG) == (0.0, 0.0)


def test_mix_turn_direction():
    left, right = mix(MotionCommand(forward=0.1, turn=0.3), CFG)
    assert right > left
    left, right = mix(MotionCommand(forward=0.1, turn=-0.3), CFG)
    assert left > right


def test_mix_straight():
    left, right = mix(MotionCommand(forward=0.1), CFG)
    assert left == right > CFG.wheel_deadzone


def test_mix_never_exceeds_max_wheel():
    for cmd in (MotionCommand(1, 1), MotionCommand(1, -1), MotionCommand(-1, 1), MotionCommand(0, 1)):
        assert all(abs(w) <= CFG.max_wheel for w in mix(cmd, CFG))


def test_mix_invert_and_trim():
    cmd = MotionCommand(forward=0.1)
    base_left, _ = mix(cmd, CFG)
    inv_left, inv_right = mix(cmd, DriveConfig(invert_left=True))
    assert inv_left == -base_left and inv_right > 0
    trim_left, trim_right = mix(cmd, DriveConfig(left_trim=0.9))
    assert trim_left < trim_right


def test_tick_rate_limits_acceleration():
    driver, calls, _ = make_driver()
    driver.send(MotionCommand(forward=0.15))
    driver._tick()
    path, payload = calls[-1]
    assert path == "/drive"
    assert 0 < payload["left"] <= CFG.max_step_per_tick
    assert 0 < payload["right"] <= CFG.max_step_per_tick


def test_ttl_expiry_sends_zero():
    driver, calls, clock = make_driver()
    driver.send(MotionCommand(forward=0.15))
    driver._tick()
    clock.t += CFG.command_ttl_s + 0.01
    driver._tick()
    assert calls[-1] == ("/drive", {"left": 0.0, "right": 0.0})


def test_zero_command_stops_immediately():
    driver, calls, _ = make_driver()
    driver.send(MotionCommand(forward=0.15))
    for _ in range(5):
        driver._tick()
    assert calls[-1][1]["left"] > CFG.max_step_per_tick
    driver.send(MotionCommand())
    driver._tick()
    assert calls[-1] == ("/drive", {"left": 0.0, "right": 0.0})


def test_never_sent_ticks_zero():
    driver, calls, _ = make_driver()
    driver._tick()
    assert calls == [("/drive", {"left": 0.0, "right": 0.0})]


def test_link_down_after_three_errors():
    def failing(path, payload):
        raise requests.ConnectionError("down")

    driver, _, _ = make_driver(post=failing)
    driver.send(MotionCommand(forward=0.1))
    for _ in range(3):
        driver._tick()
    assert driver.link_ok is False
    driver.close()  # must not raise either


def test_close_ends_with_stop():
    driver, calls, _ = make_driver()
    driver.send(MotionCommand(forward=0.1))
    driver._tick()
    driver.close()
    assert calls[-1][0] == "/stop"
    assert driver.last_wheels == (0.0, 0.0)


def test_preflight_rejects_motor_not_ok():
    replies = {"/health": {"ok": True}, "/status": {"motor": {"ok": False}}}
    driver, _, _ = make_driver(get=lambda path: FakeResponse(replies[path]))
    with pytest.raises(RuntimeError):
        driver.preflight()
    replies["/status"] = {"motor": {"ok": True}}
    assert driver.preflight() == {"motor": {"ok": True}}


def test_config_rejects_deadzone_above_max():
    with pytest.raises(ValueError):
        DriveConfig(wheel_deadzone=0.6, max_wheel=0.5)


def test_real_thread_starts_and_stops():
    calls = []
    driver = JetBotHttpDriver("http://bot:8100", CFG, post=lambda p, d: calls.append(p))
    driver.send(MotionCommand(forward=0.1))
    driver.close()
    assert calls[0] == "/drive" and calls[-1] == "/stop"
