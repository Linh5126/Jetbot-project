from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import requests

from src.following.controller import MotionCommand
from src.following.jetbot_driver import DriveConfig, JetBotHttpDriver
from src.runtime.frame_source import MjpegHttpSource

_spec = importlib.util.spec_from_file_location(
    "jetbot_agent", Path(__file__).resolve().parents[1] / "jetbot_agent" / "agent.py")
agent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agent)


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


@pytest.mark.parametrize("body", [b'{"left": NaN, "right": 0}', b'{"left": true, "right": 0}',
                                  b'{"left": "0.2", "right": 0}', b'{"left": 0.2}', b"[1, 2]", b"{bad"])
def test_parse_wheels_rejects_bad_input(body):
    with pytest.raises(ValueError):
        agent.parse_wheels(body)


def test_parse_wheels_ok():
    assert agent.parse_wheels(b'{"left": 0.2, "right": -1}') == (0.2, -1.0)


def test_drive_clamps_and_watchdog_stops():
    clock = Clock()
    motor = agent.FakeMotor()
    m = agent.MotorController(motor, "", max_wheel=0.6, watchdog_s=1.0, clock=clock)
    assert m.drive(0.9, -2.0) == (0.6, -0.6)
    clock.t = 0.9
    m.watchdog_tick()
    assert (m.left, m.right) == (0.6, -0.6)
    clock.t = 1.1
    m.watchdog_tick()
    assert (m.left, m.right) == (0.0, 0.0) and motor.history[-1][1:] == (0.0, 0.0)
    assert m.watchdog_trips == 1


def test_open_motor_failure_is_reported_not_raised(monkeypatch):
    def broken(self):
        raise OSError("no I2C")

    monkeypatch.setattr(agent.JetbotLibMotor, "__init__", broken)
    monkeypatch.setattr(agent.AdafruitMotor, "__init__", broken)
    motor, error = agent.open_motor("auto")
    assert motor is None and "jetbot" in error and "adafruit" in error


@pytest.fixture()
def running_agent(tmp_path):
    clip = tmp_path / "cam.avi"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 20, (320, 240))
    for i in range(20):
        writer.write(np.full((240, 320, 3), 10 * i, np.uint8))
    writer.release()
    motor = agent.FakeMotor()
    motors = agent.MotorController(motor, "", 0.6, 0.5)
    app = agent.App(motors, agent.Camera(str(clip), 320, 240))
    server = agent.Server(("127.0.0.1", 0), agent.Handler)
    server.app = app
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    threading.Thread(target=app.watchdog_loop, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", motor, motors
    app.stopping.set()
    server.shutdown()
    server.server_close()
    app.camera.close()


def test_pc_driver_and_stream_against_agent(running_agent):
    base, motor, motors = running_agent
    driver = JetBotHttpDriver(base, DriveConfig())
    try:
        assert driver.preflight()["motor"]["ok"] is True
        driver.send(MotionCommand(forward=0.15))
        time.sleep(0.5)
        driver.send(MotionCommand(forward=0.15))
        time.sleep(0.2)
        assert motors.left > 0 and motors.right > 0 and driver.link_ok
    finally:
        driver.close()
    assert motor.history[-1][1:] == (0.0, 0.0)
    assert requests.post(base + "/drive", json={"left": "x", "right": 0}, timeout=1).status_code == 400

    src = MjpegHttpSource(base + "/stream?w=160&h=120&fps=10&q=70&flip=1")
    try:
        packet = src.read(3.0)
        assert packet is not None and packet.frame.shape == (120, 160, 3)
    finally:
        src.close()


def test_watchdog_stops_when_client_goes_silent(running_agent):
    base, motor, motors = running_agent
    requests.post(base + "/drive", json={"left": 0.3, "right": 0.3}, timeout=1)
    assert motors.left == 0.3
    time.sleep(0.8)  # watchdog_s = 0.5
    assert (motors.left, motors.right) == (0.0, 0.0)
