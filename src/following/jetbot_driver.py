"""Real JetBot driver over the course agent's HTTP API (port 8100).

Stop layers: Guard -> zero command; this driver's TTL (main loop silent > ttl
-> zero); the agent's own watchdog (no request for ~1 s -> motors off).
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass

import requests

from .controller import MotionCommand

LOG = logging.getLogger(__name__)
ZERO = (0.0, 0.0)


@dataclass(frozen=True)
class DriveConfig:
    forward_to_wheel: float = 2.0
    turn_to_wheel: float = 0.4
    wheel_deadzone: float = 0.15
    max_wheel: float = 0.50
    left_trim: float = 1.0
    right_trim: float = 1.0
    invert_left: bool = False
    invert_right: bool = False
    send_hz: float = 10.0
    command_ttl_s: float = 0.4
    http_timeout_s: float = 0.3
    max_step_per_tick: float = 0.15

    def __post_init__(self) -> None:
        if not 0 <= self.wheel_deadzone < self.max_wheel <= 1.0:
            raise ValueError("Need 0 <= wheel_deadzone < max_wheel <= 1")
        if self.send_hz < 2 or not 0.1 <= self.command_ttl_s < 1.0:
            raise ValueError("Need send_hz >= 2 and command_ttl_s in [0.1, 1.0)")


def mix(command: MotionCommand, cfg: DriveConfig) -> tuple[float, float]:
    """(forward, turn) -> (left, right) in [-max_wheel, max_wheel]. Positive turn = left."""
    if command.forward == 0.0 and command.turn == 0.0:
        return ZERO
    f = cfg.forward_to_wheel * command.forward
    t = cfg.turn_to_wheel * command.turn

    def shape(x: float, trim: float, invert: bool) -> float:
        if abs(x) < 1e-3:
            return 0.0
        y = math.copysign(cfg.wheel_deadzone + abs(x), x) * trim
        y = max(-cfg.max_wheel, min(cfg.max_wheel, y))
        return -y if invert else y

    return shape(f - t, cfg.left_trim, cfg.invert_left), shape(f + t, cfg.right_trim, cfg.invert_right)


def _step(current: float, target: float, limit: float) -> float:
    return current + max(-limit, min(limit, target - current))


class JetBotHttpDriver:
    """Satisfies RobotDriver. send() only stores the target; a thread POSTs at send_hz."""

    def __init__(self, base_url: str, cfg: DriveConfig, *, post=None, get=None,
                 clock=time.monotonic, start_thread: bool = True) -> None:
        self.base_url = base_url.rstrip("/")
        self.cfg = cfg
        self._session = requests.Session()
        # raise_for_status: a 4xx/5xx (e.g. wrong payload format) must count as a link error
        self._post = post or (lambda path, payload: self._session.post(
            self.base_url + path, json=payload, timeout=cfg.http_timeout_s).raise_for_status())
        self._get = get or (lambda path: self._session.get(
            self.base_url + path, timeout=max(1.0, cfg.http_timeout_s)))
        self._clock = clock
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._target = ZERO
        self._t_last_send: float | None = None
        self._sent = ZERO
        self._errors = 0
        self._closed = False
        self.last_command = MotionCommand()
        self.commands_sent = 0
        # daemon: a crash before demo_pc's try/finally must not hang the process
        self._thread = threading.Thread(target=self._run, daemon=True) if start_thread else None
        if self._thread is not None:
            self._thread.start()

    def preflight(self) -> dict:
        health, status = (self._get(path) for path in ("/health", "/status"))
        for resp in (health, status):
            getattr(resp, "raise_for_status", lambda: None)()
        health, status = health.json(), status.json()
        motor = status.get("motor") if isinstance(status, dict) else None
        if not (isinstance(health, dict) and health.get("ok") is True
                and isinstance(motor, dict) and motor.get("ok") is True):
            raise RuntimeError(f"JetBot not ready: /health={health} /status={status}")
        return status

    def send(self, command: MotionCommand) -> None:
        wheels = mix(command, self.cfg)
        with self._lock:
            self._target = wheels
            self._t_last_send = self._clock()
            self.last_command = command
            self.commands_sent += 1
        if wheels == ZERO:
            self._wake.set()

    def _tick(self) -> None:
        with self._lock:
            target, t_last = self._target, self._t_last_send
        if t_last is None or self._clock() - t_last > self.cfg.command_ttl_s:
            target = ZERO
        if target == ZERO:
            wheels = ZERO      # stopping is never rate-limited
        else:
            limit = self.cfg.max_step_per_tick
            wheels = (_step(self._sent[0], target[0], limit), _step(self._sent[1], target[1], limit))
        try:
            self._post("/drive", {"left": wheels[0], "right": wheels[1]})
        except requests.RequestException as exc:
            self._errors += 1
            if self._errors == 3:
                LOG.warning("JetBot link DOWN (%s)", exc)
            return
        self._errors = 0
        self._sent = wheels

    def _run(self) -> None:
        period = 1.0 / self.cfg.send_hz
        while not self._stop.is_set():
            self._wake.clear()
            self._tick()
            self._wake.wait(period)

    @property
    def link_ok(self) -> bool:
        return self._errors < 3

    @property
    def last_wheels(self) -> tuple[float, float]:
        return self._sent

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            self._target = ZERO
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        for _ in range(2):
            try:
                self._post("/stop", {})
            except requests.RequestException:
                pass
        self._sent = ZERO
        self._session.close()
