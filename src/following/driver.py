"""Robot motion interface. This project intentionally ships no real motor driver."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .controller import MotionCommand


class RobotDriver(Protocol):
    def send(self, command: MotionCommand) -> None: ...
    def close(self) -> None: ...


@dataclass
class MockRobotDriver:
    """Records the last requested command; DOES NOT drive any hardware."""
    last_command: MotionCommand = MotionCommand()
    commands_sent: int = 0

    def send(self, command: MotionCommand) -> None:
        self.last_command = command
        self.commands_sent += 1

    def close(self) -> None:
        self.send(MotionCommand())
