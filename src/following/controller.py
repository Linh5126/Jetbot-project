"""Hardware-independent conservative image-based proportional controller."""
from __future__ import annotations

from dataclasses import dataclass

from .target_lock import LockResult


@dataclass(frozen=True)
class MotionCommand:
    """Robot-body motion request: positive forward; positive yaw = turn left."""

    forward: float = 0.0
    turn: float = 0.0


@dataclass(frozen=True)
class ControllerConfig:
    desired_person_height_ratio: float = 0.55
    forward_gain: float = 0.55
    turn_gain: float = 0.7
    max_forward: float = 0.15
    max_turn: float = 0.45
    horizontal_deadband: float = 0.07
    height_deadband: float = 0.06
    min_target_height_ratio: float = 0.08

    def __post_init__(self) -> None:
        if not 0 < self.desired_person_height_ratio < 1:
            raise ValueError("desired_person_height_ratio must be in (0, 1)")
        for name in ("forward_gain", "turn_gain", "max_forward", "max_turn"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("horizontal_deadband", "height_deadband", "min_target_height_ratio"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")


class FollowController:
    """Uses image geometry only, not a physical distance estimate.

    Output units (m/s, rad/s) describe *requested* velocities. Neither a
    monocular box height nor a mock driver ensures collision avoidance.
    Never command reverse motion. Actual deployment requires motor calibration
    and independent obstacle sensing/safety validation.
    """

    def __init__(self, config: ControllerConfig = ControllerConfig()) -> None:
        self.config = config

    def command(self, lock: LockResult, frame_width: int, frame_height: int) -> MotionCommand:
        if not lock.may_follow or frame_width <= 0 or frame_height <= 0:
            return MotionCommand()
        x1, y1, x2, y2 = lock.observation.bbox
        if not 0 <= x1 < x2 <= frame_width or not 0 <= y1 < y2 <= frame_height:
            return MotionCommand()
        person_height = (y2 - y1) / frame_height
        if person_height < self.config.min_target_height_ratio:
            return MotionCommand()
        horizontal_error = ((x1 + x2) / frame_width) - 1.0  # right of image = positive
        height_error = self.config.desired_person_height_ratio - person_height
        turn = 0.0
        if abs(horizontal_error) > self.config.horizontal_deadband:
            # Positive yaw means left turn, negative means right turn.
            turn = max(-self.config.max_turn, min(self.config.max_turn,
                -self.config.turn_gain * horizontal_error))
        forward = 0.0
        if height_error > self.config.height_deadband:
            forward = min(self.config.max_forward, self.config.forward_gain * height_error)
        return MotionCommand(forward=forward, turn=turn)
