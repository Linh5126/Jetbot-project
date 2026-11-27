"""Hardware-independent image-based controllers: proportional (PC demo) and staged (on-board)."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

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


@dataclass(frozen=True)
class StagedControllerConfig:
    forward_speed: float = 0.12          # constant forward request while approaching
    stop_height_ratio: float = 0.70      # target at least this tall (box / frame) -> stop
    resume_margin: float = 0.05          # approach again only below stop_height_ratio - margin
    turn_gain: float = 0.5
    max_turn: float = 0.30
    horizontal_deadband: float = 0.15    # on (cx - 0.5) * 2; inside it the robot does not turn
    max_side_error: float = 0.75         # target further to the side -> turn in place only
    settle_frames: int = 3               # consecutive consistent boxes before any motion
    min_continuity_iou: float = 0.10     # box overlap with the previous frame to count as consistent
    turn_smoothing: float = 0.3          # weight of the new turn request (exponential smoothing)
    min_target_height_ratio: float = 0.08
    head_cut_margin: float = 0.01        # box top this close to the frame top -> treated as close enough
    slow_band: float = 0.15              # within this much of stop_height_ratio the approach slows down
    min_forward_ratio: float = 0.5       # slowest approach, as a fraction of forward_speed
    coast_frames: int = 0                # target gone this many frames during a full-speed approach -> keep going
    search_turn: float = 0.0             # >0: target lost at the image edge -> turn that way (one direction only)
    search_frames: int = 0               # ... for at most this many frames after the brief-loss window
    search_edge: float = 0.5             # last |side error| at least this (box centre in the outer quarter)

    def __post_init__(self) -> None:
        if not 0 < self.stop_height_ratio < 1 or not 0 <= self.resume_margin < self.stop_height_ratio:
            raise ValueError("Need 0 < stop_height_ratio < 1 and 0 <= resume_margin < stop_height_ratio")
        for name in ("forward_speed", "turn_gain", "max_turn"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("horizontal_deadband", "min_continuity_iou", "min_target_height_ratio", "head_cut_margin"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if not 0 < self.slow_band < 1 or not 0 < self.min_forward_ratio <= 1:
            raise ValueError("Need slow_band in (0, 1) and min_forward_ratio in (0, 1]")
        if self.coast_frames < 0 or self.search_frames < 0 or self.search_turn < 0:
            raise ValueError("coast_frames, search_frames and search_turn must be >= 0")
        if not 0 < self.turn_smoothing <= 1 or not 0 < self.max_side_error <= 1 or self.settle_frames < 1:
            raise ValueError("Need turn_smoothing and max_side_error in (0, 1] and settle_frames >= 1")


def _box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class StagedFollowController:
    """Stop-and-go follower for the real robot, same inputs as FollowController.

    Phases: IDLE (nothing to follow) -> SETTLING (the box must stay consistent for
    settle_frames) -> APPROACH (forward_speed, slowing to min_forward_ratio within slow_band
    of the stop height, so the robot does not arrive at full speed) <-> HOLD (close enough or far to
    the side; still turns to face the target). A target that flickers out for at most
    coast_frames during a full-speed (far) approach keeps the robot going straight (COAST)
    instead of stop, settle and ramp up again; otherwise it stands still (PAUSE) but keeps the
    box, so no new settling is needed if the target comes back in the same place: sim_follow.py showed that stop-and-go halving
    the speed while Re-ID re-verifies a far target. Once the head is cut by the top of the frame the
    box stops growing as the robot gets closer, so a cut head also counts as close enough
    (simulation, scripts/sim_follow.py). Stop/resume use two thresholds so the robot
    does not stutter around the stop distance, and turn requests are smoothed to damp the
    overshoot of a slow camera loop. Never commands reverse motion.
    """

    def __init__(self, config: StagedControllerConfig = StagedControllerConfig()) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.phase = "IDLE"
        self._last_box: Optional[Tuple[float, float, float, float]] = None
        self._settled = 0
        self._turn = 0.0
        self._approaching = False
        self._missing = 0
        self._forward = 0.0
        self._last_side = 0.0

    def command(self, lock: LockResult, frame_width: int, frame_height: int) -> MotionCommand:
        if not lock.may_follow:
            return self.command_for_box(None, frame_width, frame_height)
        return self.command_for_box(lock.observation.bbox, frame_width, frame_height)

    def command_for_box(self, bbox, frame_width: int, frame_height: int) -> MotionCommand:
        """bbox: target xyxy in pixels (may poke past the frame), or None when there is none."""
        cfg = self.config
        box = None
        if bbox is not None and frame_width > 0 and frame_height > 0:
            x1, y1, x2, y2 = (min(1.0, max(0.0, v / s)) for v, s in
                              zip(bbox, (frame_width, frame_height, frame_width, frame_height)))
            if x2 > x1 and y2 - y1 >= cfg.min_target_height_ratio:
                box = (x1, y1, x2, y2)
        if box is None:
            self._missing += 1
            if self._missing <= cfg.coast_frames and self._last_box is not None:
                if self.phase in ("APPROACH", "COAST") and self._forward >= cfg.forward_speed > 0:
                    self.phase = "COAST"
                    return MotionCommand(forward=self._forward)
                # Brief loss: stand still but remember the box, so a target back in the same
                # place needs no new SETTLING (27/11 PC log: re-verify + settle stalled ~0.7 s).
                self.phase, self._turn = "PAUSE", 0.0  # an old turn request would overshoot (sim, yaw 8)
                return MotionCommand()
            if (cfg.search_turn and abs(self._last_side) >= cfg.search_edge
                    and self._missing <= cfg.coast_frames + cfg.search_frames):
                # Walked out at the side: look that way, once, never back and forth.
                self.phase = "SEARCH"
                return MotionCommand(turn=-math.copysign(cfg.search_turn, self._last_side))
            self.reset()
            return MotionCommand()
        self._missing = 0

        consistent = self._last_box is not None and _box_iou(self._last_box, box) >= cfg.min_continuity_iou
        if consistent:
            self._settled += 1
        else:
            self._settled, self._turn, self._approaching = 1, 0.0, False
        self._last_box = box
        if self._settled < cfg.settle_frames:
            self.phase = "SETTLING"
            return MotionCommand()

        height = box[3] - box[1]
        side_error = box[0] + box[2] - 1.0  # right of image = positive
        self._last_side = side_error
        head_cut = box[1] <= cfg.head_cut_margin
        if height >= cfg.stop_height_ratio or head_cut:
            self._approaching = False
        elif height < cfg.stop_height_ratio - cfg.resume_margin and box[1] > 3 * cfg.head_cut_margin:
            self._approaching = True
        if abs(side_error) <= cfg.horizontal_deadband:
            self._turn = 0.0  # no smoothing tail: tiny turns would still spin the wheels at the deadzone
        else:
            wanted = max(-cfg.max_turn, min(cfg.max_turn, -cfg.turn_gain * side_error))
            self._turn += cfg.turn_smoothing * (wanted - self._turn)
        forward = 0.0
        if self._approaching and abs(side_error) <= cfg.max_side_error:
            ratio = (cfg.stop_height_ratio - height) / cfg.slow_band
            forward = cfg.forward_speed * max(cfg.min_forward_ratio, min(1.0, ratio))
        self.phase = "APPROACH" if forward > 0 else "HOLD"
        self._forward = forward
        return MotionCommand(forward=forward, turn=self._turn)
