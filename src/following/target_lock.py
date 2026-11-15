"""Fail-closed target lock built on a *persistent* identity-manager ID.

The upstream V5/V5.1 manager performs appearance/motion association. This
module does not independently run Re-ID; it adds confirmation before the
controller can resume after any gap. Never silently select a different ID.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence


class FollowState(str, Enum):
    IDLE = "IDLE"
    TRACKING = "TRACKING"
    LOST = "LOST"
    VERIFYING = "VERIFYING"
    REACQUIRED = "REACQUIRED"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class TargetObservation:
    target_id: int
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    association: str = ""

    @classmethod
    def from_detection(cls, detection: Mapping) -> "TargetObservation":
        return cls(
            target_id=int(detection["target_id"]),
            track_id=int(detection["track_id"]),
            bbox=tuple(float(v) for v in detection["bbox"]),
            confidence=float(detection.get("confidence", 1.0)),
            association=str(detection.get("association", "")),
        )


@dataclass(frozen=True)
class LockResult:
    state: FollowState
    target_id: Optional[int]
    observation: Optional[TargetObservation]
    reason: str

    @property
    def may_follow(self) -> bool:
        return self.state in (FollowState.TRACKING, FollowState.REACQUIRED) and self.observation is not None


class TargetLockManager:
    """Hold one selected persistent ID until explicitly reset or reselected.

    Movement is disabled during LOST, VERIFYING and STOPPED. An observed
    reacquisition must be consistent for N frames. A duplicate target ID on
    different detections is ambiguous and never eligible for following.
    """

    def __init__(
        self,
        recovery_confirm_frames: int = 3,
        max_lost_frames: int = 150,
        min_confidence: float = 0.35,
    ) -> None:
        if recovery_confirm_frames < 1 or max_lost_frames < 1:
            raise ValueError("Frame limits must be positive")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        self.confirm_frames = int(recovery_confirm_frames)
        self.max_lost_frames = int(max_lost_frames)
        self.min_confidence = float(min_confidence)
        self.reset()

    def reset(self) -> None:
        self.state = FollowState.IDLE
        self.target_id: Optional[int] = None
        self._last_frame = -1
        self._last_seen = -1
        self._candidate_track_id: Optional[int] = None
        self._active_track_id: Optional[int] = None
        self._confirm_count = 0
        self._reason = "No target selected"

    def select(self, observation: TargetObservation, frame_id: int) -> LockResult:
        if not 0.0 <= observation.confidence <= 1.0:
            raise ValueError("Invalid detection confidence")
        if observation.confidence < self.min_confidence:
            raise ValueError("Selected detection has insufficient confidence")
        if frame_id < self._last_frame:
            raise ValueError("frame_id must be monotonic")
        self.target_id = observation.target_id
        self._last_frame = self._last_seen = int(frame_id)
        self._candidate_track_id = None
        self._active_track_id = observation.track_id
        self._confirm_count = 0
        self.state = FollowState.TRACKING
        self._reason = "Target selected"
        return self._result(observation)

    def stop(self, reason: str = "Operator stop") -> LockResult:
        self.state = FollowState.STOPPED
        self._reason = reason
        return self._result(None)

    def _result(self, observation: Optional[TargetObservation]) -> LockResult:
        return LockResult(self.state, self.target_id, observation, self._reason)

    def update(self, frame_id: int, detections: Sequence[Mapping]) -> LockResult:
        frame_id = int(frame_id)
        if frame_id <= self._last_frame:
            raise ValueError("Each frame_id must be strictly increasing")
        self._last_frame = frame_id
        if self.state in (FollowState.IDLE, FollowState.STOPPED):
            return self._result(None)
        if self.state in (FollowState.LOST, FollowState.VERIFYING) and (
            frame_id - self._last_seen > self.max_lost_frames
        ):
            self.state = FollowState.STOPPED
            self._reason = "Target lost timeout"
            return self._result(None)

        candidates = [
            TargetObservation.from_detection(det)
            for det in detections if int(det["target_id"]) == self.target_id
            and float(det.get("confidence", 1.0)) >= self.min_confidence
        ]
        if len(candidates) != 1:
            self._candidate_track_id = None
            self._confirm_count = 0
            if frame_id - self._last_seen > self.max_lost_frames:
                self.state = FollowState.STOPPED
                self._reason = "Target lost timeout"
            else:
                self.state = FollowState.LOST
                self._reason = "Target missing or ambiguous"
            return self._result(None)

        candidate = candidates[0]
        if (self.state in (FollowState.TRACKING, FollowState.REACQUIRED)
                and candidate.track_id == self._active_track_id):
            self._last_seen = frame_id
            self.state = FollowState.TRACKING
            self._reason = "Persistent target confirmed"
            return self._result(candidate)

        # Recoveries (or tracker-ID changes without a missing frame) require
        # repeated confirmation before any motion is requested.
        if candidate.track_id == self._candidate_track_id:
            self._confirm_count += 1
        else:
            self._candidate_track_id = candidate.track_id
            self._confirm_count = 1
        if self._confirm_count < self.confirm_frames:
            self.state = FollowState.VERIFYING
            self._reason = f"Confirming recovery {self._confirm_count}/{self.confirm_frames}"
            return self._result(None)
        self._last_seen = frame_id
        self._active_track_id = candidate.track_id
        self._candidate_track_id = None
        self._confirm_count = 0
        self.state = FollowState.REACQUIRED
        self._reason = "Recovery confirmed by repeated persistent-ID observations"
        return self._result(candidate)
