"""Frozen-reference, fail-closed single-person lock for the PC demo.

Identity continuity and permission to move are deliberately independent:
- the manual target ID/reference survive LOST and VERIFYING;
- only a current, unambiguous, *verified* observation permits motion;
- short evidence gaps may retain confirmation history, never motor permission.

This is a safety *gate*, not a claim of reliable biometric identity matching.
Appearance-only matching can still confuse similarly dressed people.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.following.target_lock import FollowState, LockResult, TargetObservation


def _unit(embedding: object) -> np.ndarray | None:
    if embedding is None:
        return None
    try:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size < 2 or not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else None


def _box_overlap(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection / smaller area, sensitive to partial-person occlusion."""
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if min(area_a, area_b) <= 0:
        return 1.0
    intersection = (max(0.0, min(ax2, bx2) - max(ax1, bx1))
                    * max(0.0, min(ay2, by2) - max(ay1, by1)))
    return intersection / min(area_a, area_b)


@dataclass(frozen=True)
class GuardConfig:
    confirm_frames: int = 5  # New ByteTrack ID: conservative confirmation.
    same_track_confirm_frames: int = 2  # Old trusted track after brief occlusion.
    min_final_clear_frames: int = 2  # Require consecutive clear frames at recovery.
    candidate_grace_frames: int = 2  # May skip this many uncertain frames.
    max_lost_frames: int = 150
    min_confidence: float = 0.35
    min_crop_quality: float = 0.30
    track_distance: float = 0.32
    same_track_recovery_distance: float = 0.28
    recovery_distance: float = 0.26
    ambiguity_margin: float = 0.08
    max_overlap: float = 0.35
    selection_max_overlap: float = 0.20
    motion_gate_in_heights: float = 3.5
    motion_window_frames: int = 30

    def __post_init__(self) -> None:
        frame_counts = (self.confirm_frames, self.same_track_confirm_frames,
                        self.min_final_clear_frames, self.max_lost_frames)
        if any(value < 1 for value in frame_counts) or self.candidate_grace_frames < 0:
            raise ValueError("Invalid confirmation/grace/timeout frame counts")
        if self.min_final_clear_frames > min(self.confirm_frames, self.same_track_confirm_frames):
            raise ValueError("min_final_clear_frames must fit each confirmation threshold")
        if not (0 <= self.min_confidence <= 1 and 0 <= self.min_crop_quality <= 1):
            raise ValueError("Confidence and quality thresholds must be in [0, 1]")
        if not (0 <= self.recovery_distance <= self.same_track_recovery_distance
                <= self.track_distance <= 2.0):
            raise ValueError("Require recovery <= same-track recovery <= track distance")
        if (self.ambiguity_margin < 0 or not 0 <= self.max_overlap <= 1
                or not 0 <= self.selection_max_overlap <= 1):
            raise ValueError("Invalid ambiguity/overlap thresholds")
        if self.motion_gate_in_heights <= 0 or self.motion_window_frames < 0:
            raise ValueError("Invalid motion gate")


@dataclass(frozen=True)
class GuardDiagnostics:
    """Read-only lock evidence; useful for UI and reproducible trace CSV."""
    identity_locked: bool
    movement_authorized: bool
    trusted_track_id: int | None
    candidate_track_id: int | None
    candidate_hits: int
    candidate_clear_streak: int
    candidate_required: int
    reference_distance: float | None
    competitor_margin: float | None
    candidate_overlap: float | None
    reason_code: str


class GuardedTargetLock:
    """Authorize following only after corroborated evidence from a frozen anchor.

    A short evidence interruption may preserve the *candidate* vote count, but
    the interruption always revokes movement immediately. New tracker IDs must
    pass longer confirmation than previously trusted IDs. Neither target ID
    nor reference embedding is ever updated automatically.
    """

    def __init__(self, config: GuardConfig | None = None) -> None:
        self.config = config or GuardConfig()
        self.reset()

    def reset(self) -> None:
        self.state = FollowState.IDLE
        self.target_id: int | None = None
        self._anchor: np.ndarray | None = None
        self._active_track: int | None = None
        self._last_frame = -1
        self._last_seen = -1  # Last *authorized* observation.
        self._last_bbox: tuple[float, ...] | None = None
        self._clear_candidate()
        self._reason = "No target selected"
        self._reason_code = "IDLE"
        self._distance_value: float | None = None
        self._margin_value: float | None = None
        self._overlap_value: float | None = None

    def _clear_candidate(self) -> None:
        self._candidate_track: int | None = None
        self._candidate_hits = 0
        self._candidate_streak = 0
        self._candidate_last_clear = -1
        self._candidate_required = 0

    @property
    def diagnostics(self) -> GuardDiagnostics:
        return GuardDiagnostics(
            identity_locked=self._anchor is not None and self.target_id is not None,
            movement_authorized=self.state in (FollowState.TRACKING, FollowState.REACQUIRED),
            trusted_track_id=self._active_track,
            candidate_track_id=self._candidate_track,
            candidate_hits=self._candidate_hits,
            candidate_clear_streak=self._candidate_streak,
            candidate_required=self._candidate_required,
            reference_distance=self._distance_value,
            competitor_margin=self._margin_value,
            candidate_overlap=self._overlap_value,
            reason_code=self._reason_code,
        )

    def _result(self, observation: TargetObservation | None = None) -> LockResult:
        return LockResult(self.state, self.target_id, observation, self._reason)

    def stop(self, reason: str = "Operator stop") -> LockResult:
        self._clear_candidate()
        self.state = FollowState.STOPPED
        self._reason = reason
        self._reason_code = "STOPPED"
        return self._result()

    def _usable(self, detection: Mapping) -> bool:
        return (float(detection.get("confidence", 0.0)) >= self.config.min_confidence
                and float(detection.get("crop_quality", 0.0)) >= self.config.min_crop_quality
                and _unit(detection.get("embedding")) is not None)

    @staticmethod
    def _overlap(detection: Mapping, detections: Sequence[Mapping]) -> float:
        return max((
            _box_overlap(detection["bbox"], other["bbox"])
            for other in detections
            if other is not detection and int(other["track_id"]) != int(detection["track_id"])
        ), default=0.0)

    def select(self, observation: TargetObservation, frame_id: int,
               detections: Sequence[Mapping] | None = None) -> LockResult:
        if frame_id < self._last_frame:
            raise ValueError("frame_id must be monotonic")
        detections = list(detections or [])
        matching = [d for d in detections if int(d["track_id"]) == observation.track_id]
        if len(matching) != 1 or int(matching[0]["target_id"]) != observation.target_id:
            raise ValueError("Selection must reference one current detection")
        detection = matching[0]
        if not self._usable(detection):
            raise ValueError("Cannot lock: weak/missing Re-ID feature or crop quality")
        if self._overlap(detection, detections) > self.config.selection_max_overlap:
            raise ValueError("Cannot lock: target already overlaps another person")
        anchor = _unit(detection["embedding"])
        assert anchor is not None
        self.reset()
        self.target_id = observation.target_id
        self._anchor = anchor.copy()  # Critical: no automatic gallery updates.
        self._active_track = observation.track_id
        self._last_frame = self._last_seen = int(frame_id)
        self._last_bbox = observation.bbox
        self.state = FollowState.TRACKING
        self._reason = "Manually selected; reference frozen"
        self._reason_code = "MANUAL_SELECTION"
        return self._result(observation)

    def _distance(self, detection: Mapping) -> float:
        vector = _unit(detection.get("embedding"))
        if self._anchor is None or vector is None or vector.shape != self._anchor.shape:
            return float("inf")
        return float(np.clip(1.0 - float(np.dot(self._anchor, vector)), 0.0, 2.0))

    def _motion_ok(self, detection: Mapping, frame_id: int) -> bool:
        if self._last_bbox is None or frame_id - self._last_seen > self.config.motion_window_frames:
            return True  # Long absence: depend on strict Re-ID and uniqueness.
        old, new = self._last_bbox, detection["bbox"]
        old_center = ((old[0] + old[2]) / 2, (old[1] + old[3]) / 2)
        new_center = ((new[0] + new[2]) / 2, (new[1] + new[3]) / 2)
        height = max(24.0, float(old[3] - old[1]), float(new[3] - new[1]))
        shift = np.hypot(new_center[0] - old_center[0], new_center[1] - old_center[1]) / height
        return float(shift) <= self.config.motion_gate_in_heights

    def _hold(self, reason: str, code: str, *, allow_grace: bool = False) -> LockResult:
        """Revoke permission now, optionally retain bounded non-actionable history."""
        if not allow_grace:
            self._clear_candidate()
        elif self._candidate_track is not None:
            # The last clear sample may be at most N uncertain frames ago.
            if self._last_frame - self._candidate_last_clear > self.config.candidate_grace_frames:
                self._clear_candidate()
            else:
                self._candidate_streak = 0
        self.state = FollowState.LOST
        self._reason = reason
        self._reason_code = code
        return self._result()

    def update(self, frame_id: int, detections: Sequence[Mapping]) -> LockResult:
        frame_id = int(frame_id)
        if frame_id <= self._last_frame:
            raise ValueError("Each frame_id must be strictly increasing")
        self._last_frame = frame_id
        if self.state in (FollowState.IDLE, FollowState.STOPPED):
            return self._result()
        if frame_id - self._last_seen > self.config.max_lost_frames:
            return self.stop("Target lost timeout; reselect manually")

        self._distance_value = self._margin_value = self._overlap_value = None
        detections = list(detections)
        # Duplicate tracker IDs corrupt evidence; never infer which instance is real.
        track_ids = [int(d["track_id"]) for d in detections]
        if len(track_ids) != len(set(track_ids)):
            return self._hold("Duplicate ByteTrack IDs: HOLD", "DUPLICATE_TRACK")

        ranked = sorted(((self._distance(d), d) for d in detections if self._usable(d)),
                        key=lambda pair: pair[0])
        if not ranked:
            return self._hold("No reliable appearance evidence: HOLD", "NO_EVIDENCE", allow_grace=True)
        best_dist, best = ranked[0]
        second_dist = ranked[1][0] if len(ranked) > 1 else float("inf")
        self._distance_value = best_dist
        self._margin_value = second_dist - best_dist if np.isfinite(second_dist) else None
        if second_dist - best_dist < self.config.ambiguity_margin:
            return self._hold("Multiple people look similar: HOLD", "AMBIGUOUS")

        same_track = int(best["track_id"]) == self._active_track
        was_authorized = self.state in (FollowState.TRACKING, FollowState.REACQUIRED)
        if same_track and was_authorized:
            threshold = self.config.track_distance
        elif same_track and frame_id - self._last_seen <= self.config.motion_window_frames:
            threshold = self.config.same_track_recovery_distance
        else:
            threshold = self.config.recovery_distance
        if best_dist > threshold:
            return self._hold(f"Appearance mismatch ({best_dist:.3f}): HOLD", "APPEARANCE_MISMATCH")

        self._overlap_value = self._overlap(best, detections)
        if self._overlap_value > self.config.max_overlap:
            # An overlap is not trustworthy evidence, but does not force a
            # candidate's *prior* unambiguous observations to be forgotten.
            return self._hold("People overlap / occlusion: HOLD", "OCCLUDED", allow_grace=True)
        if not self._motion_ok(best, frame_id):
            return self._hold("Candidate violates motion gate: HOLD", "MOTION_MISMATCH")

        observation = TargetObservation.from_detection(best)
        if same_track and was_authorized:
            self._clear_candidate()
            self.state = FollowState.TRACKING
            self._last_seen = frame_id
            self._last_bbox = observation.bbox
            self._reason = f"Frozen-reference match d={best_dist:.3f}"
            self._reason_code = "TRACK_VERIFIED"
            return self._result(observation)

        required = (self.config.same_track_confirm_frames if same_track
                    else self.config.confirm_frames)
        if (self._candidate_track != observation.track_id
                or frame_id - self._candidate_last_clear - 1 > self.config.candidate_grace_frames):
            self._clear_candidate()
            self._candidate_track = observation.track_id
            self._candidate_required = required
        else:
            self._candidate_required = required
        uninterrupted = frame_id == self._candidate_last_clear + 1
        self._candidate_hits += 1
        self._candidate_streak = self._candidate_streak + 1 if uninterrupted else 1
        self._candidate_last_clear = frame_id
        self.state = FollowState.VERIFYING
        self._reason = (f"Re-ID B{observation.track_id}: {self._candidate_hits}/{required} "
                        f"(clear {self._candidate_streak})")
        self._reason_code = "VERIFYING"
        if (self._candidate_hits < required
                or self._candidate_streak < self.config.min_final_clear_frames):
            return self._result()

        self._active_track = observation.track_id
        self._last_seen = frame_id
        self._last_bbox = observation.bbox
        self._clear_candidate()
        self.state = FollowState.REACQUIRED
        self._reason = f"Recovered with verified appearance d={best_dist:.3f}"
        self._reason_code = "RECOVERY_VERIFIED"
        return self._result(observation)
