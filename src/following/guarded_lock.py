"""Frozen-reference, fail-closed single-person lock for the PC demo.

Identity continuity and permission to move are deliberately independent:
- the manual target ID/reference survive LOST and VERIFYING;
- only a current, unambiguous, *verified* observation permits motion;
- short evidence gaps may retain confirmation history, never motor permission.

This is a safety *gate*, not a claim of reliable biometric identity matching.
Appearance-only matching can still confuse similarly dressed people.
"""

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.following.target_lock import FollowState, LockResult, TargetObservation


def _unit(embedding: object) -> Optional[np.ndarray]:
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


def _same_person_part(small: Sequence[float], big: Sequence[float]) -> bool:
    """A second detector box on one person: inside the other, sharing its top (head)."""
    big_h = float(big[3]) - float(big[1])
    small_h = float(small[3]) - float(small[1])
    centre = (float(small[0]) + float(small[2])) / 2
    return (0 < small_h < big_h and _box_overlap(small, big) >= 0.9
            and abs(float(small[1]) - float(big[1])) <= 0.1 * big_h
            and big[0] + 0.25 * (big[2] - big[0]) <= centre <= big[2] - 0.25 * (big[2] - big[0]))


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
    max_views: int = 4  # Operator-added reference views (front, side, back...).
    # Real-robot additions (27/11 corridor log: a single frozen look lost the same person whenever
    # the light or the pose changed). All default off, so the MOT17-tuned behaviour is unchanged.
    continuity_distance: float = 0.0  # >0: the trusted track, verified last frame, may drift up to this
    auto_views: int = 0               # extra looks added while verified; each must match the operator's
    auto_view_interval: int = 21      # frames between two automatic looks (~1.5 s)
    auto_view_min_distance: float = 0.18  # only a look this different from every stored one is worth adding
    blind_frames: int = 0             # trusted track seen but without usable Re-ID: keep it this many frames
    blind_min_confidence: float = 0.10
    drop_part_boxes: bool = False     # 27/11: YOLO adds an upper-body box on the same person -> OCCLUDED

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
        if self.max_views < 1:
            raise ValueError("max_views must be >= 1")
        if self.continuity_distance and not self.track_distance <= self.continuity_distance <= 2.0:
            raise ValueError("continuity_distance must be 0 (off) or >= track_distance")
        if min(self.auto_views, self.blind_frames) < 0 or self.auto_view_interval < 1:
            raise ValueError("auto_views and blind_frames must be >= 0, auto_view_interval >= 1")


@dataclass(frozen=True)
class GuardDiagnostics:
    """Read-only lock evidence; useful for UI and reproducible trace CSV."""
    identity_locked: bool
    movement_authorized: bool
    trusted_track_id: Optional[int]
    candidate_track_id: Optional[int]
    candidate_hits: int
    candidate_clear_streak: int
    candidate_required: int
    reference_distance: Optional[float]
    competitor_margin: Optional[float]
    candidate_overlap: Optional[float]
    reason_code: str


class GuardedTargetLock:
    """Authorize following only after corroborated evidence from a frozen anchor.

    A short evidence interruption may preserve the *candidate* vote count, but
    the interruption always revokes movement immediately. New tracker IDs must
    pass longer confirmation than previously trusted IDs. Neither target ID
    nor reference embedding is updated automatically by default; the operator may
    add views of the *currently verified* target with add_view(). On the robot
    (GuardConfig.auto_views > 0) up to auto_views extra looks are added while the
    target is verified, alone and unambiguous, and only if the look still matches the
    operator's own views within track_distance, so the reference cannot drift to
    someone else. continuity_distance and blind_frames relax only the *trusted, continuous*
    track; a new or recovered track still needs the strict recovery thresholds.
    """

    def __init__(self, config: Optional[GuardConfig] = None) -> None:
        self.config = config or GuardConfig()
        self.reset()

    def reset(self) -> None:
        self.state = FollowState.IDLE
        self.target_id: Optional[int] = None
        self._anchors: List[np.ndarray] = []
        self._auto_anchors: List[np.ndarray] = []
        self._next_auto_view = 0
        self._active_track: Optional[int] = None
        self._last_frame = -1
        self._last_seen = -1  # Last *authorized* observation.
        self._last_bbox: Optional[Tuple[float, ...]] = None
        self._clear_candidate()
        self._reason = "No target selected"
        self._reason_code = "IDLE"
        self._distance_value: Optional[float] = None
        self._margin_value: Optional[float] = None
        self._overlap_value: Optional[float] = None

    def _clear_candidate(self) -> None:
        self._candidate_track: Optional[int] = None
        self._candidate_hits = 0
        self._candidate_streak = 0
        self._candidate_last_clear = -1
        self._candidate_required = 0

    @property
    def diagnostics(self) -> GuardDiagnostics:
        return GuardDiagnostics(
            identity_locked=bool(self._anchors) and self.target_id is not None,
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

    def _result(self, observation: Optional[TargetObservation] = None) -> LockResult:
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
               detections: Optional[Sequence[Mapping]] = None) -> LockResult:
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
        self._anchors = [anchor.copy()]  # operator looks; automatic ones only with GuardConfig.auto_views
        self._active_track = observation.track_id
        self._last_frame = self._last_seen = int(frame_id)
        self._last_bbox = observation.bbox
        self.state = FollowState.TRACKING
        self._reason = "Manually selected; reference frozen"
        self._reason_code = "MANUAL_SELECTION"
        return self._result(observation)

    def _distance(self, detection: Mapping) -> float:
        vector = _unit(detection.get("embedding"))
        if not self._anchors or vector is None or vector.shape != self._anchors[0].shape:
            return float("inf")
        best = max(float(np.dot(anchor, vector)) for anchor in self._anchors + self._auto_anchors)
        return float(np.clip(1.0 - best, 0.0, 2.0))

    def _operator_distance(self, vector: np.ndarray) -> float:
        return float(1.0 - max(float(np.dot(anchor, vector)) for anchor in self._anchors))

    def _maybe_add_auto_view(self, frame_id: int, detection: Mapping, distance: float,
                             margin: float, detections: Sequence[Mapping]) -> None:
        cfg = self.config
        vector = _unit(detection["embedding"])
        if (not cfg.auto_views or frame_id < self._next_auto_view or vector is None
                or distance < cfg.auto_view_min_distance or margin < 2 * cfg.ambiguity_margin
                or self._overlap(detection, detections) > cfg.selection_max_overlap
                or self._operator_distance(vector) > cfg.track_distance):
            return
        self._auto_anchors = (self._auto_anchors + [vector.copy()])[-cfg.auto_views:]  # oldest out
        self._next_auto_view = frame_id + cfg.auto_view_interval

    def _drop_part_boxes(self, detections: List[Mapping]) -> List[Mapping]:
        """Keep one box per person: the trusted track's if it is one of the pair, else the larger."""
        drop = set()
        for i, a in enumerate(detections):
            for j, b in enumerate(detections):
                if i != j and _same_person_part(a["bbox"], b["bbox"]):
                    drop.add(j if int(a["track_id"]) == self._active_track else i)  # a inside b
        return [d for k, d in enumerate(detections) if k not in drop]

    def _blind_continue(self, frame_id: int, detections: Sequence[Mapping]) -> Optional[LockResult]:
        """Trusted track in view but without usable Re-ID (backlight, half hidden): keep it briefly."""
        cfg = self.config
        if (not cfg.blind_frames or self.state not in (FollowState.TRACKING, FollowState.REACQUIRED)
                or frame_id - self._last_seen > cfg.blind_frames):
            return None
        trusted = [d for d in detections if int(d["track_id"]) == self._active_track]
        if (len(trusted) != 1 or self._usable(trusted[0])
                or float(trusted[0].get("confidence", 0.0)) < cfg.blind_min_confidence
                or self._overlap(trusted[0], detections) > cfg.max_overlap
                or not self._motion_ok(trusted[0], frame_id)
                or any(self._usable(d) and self._distance(d) <= cfg.track_distance
                       for d in detections if d is not trusted[0])):
            return None
        observation = TargetObservation.from_detection(trusted[0])
        self._last_bbox = observation.bbox  # _last_seen stays: the blind stretch is bounded
        self.state = FollowState.TRACKING
        self._reason = f"Same track without Re-ID ({frame_id - self._last_seen}/{cfg.blind_frames})"
        self._reason_code = "TRACK_CONTINUED"
        return self._result(observation)

    @property
    def view_count(self) -> int:
        return len(self._anchors)

    def add_view(self, frame_id: int, detections: Sequence[Mapping]) -> int:
        """Operator-confirmed: add the target's current look (e.g. its back).

        Only for the trusted track verified in this very frame, alone in view;
        the operator must confirm on screen that the box is the right person.
        """
        if (self.state not in (FollowState.TRACKING, FollowState.REACQUIRED)
                or int(frame_id) != self._last_seen):
            raise ValueError("Add a view only while the target is verified (TRACKING)")
        if len(self._anchors) >= self.config.max_views:
            raise ValueError(f"Already {len(self._anchors)} views; reselect to start over")
        detections = list(detections)
        matching = [d for d in detections if int(d["track_id"]) == self._active_track]
        if len(matching) != 1 or not self._usable(matching[0]):
            raise ValueError("Target view is weak or missing in this frame")
        if self._overlap(matching[0], detections) > self.config.selection_max_overlap:
            raise ValueError("Target overlaps another person; add the view when alone")
        anchor = _unit(matching[0]["embedding"])
        assert anchor is not None
        self._anchors.append(anchor.copy())
        return len(self._anchors)

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

        if self.config.drop_part_boxes:
            detections = self._drop_part_boxes(detections)
        blind = self._blind_continue(frame_id, detections)
        if blind is not None:
            return blind
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
            threshold = max(self.config.track_distance, self.config.continuity_distance)
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
            self._maybe_add_auto_view(frame_id, best, best_dist,
                                      second_dist - best_dist, detections)
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
