from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


def box_center(box):
    x1, y1, x2, y2 = box
    return np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def box_height(box):
    return max(1.0, float(box[3] - box[1]))


def cosine_distance(a, b):
    if a is None or b is None:
        return 1.0
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


@dataclass
class TargetProfile:
    target_id: int
    bbox: np.ndarray
    center: np.ndarray
    last_frame: int
    current_track_id: int
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    embedding: np.ndarray | None = None
    track_ids: list[int] = field(default_factory=list)
    status: str = "Active"


class IdentityManager:
    """
    Baseline: raw ByteTrack ID.
    V1: motion-only matching for a newly created ByteTrack ID.
    V2: OSNet appearance matching, with a loose motion plausibility gate.
    V3: weighted appearance + motion matching.
    """

    def __init__(
        self, mode, fps=30.0, max_lost_seconds=3.0,
        max_center_distance=180.0, max_height_ratio=2.0,
        appearance_threshold=0.35, motion_weight=0.45,
        appearance_weight=0.55, max_motion_speed=12.0
    ):
        self.mode = mode
        self.fps = max(1.0, float(fps))
        self.max_lost_frames = max(1, int(float(max_lost_seconds) * self.fps))
        self.max_center_distance = float(max_center_distance)
        self.max_height_ratio = float(max_height_ratio)
        self.appearance_threshold = float(appearance_threshold)
        self.motion_weight = float(motion_weight)
        self.appearance_weight = float(appearance_weight)
        self.max_motion_speed = float(max_motion_speed)
        self.next_target_id = 1
        self.targets: dict[int, TargetProfile] = {}
        self.track_to_target: dict[int, int] = {}
        self.associations: dict[int, str] = {}

    def has_track(self, track_id):
        return int(track_id) in self.track_to_target

    def status_of(self, target_id):
        p = self.targets.get(int(target_id))
        return p.status if p else "Unknown"

    def association_of(self, track_id):
        return self.associations.get(int(track_id), "existing")

    def _new_target(self, det, frame_id, embedding):
        target_id = self.next_target_id
        self.next_target_id += 1
        bbox = np.asarray(det["bbox"], dtype=np.float32)
        track_id = int(det["track_id"])
        emb = None if embedding is None else np.asarray(embedding, dtype=np.float32).copy()
        self.targets[target_id] = TargetProfile(
            target_id=target_id,
            bbox=bbox,
            center=box_center(bbox),
            last_frame=frame_id,
            current_track_id=track_id,
            embedding=emb,
            track_ids=[track_id],
        )
        self.track_to_target[track_id] = target_id
        self.associations[track_id] = "new"
        return target_id

    def _update_profile(self, profile, det, frame_id, embedding):
        bbox = np.asarray(det["bbox"], dtype=np.float32)
        center = box_center(bbox)
        dt = max(1, frame_id - profile.last_frame)
        measured_velocity = (center - profile.center) / dt
        profile.velocity = 0.65 * profile.velocity + 0.35 * measured_velocity
        profile.bbox = bbox
        profile.center = center
        profile.last_frame = frame_id
        profile.current_track_id = int(det["track_id"])
        profile.status = "Active"

        track_id = int(det["track_id"])
        if not profile.track_ids or profile.track_ids[-1] != track_id:
            profile.track_ids.append(track_id)
        self.track_to_target[track_id] = profile.target_id

        if embedding is not None:
            emb = np.asarray(embedding, dtype=np.float32)
            if profile.embedding is None:
                profile.embedding = emb.copy()
            else:
                updated = 0.85 * profile.embedding + 0.15 * emb
                norm = np.linalg.norm(updated)
                profile.embedding = updated / norm if norm > 1e-12 else updated

    def _motion_distance(self, profile, bbox, frame_id):
        gap = max(1, frame_id - profile.last_frame)
        predicted = profile.center + profile.velocity * gap
        current = box_center(bbox)
        distance = float(np.linalg.norm(current - predicted))
        allowed = self.max_center_distance + self.max_motion_speed * gap
        ratio = box_height(bbox) / box_height(profile.bbox)
        height_ok = (1.0 / self.max_height_ratio) <= ratio <= self.max_height_ratio
        return distance, allowed, height_ok

    def update(self, frame_id, detections, embeddings=None):
        embeddings = embeddings or {}
        if self.mode == "baseline":
            return {int(d["track_id"]): int(d["track_id"]) for d in detections}

        present_tracks = {int(d["track_id"]) for d in detections}

        # Update status of active profiles before matching new tracker IDs.
        for profile in self.targets.values():
            if profile.status == "Expired":
                continue
            gap = frame_id - profile.last_frame
            if gap > self.max_lost_frames:
                profile.status = "Expired"
            elif profile.current_track_id not in present_tracks:
                profile.status = "Lost"

        assigned = {}
        used_targets = set()

        # Pass 1: retain existing ByteTrack-to-target mappings.
        for det in detections:
            track_id = int(det["track_id"])
            target_id = self.track_to_target.get(track_id)
            profile = self.targets.get(target_id) if target_id is not None else None
            if profile is None or profile.status == "Expired":
                continue
            self._update_profile(profile, det, frame_id, embeddings.get(track_id))
            assigned[track_id] = target_id
            used_targets.add(target_id)
            self.associations[track_id] = "existing"

        # Pass 2: recover a lost target when ByteTrack creates a new track ID.
        for det in detections:
            track_id = int(det["track_id"])
            if track_id in assigned:
                continue
            embedding = embeddings.get(track_id)
            candidates = []
            bbox = np.asarray(det["bbox"], dtype=np.float32)

            for target_id, profile in self.targets.items():
                if target_id in used_targets or profile.status != "Lost":
                    continue
                gap = frame_id - profile.last_frame
                if gap <= 0 or gap > self.max_lost_frames:
                    continue
                motion_dist, motion_allowed, height_ok = self._motion_distance(
                    profile, bbox, frame_id
                )
                if not height_ok or motion_dist > motion_allowed:
                    continue
                motion_score = min(1.0, motion_dist / max(1.0, motion_allowed))
                appearance_score = cosine_distance(profile.embedding, embedding)

                if self.mode == "v1":
                    # Conservative motion-only recovery.
                    accepted = motion_dist <= self.max_center_distance
                    score = motion_score
                elif self.mode == "v2":
                    # Appearance is decisive; motion is a broad physical plausibility gate.
                    accepted = (
                        embedding is not None
                        and profile.embedding is not None
                        and appearance_score <= self.appearance_threshold
                    )
                    score = appearance_score
                else:
                    # V3 requires usable appearance and combines both cues.
                    accepted = (
                        embedding is not None
                        and profile.embedding is not None
                        and appearance_score <= self.appearance_threshold
                    )
                    score = (
                        self.motion_weight * motion_score
                        + self.appearance_weight * appearance_score
                    )
                    accepted = accepted and score <= self.appearance_threshold

                if accepted:
                    candidates.append((float(score), target_id))

            if candidates:
                _, target_id = min(candidates, key=lambda x: x[0])
                profile = self.targets[target_id]
                self._update_profile(profile, det, frame_id, embedding)
                self.associations[track_id] = (
                    "motion" if self.mode == "v1"
                    else "appearance" if self.mode == "v2"
                    else "appearance_motion_fusion"
                )
            else:
                target_id = self._new_target(det, frame_id, embedding)

            assigned[track_id] = target_id
            used_targets.add(target_id)

        return assigned
