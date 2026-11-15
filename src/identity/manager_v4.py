from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - fallback for minimal Jetson environments
    linear_sum_assignment = None


def _box_center(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    return np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def _box_height(box: np.ndarray) -> float:
    return max(1.0, float(box[3] - box[1]))


def _normalise_embedding(embedding) -> Optional[np.ndarray]:
    if embedding is None:
        return None
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.size == 0 or not np.isfinite(norm) or norm < 1e-12:
        return None
    return vector / norm


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Inputs are normalised when they enter the gallery.
    return float(np.clip(1.0 - np.dot(a, b), 0.0, 2.0))


@dataclass
class TargetProfileV4:
    target_id: int
    bbox: np.ndarray
    center: np.ndarray
    last_frame: int
    current_track_id: Optional[int]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    gallery: deque = field(default_factory=deque)
    status: str = "Active"
    hits: int = 1
    last_confidence: float = 1.0


class IdentityManagerV4:
    """Long-term identity association with exclusive track ownership.

    Design decisions compared with the original V1-V3 manager:
      * each target has at most one currently-owned ByteTrack ID;
      * lost-target recovery is solved globally (Hungarian assignment), not
        greedily in detection iteration order;
      * appearance is matched against a small feature gallery;
      * motion is normalised by person height and elapsed time;
      * ambiguous or implausible matches are rejected instead of forced.

    The manager only assigns persistent IDs. It does not choose the human that
    the robot should follow; target selection and robot control are separate
    system layers.
    """

    def __init__(
        self,
        fps: float = 30.0,
        max_lost_seconds: float = 20.0,
        max_height_ratio: float = 1.9,
        appearance_gate: float = 0.48,
        active_appearance_gate: float = 0.62,
        appearance_weight: float = 0.72,
        motion_weight: float = 0.28,
        motion_gate_base: float = 0.75,
        motion_gate_per_second: float = 0.12,
        max_motion_gate: float = 2.8,
        max_fusion_cost: float = 0.52,
        ambiguity_margin: float = 0.035,
        gallery_size: int = 8,
        gallery_update_gate: float = 0.42,
        min_embedding_confidence: float = 0.35,
        min_box_height: float = 24.0,
    ):
        self.fps = max(1.0, float(fps))
        self.max_lost_frames = max(1, int(float(max_lost_seconds) * self.fps))
        self.max_height_ratio = max(1.01, float(max_height_ratio))
        self.appearance_gate = float(appearance_gate)
        self.active_appearance_gate = float(active_appearance_gate)
        total_weight = max(1e-8, float(appearance_weight) + float(motion_weight))
        self.appearance_weight = float(appearance_weight) / total_weight
        self.motion_weight = float(motion_weight) / total_weight
        self.motion_gate_base = float(motion_gate_base)
        self.motion_gate_per_second = float(motion_gate_per_second)
        self.max_motion_gate = float(max_motion_gate)
        self.max_fusion_cost = float(max_fusion_cost)
        self.ambiguity_margin = max(0.0, float(ambiguity_margin))
        self.gallery_size = max(1, int(gallery_size))
        self.gallery_update_gate = float(gallery_update_gate)
        self.min_embedding_confidence = float(min_embedding_confidence)
        self.min_box_height = float(min_box_height)

        self.next_target_id = 1
        self.targets: dict[int, TargetProfileV4] = {}
        # Only the currently-owned track ID is present in this map.
        self.track_to_target: dict[int, int] = {}
        self.associations: dict[int, str] = {}
        self.counters = {
            "new_targets": 0,
            "reidentified": 0,
            "active_conflicts": 0,
            "rejected_appearance": 0,
            "rejected_motion": 0,
            "rejected_ambiguous": 0,
        }

    def has_track(self, track_id: int) -> bool:
        return int(track_id) in self.track_to_target

    def status_of(self, target_id: int) -> str:
        profile = self.targets.get(int(target_id))
        return profile.status if profile else "Unknown"

    def association_of(self, track_id: int) -> str:
        return self.associations.get(int(track_id), "existing")

    def diagnostics(self) -> dict:
        active = sum(p.status == "Active" for p in self.targets.values())
        lost = sum(p.status == "Lost" for p in self.targets.values())
        expired = sum(p.status == "Expired" for p in self.targets.values())
        return {
            **self.counters,
            "profiles_total": len(self.targets),
            "profiles_active": active,
            "profiles_lost": lost,
            "profiles_expired": expired,
        }

    def _add_gallery_sample(self, profile: TargetProfileV4, embedding) -> None:
        vector = _normalise_embedding(embedding)
        if vector is None:
            return
        if profile.gallery:
            distances = [_cosine_distance(vector, old) for old in profile.gallery]
            # Do not let a poor crop / switched track poison the identity gallery.
            if min(distances) > self.gallery_update_gate:
                return
        profile.gallery.append(vector.copy())
        while len(profile.gallery) > self.gallery_size:
            profile.gallery.popleft()

    def _gallery_distance(self, profile: TargetProfileV4, embedding) -> Optional[float]:
        vector = _normalise_embedding(embedding)
        if vector is None or not profile.gallery:
            return None
        distances = sorted(_cosine_distance(vector, old) for old in profile.gallery)
        # Average the closest few samples: more robust than comparing to one EMA.
        count = min(3, len(distances))
        return float(np.mean(distances[:count]))

    def _new_target(self, frame_id: int, det: dict, embedding) -> int:
        target_id = self.next_target_id
        self.next_target_id += 1
        bbox = np.asarray(det["bbox"], dtype=np.float32)
        track_id = int(det["track_id"])
        profile = TargetProfileV4(
            target_id=target_id,
            bbox=bbox,
            center=_box_center(bbox),
            last_frame=int(frame_id),
            current_track_id=track_id,
            last_confidence=float(det.get("confidence", 1.0)),
        )
        if (
            float(det.get("confidence", 1.0)) >= self.min_embedding_confidence
            and _box_height(bbox) >= self.min_box_height
        ):
            self._add_gallery_sample(profile, embedding)
        self.targets[target_id] = profile
        self.track_to_target[track_id] = target_id
        self.associations[track_id] = "new"
        self.counters["new_targets"] += 1
        return target_id

    def _bind_track(self, profile: TargetProfileV4, track_id: int) -> None:
        # Remove stale ownership first, including any duplicated legacy mapping.
        for old_track, old_target in list(self.track_to_target.items()):
            if old_target == profile.target_id:
                del self.track_to_target[old_track]
        profile.current_track_id = int(track_id)
        self.track_to_target[int(track_id)] = profile.target_id

    def _update_profile(self, profile: TargetProfileV4, frame_id: int, det: dict, embedding) -> None:
        bbox = np.asarray(det["bbox"], dtype=np.float32)
        center = _box_center(bbox)
        dt = max(1, int(frame_id) - int(profile.last_frame))
        measured_velocity = (center - profile.center) / float(dt)
        # A smoothed constant-velocity estimate, less sensitive to one noisy box.
        profile.velocity = 0.70 * profile.velocity + 0.30 * measured_velocity
        profile.bbox = bbox
        profile.center = center
        profile.last_frame = int(frame_id)
        profile.status = "Active"
        profile.hits += 1
        profile.last_confidence = float(det.get("confidence", 1.0))
        self._bind_track(profile, int(det["track_id"]))

        confidence = float(det.get("confidence", 1.0))
        if (
            confidence >= self.min_embedding_confidence
            and _box_height(bbox) >= self.min_box_height
        ):
            self._add_gallery_sample(profile, embedding)

    def _motion_cost(self, profile: TargetProfileV4, bbox: np.ndarray, frame_id: int):
        gap_frames = max(1, int(frame_id) - int(profile.last_frame))
        gap_seconds = gap_frames / self.fps
        predicted = profile.center + profile.velocity * gap_frames
        observed = _box_center(bbox)
        scale = max(24.0, 0.5 * (_box_height(profile.bbox) + _box_height(bbox)))
        normalised_error = float(np.linalg.norm(observed - predicted) / scale)
        allowed = min(
            self.max_motion_gate,
            self.motion_gate_base + self.motion_gate_per_second * gap_seconds,
        )
        height_ratio = _box_height(bbox) / _box_height(profile.bbox)
        height_ok = (1.0 / self.max_height_ratio) <= height_ratio <= self.max_height_ratio
        return normalised_error, allowed, height_ok

    @staticmethod
    def _solve_assignment(cost_matrix: np.ndarray):
        if cost_matrix.size == 0:
            return [], []
        if linear_sum_assignment is not None:
            return linear_sum_assignment(cost_matrix)

        # Deterministic fallback; scipy is recommended for proper global matching.
        pairs = []
        used_rows, used_cols = set(), set()
        candidates = [
            (float(cost_matrix[r, c]), r, c)
            for r in range(cost_matrix.shape[0])
            for c in range(cost_matrix.shape[1])
        ]
        for cost, row, col in sorted(candidates):
            if row not in used_rows and col not in used_cols:
                pairs.append((row, col))
                used_rows.add(row)
                used_cols.add(col)
        if not pairs:
            return [], []
        rows, cols = zip(*pairs)
        return np.asarray(rows), np.asarray(cols)

    def update(self, frame_id: int, detections: list[dict], embeddings=None) -> dict[int, int]:
        embeddings = embeddings or {}
        frame_id = int(frame_id)
        detections = list(detections)
        present_track_ids = {int(d["track_id"]) for d in detections}

        # Update lost/expired state before association.
        for profile in self.targets.values():
            if profile.status == "Expired":
                continue
            gap = frame_id - profile.last_frame
            if gap > self.max_lost_frames:
                profile.status = "Expired"
                if profile.current_track_id is not None:
                    self.track_to_target.pop(profile.current_track_id, None)
                    profile.current_track_id = None
            elif profile.current_track_id not in present_track_ids:
                profile.status = "Lost"

        assigned: dict[int, int] = {}
        used_targets: set[int] = set()
        unmatched_detections: list[dict] = []

        # Pass 1: keep reliable, currently-owned ByteTrack identities.
        # This pass is one-to-one by construction; stale mappings are deleted.
        for det in detections:
            track_id = int(det["track_id"])
            target_id = self.track_to_target.get(track_id)
            profile = self.targets.get(target_id) if target_id is not None else None
            if profile is None or profile.status == "Expired" or target_id in used_targets:
                if target_id is not None and target_id in used_targets:
                    self.track_to_target.pop(track_id, None)
                unmatched_detections.append(det)
                continue

            embedding = embeddings.get(track_id)
            app_distance = self._gallery_distance(profile, embedding)
            confidence = float(det.get("confidence", 1.0))
            box = np.asarray(det["bbox"], dtype=np.float32)
            quality_ok = confidence >= self.min_embedding_confidence and _box_height(box) >= self.min_box_height
            if quality_ok and app_distance is not None and app_distance > self.active_appearance_gate:
                # Suspect ByteTrack ID switch: release the old identity and try
                # a global re-association instead of corrupting its gallery.
                self.track_to_target.pop(track_id, None)
                profile.status = "Lost"
                profile.current_track_id = None
                unmatched_detections.append(det)
                self.counters["active_conflicts"] += 1
                continue

            self._update_profile(profile, frame_id, det, embedding)
            assigned[track_id] = profile.target_id
            used_targets.add(profile.target_id)
            self.associations[track_id] = "existing"

        if not unmatched_detections:
            return assigned

        # Pass 2: create one global cost matrix for all new tracks and lost IDs.
        lost_profiles = [
            p for p in self.targets.values()
            if p.status == "Lost" and p.target_id not in used_targets and p.gallery
            and (frame_id - p.last_frame) <= self.max_lost_frames
        ]
        if not lost_profiles:
            for det in unmatched_detections:
                tid = int(det["track_id"])
                target_id = self._new_target(frame_id, det, embeddings.get(tid))
                assigned[tid] = target_id
            return assigned

        invalid_cost = 1e6
        cost_matrix = np.full((len(unmatched_detections), len(lost_profiles)), invalid_cost, dtype=np.float64)
        components: dict[tuple[int, int], tuple[float, float, float]] = {}
        for row, det in enumerate(unmatched_detections):
            track_id = int(det["track_id"])
            embedding = embeddings.get(track_id)
            bbox = np.asarray(det["bbox"], dtype=np.float32)
            confidence = float(det.get("confidence", 1.0))
            if embedding is None or confidence < self.min_embedding_confidence or _box_height(bbox) < self.min_box_height:
                continue
            for col, profile in enumerate(lost_profiles):
                appearance = self._gallery_distance(profile, embedding)
                if appearance is None or appearance > self.appearance_gate:
                    self.counters["rejected_appearance"] += 1
                    continue
                motion_error, motion_allowed, height_ok = self._motion_cost(profile, bbox, frame_id)
                if not height_ok or motion_error > motion_allowed:
                    self.counters["rejected_motion"] += 1
                    continue
                motion_score = min(1.0, motion_error / max(1e-6, motion_allowed))
                fusion_cost = self.appearance_weight * appearance + self.motion_weight * motion_score
                components[(row, col)] = (appearance, motion_score, fusion_cost)
                cost_matrix[row, col] = fusion_cost

        rows, cols = self._solve_assignment(cost_matrix)
        matched_rows = set()
        matched_cols = set()
        for row, col in zip(rows, cols):
            row, col = int(row), int(col)
            if cost_matrix[row, col] >= invalid_cost or cost_matrix[row, col] > self.max_fusion_cost:
                continue

            # Reject near-ties on either side. A new ID is safer than a wrong merge.
            row_alternatives = np.sort(cost_matrix[row][cost_matrix[row] < invalid_cost])
            col_alternatives = np.sort(cost_matrix[:, col][cost_matrix[:, col] < invalid_cost])
            row_clear = len(row_alternatives) < 2 or (row_alternatives[1] - row_alternatives[0]) >= self.ambiguity_margin
            col_clear = len(col_alternatives) < 2 or (col_alternatives[1] - col_alternatives[0]) >= self.ambiguity_margin
            if not row_clear or not col_clear:
                self.counters["rejected_ambiguous"] += 1
                continue

            det = unmatched_detections[row]
            profile = lost_profiles[col]
            track_id = int(det["track_id"])
            self._update_profile(profile, frame_id, det, embeddings.get(track_id))
            assigned[track_id] = profile.target_id
            used_targets.add(profile.target_id)
            self.associations[track_id] = "global_reid_fusion"
            self.counters["reidentified"] += 1
            matched_rows.add(row)
            matched_cols.add(col)

        # Unmatched or ambiguous tracks get fresh IDs; never force a dubious merge.
        for row, det in enumerate(unmatched_detections):
            if row in matched_rows:
                continue
            track_id = int(det["track_id"])
            target_id = self._new_target(frame_id, det, embeddings.get(track_id))
            assigned[track_id] = target_id
            if any(cost_matrix[row, c] < invalid_cost for c in range(cost_matrix.shape[1])):
                self.associations[track_id] = "new_after_reid_reject"

        return assigned
