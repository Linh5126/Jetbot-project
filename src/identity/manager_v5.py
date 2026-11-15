from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
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
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return None
    return vector / norm


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(1.0 - np.dot(a, b), 0.0, 2.0))


@dataclass
class TargetProfileV5:
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
    motion_uncertainty: float = 0.15


class IdentityManagerV5:
    """V5 long-term identity manager with reproducible association ablations.

    Strategies:
      - appearance_only: cosine Re-ID distance + one-to-one assignment.
      - motion_only: normalised constant-velocity prediction + one-to-one assignment.
      - fixed_fusion: hard appearance/motion gates and fixed weighted cost.
      - adaptive_fusion: gap/uncertainty-aware appearance-motion weighting; a
        very strong appearance match may override a weak motion prediction.

    All strategies share the same persistent-ID bookkeeping and Hungarian
    assignment. They are intended for controlled ablation, not separate trackers.
    """

    STRATEGIES = {
        "appearance_only", "motion_only", "fixed_fusion", "adaptive_fusion"
    }

    def __init__(
        self,
        fps: float = 30.0,
        strategy: str = "adaptive_fusion",
        max_lost_seconds: float = 20.0,
        max_height_ratio: float = 2.0,
        appearance_gate: float = 0.48,
        active_appearance_gate: float = 0.72,
        appearance_weight: float = 0.72,
        motion_weight: float = 0.28,
        motion_gate_base: float = 0.75,
        motion_gate_per_second: float = 0.14,
        max_motion_gate: float = 3.2,
        max_fusion_cost: float = 0.68,
        ambiguity_margin: float = 0.035,
        gallery_size: int = 12,
        gallery_update_gate: float = 0.55,
        min_embedding_confidence: float = 0.35,
        min_box_height: float = 28.0,
        strong_appearance_distance: float = 0.18,
        adaptive_gap_seconds: float = 3.0,
        uncertainty_weight: float = 0.15,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown V5 strategy: {strategy}. Expected one of {sorted(self.STRATEGIES)}")
        self.fps = max(1.0, float(fps))
        self.strategy = strategy
        self.max_lost_frames = max(1, int(float(max_lost_seconds) * self.fps))
        self.max_height_ratio = max(1.01, float(max_height_ratio))
        self.appearance_gate = max(1e-6, float(appearance_gate))
        self.active_appearance_gate = max(self.appearance_gate, float(active_appearance_gate))
        total_weight = max(1e-8, float(appearance_weight) + float(motion_weight))
        self.appearance_weight = float(appearance_weight) / total_weight
        self.motion_weight = float(motion_weight) / total_weight
        self.motion_gate_base = max(0.0, float(motion_gate_base))
        self.motion_gate_per_second = max(0.0, float(motion_gate_per_second))
        self.max_motion_gate = max(self.motion_gate_base, float(max_motion_gate))
        self.max_fusion_cost = max(0.0, float(max_fusion_cost))
        self.ambiguity_margin = max(0.0, float(ambiguity_margin))
        self.gallery_size = max(1, int(gallery_size))
        self.gallery_update_gate = max(0.0, float(gallery_update_gate))
        self.min_embedding_confidence = float(min_embedding_confidence)
        self.min_box_height = float(min_box_height)
        self.strong_appearance_distance = max(0.0, float(strong_appearance_distance))
        self.adaptive_gap_seconds = max(0.1, float(adaptive_gap_seconds))
        self.uncertainty_weight = max(0.0, float(uncertainty_weight))

        self.next_target_id = 1
        self.targets: dict[int, TargetProfileV5] = {}
        self.track_to_target: dict[int, int] = {}
        self.associations: dict[int, str] = {}
        self.counters = {
            "new_targets": 0,
            "reidentified": 0,
            "active_conflicts": 0,
            "rejected_appearance": 0,
            "rejected_motion": 0,
            "rejected_ambiguous": 0,
            "rejected_quality": 0,
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
        gallery_items = sum(len(p.gallery) for p in self.targets.values())
        return {
            "strategy": self.strategy,
            **self.counters,
            "profiles_total": len(self.targets),
            "profiles_active": active,
            "profiles_lost": lost,
            "profiles_expired": expired,
            "gallery_items": gallery_items,
        }

    def _quality_ok(self, det: dict) -> bool:
        return (
            float(det.get("confidence", 1.0)) >= self.min_embedding_confidence
            and _box_height(np.asarray(det["bbox"], dtype=np.float32)) >= self.min_box_height
        )

    def _add_gallery_sample(self, profile: TargetProfileV5, embedding) -> None:
        vector = _normalise_embedding(embedding)
        if vector is None:
            return
        if profile.gallery:
            dists = [_cosine_distance(vector, old) for old in profile.gallery]
            # Prevent low-quality crops and identity switches from poisoning the gallery.
            if min(dists) > self.gallery_update_gate:
                return
        profile.gallery.append(vector.copy())
        while len(profile.gallery) > self.gallery_size:
            profile.gallery.popleft()

    def _gallery_distance(self, profile: TargetProfileV5, embedding) -> Optional[float]:
        vector = _normalise_embedding(embedding)
        if vector is None or not profile.gallery:
            return None
        distances = sorted(_cosine_distance(vector, old) for old in profile.gallery)
        k = min(3, len(distances))
        return float(np.mean(distances[:k]))

    def _new_target(self, frame_id: int, det: dict, embedding) -> int:
        target_id = self.next_target_id
        self.next_target_id += 1
        bbox = np.asarray(det["bbox"], dtype=np.float32)
        track_id = int(det["track_id"])
        profile = TargetProfileV5(
            target_id=target_id,
            bbox=bbox,
            center=_box_center(bbox),
            last_frame=int(frame_id),
            current_track_id=track_id,
            last_confidence=float(det.get("confidence", 1.0)),
        )
        if self.strategy != "motion_only" and self._quality_ok(det):
            self._add_gallery_sample(profile, embedding)
        self.targets[target_id] = profile
        self.track_to_target[track_id] = target_id
        self.associations[track_id] = "new"
        self.counters["new_targets"] += 1
        return target_id

    def _bind_track(self, profile: TargetProfileV5, track_id: int) -> None:
        for old_track, old_target in list(self.track_to_target.items()):
            if old_target == profile.target_id:
                del self.track_to_target[old_track]
        profile.current_track_id = int(track_id)
        self.track_to_target[int(track_id)] = profile.target_id

    def _update_profile(self, profile: TargetProfileV5, frame_id: int, det: dict, embedding) -> None:
        bbox = np.asarray(det["bbox"], dtype=np.float32)
        center = _box_center(bbox)
        dt = max(1, int(frame_id) - int(profile.last_frame))
        measured_velocity = (center - profile.center) / float(dt)
        scale = max(24.0, 0.5 * (_box_height(profile.bbox) + _box_height(bbox)))
        innovation = float(np.linalg.norm(measured_velocity - profile.velocity) * dt / scale)
        profile.motion_uncertainty = float(
            np.clip(0.80 * profile.motion_uncertainty + 0.20 * innovation, 0.05, 2.0)
        )
        profile.velocity = 0.70 * profile.velocity + 0.30 * measured_velocity
        profile.bbox = bbox
        profile.center = center
        profile.last_frame = int(frame_id)
        profile.status = "Active"
        profile.hits += 1
        profile.last_confidence = float(det.get("confidence", 1.0))
        self._bind_track(profile, int(det["track_id"]))
        if self.strategy != "motion_only" and self._quality_ok(det):
            self._add_gallery_sample(profile, embedding)

    def _motion_components(self, profile: TargetProfileV5, bbox: np.ndarray, frame_id: int):
        gap_frames = max(1, int(frame_id) - int(profile.last_frame))
        gap_seconds = gap_frames / self.fps
        predicted = profile.center + profile.velocity * gap_frames
        observed = _box_center(bbox)
        scale = max(24.0, 0.5 * (_box_height(profile.bbox) + _box_height(bbox)))
        error = float(np.linalg.norm(observed - predicted) / scale)
        uncertainty_bonus = self.uncertainty_weight * profile.motion_uncertainty
        allowed = min(
            self.max_motion_gate,
            self.motion_gate_base + self.motion_gate_per_second * gap_seconds + uncertainty_bonus,
        )
        ratio = _box_height(bbox) / _box_height(profile.bbox)
        height_ok = (1.0 / self.max_height_ratio) <= ratio <= self.max_height_ratio
        score = min(1.5, error / max(1e-6, allowed))
        return error, allowed, score, height_ok, gap_seconds

    def _pair_cost(self, profile: TargetProfileV5, det: dict, embedding, frame_id: int):
        bbox = np.asarray(det["bbox"], dtype=np.float32)
        quality_ok = _box_height(bbox) >= self.min_box_height
        if self.strategy != "motion_only":
            quality_ok = quality_ok and float(det.get("confidence", 1.0)) >= self.min_embedding_confidence
        if not quality_ok:
            self.counters["rejected_quality"] += 1
            return None

        app = self._gallery_distance(profile, embedding)
        motion_error, motion_allowed, motion_score, height_ok, gap_seconds = self._motion_components(
            profile, bbox, frame_id
        )

        if self.strategy == "appearance_only":
            if app is None or app > self.appearance_gate:
                self.counters["rejected_appearance"] += 1
                return None
            return app / self.appearance_gate

        if self.strategy == "motion_only":
            if not height_ok or motion_error > motion_allowed:
                self.counters["rejected_motion"] += 1
                return None
            return min(1.0, motion_score)

        if app is None or app > self.appearance_gate:
            self.counters["rejected_appearance"] += 1
            return None
        if not height_ok:
            self.counters["rejected_motion"] += 1
            return None

        if self.strategy == "fixed_fusion":
            if motion_error > motion_allowed:
                self.counters["rejected_motion"] += 1
                return None
            return (
                self.appearance_weight * (app / self.appearance_gate)
                + self.motion_weight * min(1.0, motion_score)
            )

        # Adaptive fusion: when the gap or motion uncertainty grows, appearance
        # receives more weight. A very strong appearance match can survive an
        # imperfect constant-velocity prediction, but still pays a motion cost.
        strong_match = app <= self.strong_appearance_distance
        if motion_error > motion_allowed and not strong_match:
            self.counters["rejected_motion"] += 1
            return None
        gap_factor = gap_seconds / (gap_seconds + self.adaptive_gap_seconds)
        uncertainty_factor = min(1.0, profile.motion_uncertainty / 1.0)
        app_weight = float(np.clip(
            0.65 + 0.20 * gap_factor + 0.08 * uncertainty_factor,
            0.60, 0.92
        ))
        motion_weight = 1.0 - app_weight
        cost = app_weight * (app / self.appearance_gate) + motion_weight * min(1.5, motion_score)
        if strong_match and motion_error > motion_allowed:
            # Do not make extreme, non-physical jumps free; retain a bounded penalty.
            cost += 0.10 * min(1.0, motion_error / max(motion_allowed, 1e-6) - 1.0)
        return float(cost)

    @staticmethod
    def _solve_assignment(cost_matrix: np.ndarray):
        if cost_matrix.size == 0:
            return np.asarray([], dtype=int), np.asarray([], dtype=int)
        if linear_sum_assignment is not None:
            return linear_sum_assignment(cost_matrix)
        pairs, used_rows, used_cols = [], set(), set()
        candidates = sorted(
            (float(cost_matrix[r, c]), r, c)
            for r in range(cost_matrix.shape[0])
            for c in range(cost_matrix.shape[1])
        )
        for _, row, col in candidates:
            if row not in used_rows and col not in used_cols:
                pairs.append((row, col))
                used_rows.add(row)
                used_cols.add(col)
        if not pairs:
            return np.asarray([], dtype=int), np.asarray([], dtype=int)
        rows, cols = zip(*pairs)
        return np.asarray(rows), np.asarray(cols)

    def update(self, frame_id: int, detections: list[dict], embeddings=None) -> dict[int, int]:
        embeddings = embeddings or {}
        frame_id = int(frame_id)
        detections = list(detections)
        present_track_ids = {int(d["track_id"]) for d in detections}

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
        unmatched: list[dict] = []

        # Pass A: keep a current tracker identity unless strong Re-ID evidence
        # indicates that the tracker has attached that ID to a different person.
        for det in detections:
            track_id = int(det["track_id"])
            target_id = self.track_to_target.get(track_id)
            profile = self.targets.get(target_id) if target_id is not None else None
            if profile is None or profile.status == "Expired" or target_id in used_targets:
                if target_id is not None and target_id in used_targets:
                    self.track_to_target.pop(track_id, None)
                unmatched.append(det)
                continue

            app = self._gallery_distance(profile, embeddings.get(track_id))
            quality_ok = self._quality_ok(det)
            if (
                self.strategy != "motion_only"
                and quality_ok
                and app is not None
                and app > self.active_appearance_gate
            ):
                self.track_to_target.pop(track_id, None)
                profile.status = "Lost"
                profile.current_track_id = None
                unmatched.append(det)
                self.counters["active_conflicts"] += 1
                continue

            self._update_profile(profile, frame_id, det, embeddings.get(track_id))
            assigned[track_id] = profile.target_id
            used_targets.add(profile.target_id)
            self.associations[track_id] = "existing"

        if not unmatched:
            return assigned

        lost_profiles = [
            p for p in self.targets.values()
            if p.status == "Lost"
            and p.target_id not in used_targets
            and (frame_id - p.last_frame) <= self.max_lost_frames
            and (self.strategy == "motion_only" or len(p.gallery) > 0)
        ]
        if not lost_profiles:
            for det in unmatched:
                tid = int(det["track_id"])
                assigned[tid] = self._new_target(frame_id, det, embeddings.get(tid))
            return assigned

        invalid = 1e6
        costs = np.full((len(unmatched), len(lost_profiles)), invalid, dtype=np.float64)
        for row, det in enumerate(unmatched):
            tid = int(det["track_id"])
            for col, profile in enumerate(lost_profiles):
                cost = self._pair_cost(profile, det, embeddings.get(tid), frame_id)
                max_allowed_cost = (
                    1.0 if self.strategy in {"appearance_only", "motion_only"}
                    else self.max_fusion_cost
                )
                if cost is not None and cost <= max_allowed_cost:
                    costs[row, col] = cost

        rows, cols = self._solve_assignment(costs)
        matched_rows: set[int] = set()
        matched_cols: set[int] = set()
        for row, col in zip(rows, cols):
            row, col = int(row), int(col)
            cost = float(costs[row, col])
            max_allowed_cost = (
                1.0 if self.strategy in {"appearance_only", "motion_only"}
                else self.max_fusion_cost
            )
            if cost >= invalid or cost > max_allowed_cost:
                continue
            row_alts = np.sort(costs[row][costs[row] < invalid])
            col_alts = np.sort(costs[:, col][costs[:, col] < invalid])
            row_clear = len(row_alts) < 2 or (row_alts[1] - row_alts[0]) >= self.ambiguity_margin
            col_clear = len(col_alts) < 2 or (col_alts[1] - col_alts[0]) >= self.ambiguity_margin
            if not row_clear or not col_clear:
                self.counters["rejected_ambiguous"] += 1
                continue

            det = unmatched[row]
            profile = lost_profiles[col]
            tid = int(det["track_id"])
            self._update_profile(profile, frame_id, det, embeddings.get(tid))
            assigned[tid] = profile.target_id
            used_targets.add(profile.target_id)
            self.associations[tid] = f"reid_{self.strategy}"
            self.counters["reidentified"] += 1
            matched_rows.add(row)
            matched_cols.add(col)

        for row, det in enumerate(unmatched):
            if row in matched_rows:
                continue
            tid = int(det["track_id"])
            assigned[tid] = self._new_target(frame_id, det, embeddings.get(tid))
            if np.any(costs[row] < invalid):
                self.associations[tid] = "new_after_reid_reject"

        return assigned
