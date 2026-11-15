from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from src.identity.manager_v5 import (
    IdentityManagerV5,
    TargetProfileV5,
    _cosine_distance,
    _normalise_embedding,
)


class IdentityManagerV51(IdentityManagerV5):
    """V5.1: V5 association plus a quality/diversity-aware Re-ID gallery.

    V5 remains unchanged. V5.1 only admits embeddings with sufficient crop quality,
    rejects appearance outliers, replaces near-duplicate low-quality samples with
    better samples, and evicts the lowest-quality gallery item when capacity is hit.
    """

    def __init__(
        self,
        *args,
        min_crop_quality: float = 0.42,
        gallery_diversity_distance: float = 0.035,
        gallery_replace_quality_margin: float = 0.06,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_crop_quality = float(np.clip(min_crop_quality, 0.0, 1.0))
        self.gallery_diversity_distance = float(np.clip(gallery_diversity_distance, 0.0, 2.0))
        self.gallery_replace_quality_margin = float(np.clip(gallery_replace_quality_margin, 0.0, 1.0))
        self._gallery_quality: dict[int, list[float]] = {}
        self._current_crop_quality: Optional[float] = None
        self.counters.update({
            "quality_samples_seen": 0,
            "quality_samples_accepted": 0,
            "quality_samples_rejected": 0,
            "quality_metadata_missing": 0,
            "gallery_outlier_rejected": 0,
            "gallery_redundant_rejected": 0,
            "gallery_quality_replacements": 0,
            "gallery_quality_evictions": 0,
        })

    def _quality_score(self, det: dict) -> Optional[float]:
        value = det.get("crop_quality")
        if value is None:
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(score):
            return None
        return float(np.clip(score, 0.0, 1.0))

    def _quality_ok(self, det: dict) -> bool:
        if not super()._quality_ok(det):
            return False
        score = self._quality_score(det)
        return score is not None and score >= self.min_crop_quality

    def update(self, frame_id: int, detections: list[dict], embeddings=None) -> dict[int, int]:
        embeddings = embeddings or {}
        for det in detections:
            track_id = int(det["track_id"])
            if track_id not in embeddings:
                continue
            self.counters["quality_samples_seen"] += 1
            score = self._quality_score(det)
            if score is None:
                self.counters["quality_metadata_missing"] += 1
                self.counters["quality_samples_rejected"] += 1
            elif score >= self.min_crop_quality:
                self.counters["quality_samples_accepted"] += 1
            else:
                self.counters["quality_samples_rejected"] += 1
        return super().update(frame_id, detections, embeddings)

    def _add_gallery_sample(self, profile: TargetProfileV5, embedding) -> None:
        vector = _normalise_embedding(embedding)
        if vector is None:
            return

        # The embedding is only admitted if the upstream crop-quality score is
        # present and above threshold. Missing metadata is fail-closed.
        score = getattr(profile, "_pending_crop_quality", None)
        # _add_gallery_sample is called from _new_target/_update_profile; the
        # current detection's quality is placed immediately before the call below.
        if score is None:
            score = self._current_crop_quality
        if score is None or float(score) < self.min_crop_quality:
            return
        score = float(score)

        qualities = self._gallery_quality.setdefault(int(profile.target_id), [])
        samples = list(profile.gallery)
        if len(qualities) != len(samples):
            # Defensive recovery for profiles created by external callers.
            qualities[:] = [0.5] * len(samples)

        if samples:
            distances = [_cosine_distance(vector, old) for old in samples]
            nearest_idx = int(np.argmin(distances))
            nearest_distance = float(distances[nearest_idx])
            if nearest_distance > self.gallery_update_gate:
                self.counters["gallery_outlier_rejected"] += 1
                return
            if nearest_distance <= self.gallery_diversity_distance:
                if score >= qualities[nearest_idx] + self.gallery_replace_quality_margin:
                    samples[nearest_idx] = vector.copy()
                    qualities[nearest_idx] = score
                    profile.gallery = deque(samples)
                    self.counters["gallery_quality_replacements"] += 1
                else:
                    self.counters["gallery_redundant_rejected"] += 1
                return

        samples.append(vector.copy())
        qualities.append(score)
        if len(samples) > self.gallery_size:
            worst_idx = int(np.argmin(qualities))
            samples.pop(worst_idx)
            qualities.pop(worst_idx)
            self.counters["gallery_quality_evictions"] += 1
        profile.gallery = deque(samples)

    def _new_target(self, frame_id: int, det: dict, embedding) -> int:
        self._current_crop_quality = self._quality_score(det)
        try:
            return super()._new_target(frame_id, det, embedding)
        finally:
            self._current_crop_quality = None

    def _update_profile(self, profile: TargetProfileV5, frame_id: int, det: dict, embedding) -> None:
        self._current_crop_quality = self._quality_score(det)
        try:
            super()._update_profile(profile, frame_id, det, embedding)
        finally:
            self._current_crop_quality = None

    def diagnostics(self) -> dict:
        result = super().diagnostics()
        all_scores = [score for values in self._gallery_quality.values() for score in values]
        result["gallery_mean_quality"] = float(np.mean(all_scores)) if all_scores else None
        result["gallery_min_quality"] = float(np.min(all_scores)) if all_scores else None
        result["quality_policy"] = {
            "min_crop_quality": self.min_crop_quality,
            "gallery_diversity_distance": self.gallery_diversity_distance,
            "gallery_replace_quality_margin": self.gallery_replace_quality_margin,
        }
        return result
