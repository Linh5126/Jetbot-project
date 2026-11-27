
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np


def score_person_crop_quality(
    crop: np.ndarray,
    bbox: Union[List[float], Tuple[float, ...]],
    frame_shape: Tuple[int, ...],
    detection_confidence: float = 1.0,
) -> Dict[str, float]:
    """Return an inexpensive, deterministic quality estimate for a person crop.

    The score is a heuristic for gallery admission, not a calibrated probability.
    Components cover blur, exposure, contrast, available pixel resolution, detector
    confidence, and clipping by the image border.
    """
    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
        return {
            "score": 0.0, "blur": 0.0, "brightness": 0.0,
            "contrast": 0.0, "resolution": 0.0, "confidence": 0.0,
            "border": 0.0,
        }

    height, width = crop.shape[:2]
    if height < 2 or width < 2:
        return {
            "score": 0.0, "blur": 0.0, "brightness": 0.0,
            "contrast": 0.0, "resolution": 0.0, "confidence": 0.0,
            "border": 0.0,
        }

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = np.asarray(gray, dtype=np.uint8)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_intensity = float(gray.mean())
    contrast_std = float(gray.std())

    blur_score = float(np.clip((laplacian_variance - 8.0) / 92.0, 0.0, 1.0))
    brightness_score = float(np.clip(1.0 - abs(mean_intensity - 127.5) / 115.0, 0.0, 1.0))
    contrast_score = float(np.clip(contrast_std / 55.0, 0.0, 1.0))
    resolution_score = float(np.sqrt(
        min(1.0, height / 96.0) * min(1.0, width / 32.0)
    ))
    confidence_score = float(np.clip(
        (float(detection_confidence) - 0.20) / 0.75, 0.0, 1.0
    ))

    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    x1, y1, x2, y2 = [float(v) for v in bbox]
    # A crop touching the image edge may contain a truncated person. A small
    # margin is tolerated; the score degrades smoothly rather than hard rejecting.
    margin = max(2.0, 0.01 * min(frame_h, frame_w))
    touches = sum((
        x1 <= margin, y1 <= margin,
        x2 >= frame_w - margin, y2 >= frame_h - margin,
    ))
    border_score = float(np.clip(1.0 - 0.18 * touches, 0.28, 1.0))

    score = (
        0.32 * blur_score
        + 0.18 * brightness_score
        + 0.18 * contrast_score
        + 0.14 * resolution_score
        + 0.10 * confidence_score
        + 0.08 * border_score
    )
    return {
        "score": float(np.clip(score, 0.0, 1.0)),
        "blur": blur_score,
        "brightness": brightness_score,
        "contrast": contrast_score,
        "resolution": resolution_score,
        "confidence": confidence_score,
        "border": border_score,
    }
