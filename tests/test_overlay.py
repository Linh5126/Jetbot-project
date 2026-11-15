"""UI regression tests; no camera, OSNet, GPU or robot required."""
from __future__ import annotations

import cv2
import numpy as np

from demo_pc import draw_frame, select_at_point
from src.following.controller import MotionCommand
from src.following.target_lock import FollowState, LockResult, TargetObservation
from src.runtime.overlay import MIN_CANVAS_HEIGHT, SIDEBAR_WIDTH, _intersects, plan_box_labels


def _det(target_id: int, x: int, y: int) -> dict:
    return {"target_id": target_id, "track_id": 100 + target_id,
            "bbox": (x, y, x + 95, y + 145), "confidence": 0.95}


def test_dense_scene_labels_do_not_overlap_or_leave_frame():
    detections = [_det(i, 50 + (i % 5) * 28, 80 + (i // 5) * 26) for i in range(14)]
    labels = plan_box_labels(detections, target_id=12, image_width=640, image_height=480)
    assert any(label.target and "T12" in label.text for label in labels)
    assert len(labels) <= len(detections)
    for idx, label in enumerate(labels):
        left, top, right, bottom = label.rect
        assert 0 <= left < right <= 640
        assert 0 <= top < bottom <= 480
        assert all(not _intersects(label.rect, other.rect) for other in labels[idx + 1:])


def test_diagnostics_render_outside_camera_and_preserve_click_coordinates():
    blank = np.full((480, 640, 3), 45, dtype=np.uint8)
    detections = [_det(7, 200, 150)]
    obs = TargetObservation.from_detection(detections[0])
    locked = LockResult(FollowState.TRACKING, 7, obs, "Persistent target confirmed")
    rendered = draw_frame(blank, detections, locked, MotionCommand(0.12, -0.04), "v5")
    assert rendered.shape == (480, 640 + SIDEBAR_WIDTH, 3)
    assert np.array_equal(blank, np.full((480, 640, 3), 45, dtype=np.uint8))
    assert select_at_point(detections, 220, 180)["target_id"] == 7
    assert select_at_point(detections, 700, 180) is None
    # Camera content remains identical away from bounding boxes/tags.
    assert np.array_equal(rendered[300:450, 400:600], blank[300:450, 400:600])
    assert np.any(rendered[:, 640:] != 45)  # Separate HUD is present.


def test_video_writer_frame_size_matches_rendered_size(tmp_path):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    idle = LockResult(FollowState.IDLE, None, None, "No target selected")
    rendered = draw_frame(frame, [], idle, MotionCommand(), "baseline")
    assert rendered.shape == (MIN_CANVAS_HEIGHT, 320 + SIDEBAR_WIDTH, 3)
    out_file = tmp_path / "overlay_test.mp4"
    writer = cv2.VideoWriter(str(out_file), cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                             (rendered.shape[1], rendered.shape[0]))
    if writer.isOpened():
        writer.write(rendered)
        writer.release()
        clip = cv2.VideoCapture(str(out_file))
        ok, decoded = clip.read()
        clip.release()
        assert ok and decoded.shape[:2] == rendered.shape[:2]
    else:
        # Local OpenCV codecs can vary; geometry assertion above is mandatory.
        writer.release()
