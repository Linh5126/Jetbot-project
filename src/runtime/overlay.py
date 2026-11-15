"""Compact, collision-aware OpenCV overlay for the PC person-following demo.

All diagnostic text lives in a separate right sidebar, never over the camera
image. Short per-person labels are placed without overlapping each other;
non-target labels may be hidden in dense scenes to keep the view readable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np

from src.following.controller import MotionCommand
from src.following.target_lock import FollowState, LockResult


SIDEBAR_WIDTH = 360
MIN_CANVAS_HEIGHT = 480

_BACKGROUND = (27, 31, 37)
_PANEL = (38, 43, 50)
_MUTED = (167, 177, 188)
_WHITE = (244, 245, 247)
_TARGET = (78, 207, 109)
_PERSON = (215, 164, 85)
_WARNING = (56, 190, 248)
_DANGER = (90, 90, 228)


@dataclass(frozen=True)
class BoxLabel:
    text: str
    rect: tuple[int, int, int, int]  # left, top, right, bottom
    target: bool


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Adjacent edges are allowed; only positive-area overlap is rejected."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def plan_box_labels(
    detections: Sequence[Mapping], target_id: int | None, image_width: int, image_height: int,
    active_track_id: int | None = None,
) -> list[BoxLabel]:
    """Plan non-overlapping labels; prioritize the selected target.

    Boxes remain visible even when crowded labels must be omitted. Positions
    stay inside the camera frame so sidebar text is never covered.
    """
    ordered = sorted(detections, key=lambda d: int(
        d["track_id"] == active_track_id if active_track_id is not None else d["target_id"] == target_id
    ), reverse=True)
    placed: list[BoxLabel] = []
    for det in ordered:
        x1, y1, x2, y2 = (int(round(float(v))) for v in det["bbox"])
        chosen = (det["track_id"] == active_track_id if active_track_id is not None
                  else target_id is not None and det["target_id"] == target_id)
        label = f"TARGET T{target_id} B{det['track_id']}" if chosen else f"T{det['target_id']}"
        scale = 0.48 if chosen else 0.41
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        tag_w = text_size[0] + 10
        tag_h = text_size[1] + baseline + 7
        if tag_w >= image_width or tag_h >= image_height:
            continue
        # Try immediately above, inside top, below, then right-aligned.
        left = min(max(0, x1), image_width - tag_w)
        right_aligned = min(max(0, x2 - tag_w), image_width - tag_w)
        positions = (
            (left, y1 - tag_h - 3),
            (left, y1 + 3),
            (left, y2 + 3),
            (right_aligned, y1 - tag_h - 3),
            (right_aligned, y1 + 3),
        )
        for px, py in positions:
            rect = (px, py, px + tag_w, py + tag_h)
            if py < 0 or py + tag_h > image_height:
                continue
            if any(_intersects(rect, earlier.rect) for earlier in placed):
                continue
            placed.append(BoxLabel(label, rect, chosen))
            break
    return placed


def _fit_text(text: str, max_width: int, scale: float, thickness: int = 1) -> str:
    """Trim text to measured pixel width, not by character count."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    if cv2.getTextSize(text, font, scale, thickness)[0][0] <= max_width:
        return text
    suffix = "..."
    while text and cv2.getTextSize(text + suffix, font, scale, thickness)[0][0] > max_width:
        text = text[:-1]
    return text + suffix if text else suffix


def _write(
    image: np.ndarray, text: str, x: int, y: int, *, color: tuple[int, int, int] = _WHITE,
    scale: float = 0.48, thickness: int = 1, max_width: int = SIDEBAR_WIDTH - 34,
) -> None:
    text = _fit_text(str(text), max_width, scale, thickness)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def render_demo_frame(
    frame: np.ndarray,
    detections: Sequence[Mapping],
    result: LockResult,
    command: MotionCommand,
    mode: str,
    diagnostics=None,
) -> np.ndarray:
    """Render a camera frame with a non-overlapping diagnostic sidebar.

    The camera image remains at (0, 0), so clicking a bbox continues to use
    the original detector coordinates. The output width grows by 360 pixels;
    callers must initialize VideoWriter with the rendered frame size.
    """
    height, width = frame.shape[:2]
    canvas_height = max(height, MIN_CANVAS_HEIGHT)
    canvas = np.full((canvas_height, width + SIDEBAR_WIDTH, 3), _BACKGROUND, dtype=np.uint8)
    camera = canvas[:height, :width]
    camera[:] = frame
    panel = canvas[:, width:]
    panel[:] = _PANEL

    active_track_id = result.observation.track_id if result.may_follow else None
    for det in detections:
        x1, y1, x2, y2 = (int(round(float(v))) for v in det["bbox"])
        chosen = active_track_id is not None and det["track_id"] == active_track_id
        cv2.rectangle(camera, (x1, y1), (x2, y2), _TARGET if chosen else _PERSON,
                      3 if chosen else 1)

    for item in plan_box_labels(detections, result.target_id if result.may_follow else None,
                                width, height, active_track_id=active_track_id):
        x1, y1, x2, y2 = item.rect
        color = _TARGET if item.target else _PERSON
        cv2.rectangle(camera, (x1, y1), (x2, y2), color, -1)
        scale = 0.48 if item.target else 0.41
        _, baseline = cv2.getTextSize(item.text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        _write(camera, item.text, x1 + 5, y2 - baseline - 3,
               color=_BACKGROUND, scale=scale, max_width=x2 - x1 - 9)

    x = 17
    cv2.rectangle(panel, (0, 0), (3, canvas_height), _TARGET, -1)
    _write(panel, "JETBOT | PC DEMO", x, 34, scale=0.64, thickness=2)
    _write(panel, f"{mode.upper()}  |  MOCK DRIVER", x, 59, color=_MUTED, scale=0.42)
    for y in (76, 216, 310):
        cv2.line(panel, (x, y), (SIDEBAR_WIDTH - x, y), (78, 84, 93), 1)

    _write(panel, "LOCKED TARGET", x, 105, color=_MUTED, scale=0.43)
    if diagnostics is not None:
        identity_text = "IDENTITY SAVED" if diagnostics.identity_locked else "NO IDENTITY"
        _write(panel, identity_text, 162, 105, color=_TARGET if diagnostics.identity_locked
               else _MUTED, scale=0.35, max_width=182)
    target_text = "None" if result.target_id is None else f"T{result.target_id}"
    _write(panel, target_text, x, 139, scale=0.8, thickness=2)
    _write(panel, "STATE", x, 169, color=_MUTED, scale=0.43)
    state_color = {
        FollowState.TRACKING: _TARGET, FollowState.REACQUIRED: _TARGET,
        FollowState.LOST: _WARNING, FollowState.VERIFYING: _WARNING,
        FollowState.STOPPED: _DANGER,
    }.get(result.state, _MUTED)
    _write(panel, result.state.value, x, 198, color=state_color, scale=0.68, thickness=2)
    _write(panel, "MOVE" if result.may_follow else "HOLD", 258, 198,
           color=_TARGET if result.may_follow else _WARNING, scale=0.55, thickness=2,
           max_width=SIDEBAR_WIDTH - 275)

    _write(panel, "ROBOT COMMAND (MOCK)", x, 242, color=_MUTED, scale=0.43)
    _write(panel, f"Forward : {command.forward:+.3f}", x, 270, scale=0.51)
    _write(panel, f"Turn      : {command.turn:+.3f}", x, 295, scale=0.51)

    _write(panel, "CONTROLS", x, 337, color=_MUTED, scale=0.43)
    _write(panel, "CLICK  Select person", x, 361, scale=0.46)
    _write(panel, "R  Reset    SPACE  Stop", x, 383, scale=0.46)
    _write(panel, "Q / ESC  Exit", x, 405, scale=0.46)
    _write(panel, "DETAIL", x, 431, color=_MUTED, scale=0.43)
    _write(panel, result.reason, x, 456, color=_WHITE, scale=0.42)
    if canvas_height >= 505:
        _write(panel, f"Visible people: {len(detections)}", x, 489, color=_MUTED, scale=0.41)
    return canvas
