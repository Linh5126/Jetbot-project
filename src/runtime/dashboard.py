"""Read-only OpenCV dashboard for the PC person-following demo.

The dashboard consumes existing inference/lock/controller outputs. It never
changes identities, thresholds, command authorization, or the motor driver.
Camera pixels start at (0, 0), preserving click-to-select coordinates.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import cv2
import numpy as np

from src.following.controller import MotionCommand
from src.following.target_lock import FollowState, LockResult
from src.runtime.overlay import plan_box_labels, _fit_text

# OpenCV uses BGR throughout.
SIDEBAR_WIDTH = 446
MIN_CANVAS_HEIGHT = 720
BG = (17, 21, 28)
PANEL = (23, 28, 36)
CARD = (33, 40, 50)
HAIR = (61, 70, 82)
TEXT = (241, 245, 248)
MUTED = (155, 166, 179)
GREEN = (121, 214, 128)
AMBER = (76, 187, 246)
RED = (97, 99, 242)
BLUE = (235, 184, 110)

_STATE_COLORS = {
    FollowState.TRACKING: GREEN,
    FollowState.REACQUIRED: GREEN,
    FollowState.LOST: AMBER,
    FollowState.VERIFYING: BLUE,
    FollowState.STOPPED: RED,
    FollowState.IDLE: MUTED,
}


@dataclass
class DashboardTelemetry:
    """Small bounded read-only history; never participates in decision-making."""
    history: deque[FollowState] = field(default_factory=lambda: deque(maxlen=180))
    events: deque[tuple[int, str]] = field(default_factory=lambda: deque(maxlen=5))
    durations: deque[float] = field(default_factory=lambda: deque(maxlen=40))
    previous_state: FollowState | None = None
    frame_id: int = 0
    people: int = 0

    def observe(self, *, frame_id: int, result: LockResult, people: int,
                elapsed_s: float) -> None:
        self.frame_id = frame_id
        self.people = people
        self.history.append(result.state)
        if 0 < elapsed_s < 300:
            self.durations.append(elapsed_s)
        if result.state != self.previous_state:
            self.events.appendleft((frame_id, result.state.value))
            self.previous_state = result.state

    def add_event(self, frame_id: int, label: str) -> None:
        self.events.appendleft((frame_id, label))

    @property
    def processing_fps(self) -> float | None:
        return len(self.durations) / sum(self.durations) if self.durations else None

    @property
    def hold_fraction(self) -> float | None:
        active = [state for state in self.history if state != FollowState.IDLE]
        return (sum(state not in (FollowState.TRACKING, FollowState.REACQUIRED)
                    for state in active) / len(active)) if active else None


def _rectangle(img, x1, y1, x2, y2, color, *, radius=0, thickness=-1):
    """Small rounded rectangle; no extra imaging packages needed."""
    if x2 <= x1 or y2 <= y1:
        return
    if radius <= 0 or thickness != -1:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        return
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r),
                   (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def _text(img, value, x, y, *, color=TEXT, scale=0.48, thick=1, max_width=400):
    value = _fit_text(str(value), max_width, scale, thick)
    cv2.putText(img, value, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def _section(panel, heading, y):
    _text(panel, heading.upper(), 20, y, color=MUTED, scale=0.41, thick=1)
    cv2.line(panel, (20, y + 10), (SIDEBAR_WIDTH - 20, y + 10), HAIR, 1)


def _pill(panel, text, x, y, color, width=128):
    _rectangle(panel, x, y, x + width, y + 28, CARD, radius=8)
    cv2.circle(panel, (x + 13, y + 14), 4, color, -1, cv2.LINE_AA)
    _text(panel, text, x + 25, y + 19, color=TEXT, scale=0.42, thick=1,
          max_width=width - 32)


def _meter(panel, x, y, width, ratio, fill):
    _rectangle(panel, x, y, x + width, y + 7, HAIR, radius=3)
    if ratio is not None:
        fill_w = int(round(width * max(0.0, min(1.0, float(ratio)))))
        if fill_w:
            _rectangle(panel, x, y, x + max(fill_w, 6), y + 7, fill, radius=3)


def _image_crop(panel: np.ndarray, crop: np.ndarray | None, x: int, y: int,
                width: int, height: int) -> None:
    _rectangle(panel, x, y, x + width, y + height, CARD, radius=7)
    if crop is None or crop.size == 0:
        _text(panel, "NO REF", x + 10, y + height // 2, color=MUTED,
              scale=0.36, max_width=width - 15)
        return
    ch, cw = crop.shape[:2]
    if not ch or not cw:
        return
    scale = min((width - 6) / cw, (height - 6) / ch)
    sw, sh = max(1, int(cw * scale)), max(1, int(ch * scale))
    small = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA)
    xx = x + (width - sw) // 2
    yy = y + (height - sh) // 2
    panel[yy:yy + sh, xx:xx + sw] = small


def _timeline(panel: np.ndarray, telemetry: DashboardTelemetry, y: int) -> None:
    _section(panel, "Identity timeline (last 180 frames)", y)
    x1, width = 20, SIDEBAR_WIDTH - 40
    _rectangle(panel, x1, y + 24, x1 + width, y + 42, CARD, radius=3)
    series = list(telemetry.history)
    if not series:
        _text(panel, "Waiting for frames", x1, y + 60, color=MUTED, scale=0.38)
        return
    for i, state in enumerate(series):
        left = x1 + int(i * width / len(series))
        right = x1 + max(1, int((i + 1) * width / len(series)))
        cv2.rectangle(panel, (left, y + 24), (right, y + 42),
                      _STATE_COLORS.get(state, MUTED), -1)
    _text(panel, "TRACKING", x1, y + 63, color=GREEN, scale=0.33)
    _text(panel, "VERIFY", x1 + 109, y + 63, color=BLUE, scale=0.33)
    _text(panel, "LOST", x1 + 210, y + 63, color=AMBER, scale=0.33)
    _text(panel, "STOP", x1 + 301, y + 63, color=RED, scale=0.33)


def _view_camera(canvas, frame, detections, result, *, show_debug=False):
    height, width = frame.shape[:2]
    camera = canvas[:height, :width]
    camera[:] = frame
    active = result.observation.track_id if result.may_follow else None
    for det in detections:
        x1, y1, x2, y2 = [int(round(float(v))) for v in det["bbox"]]
        chosen = active is not None and det["track_id"] == active
        cv2.rectangle(camera, (x1, y1), (x2, y2), GREEN if chosen else BLUE,
                      3 if chosen else 1)
        if show_debug and chosen:
            _text(camera, f"conf {det.get('confidence', 0):.2f}", max(x1, 0),
                  min(height - 8, y2 + 18), color=GREEN, scale=0.42)
    for entry in plan_box_labels(detections, result.target_id if result.may_follow else None,
                                 width, height, active_track_id=active):
        x1, y1, x2, y2 = entry.rect
        fill = GREEN if entry.target else BLUE
        cv2.rectangle(camera, (x1, y1), (x2, y2), fill, -1)
        _, baseline = cv2.getTextSize(entry.text, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.48 if entry.target else 0.41, 1)
        _text(camera, entry.text, x1 + 5, y2 - baseline - 3,
              color=BG, scale=0.48 if entry.target else 0.41, max_width=x2 - x1 - 8)


def _camera_footer(canvas, frame_height: int, frame_width: int, canvas_height: int,
                   result: LockResult, model_label: str, view_mode: str):
    if canvas_height - frame_height < 92:
        return
    y = frame_height + 14
    _rectangle(canvas, 0, frame_height, frame_width - 1, canvas_height - 1, PANEL)
    _text(canvas, "VISUAL PIPELINE", 16, y + 12, color=MUTED, scale=0.42)
    modules = ["YOLOv8n", "ByteTrack", "OSNet", "V5 Fusion", "Guard"]
    usable = frame_width - 32
    gap = 8
    unit = max(55, (usable - gap * 4) // 5)
    if unit * 5 + gap * 4 <= usable:
        for i, item in enumerate(modules):
            xx = 16 + i * (unit + gap)
            _rectangle(canvas, xx, y + 25, xx + unit, y + 62, CARD, radius=7)
            _text(canvas, item, xx + 7, y + 49, color=GREEN if i == 4 and result.may_follow
                  else TEXT, scale=0.39, max_width=unit - 12)
    _text(canvas, f"{view_mode.upper()}  |  {model_label}", 16, canvas_height - 16,
          color=MUTED, scale=0.4, max_width=usable)


def _header(panel, view_mode: str):
    _rectangle(panel, 0, 0, SIDEBAR_WIDTH - 1, 80, BG)
    _text(panel, "JETBOT / IDENTITY LAB", 20, 30, scale=0.66, thick=2)
    _text(panel, "CV  |  PC MOCK  |  NO REAL MOTORS", 20, 56, color=MUTED, scale=0.40)
    _pill(panel, view_mode.upper(), 292, 17, BLUE, width=137)


def _render_presentation(panel, result, command, diagnostics,
                         telemetry, reference_crop):
    color = _STATE_COLORS[result.state]
    _section(panel, "Selected target", 102)
    _image_crop(panel, reference_crop, 20, 125, 108, 132)
    target = f"T{result.target_id}" if result.target_id is not None else "NONE"
    _text(panel, target, 146, 164, scale=1.02, thick=2, max_width=270)
    _text(panel, "Frozen reference" if reference_crop is not None else "Click a person to lock",
          146, 190, color=MUTED, scale=0.41, max_width=260)
    _pill(panel, result.state.value, 146, 216, color, width=170)

    _section(panel, "Robot decision", 285)
    _rectangle(panel, 20, 300, SIDEBAR_WIDTH - 20, 386, CARD, radius=10)
    _text(panel, "FOLLOW AUTHORIZED" if result.may_follow else "HOLD / STOP",
          38, 337, color=GREEN if result.may_follow else AMBER, scale=0.7, thick=2,
          max_width=380)
    _text(panel, "Mock forward %+0.3f  |  turn %+0.3f" % (command.forward, command.turn),
          38, 368, color=MUTED, scale=0.47)

    _section(panel, "Why this decision", 416)
    _text(panel, getattr(diagnostics, "reason_code", result.state.value),
          20, 448, color=color, scale=0.60, thick=2)
    _text(panel, result.reason, 20, 478, color=TEXT, scale=0.43, max_width=407)
    _timeline(panel, telemetry, 523)
    if telemetry.events:
        _text(panel, "RECENT EVENTS", 20, 610, color=MUTED, scale=0.36)
        for i, (frame_no, event) in enumerate(list(telemetry.events)[:2]):
            _text(panel, f"#{frame_no:04d}  {event}", 20, 630 + i * 21,
                  color=TEXT, scale=0.42, max_width=405)
    _text(panel, "CLICK lock  |  A add view  |  R reset  |  SPACE stop  |  TAB research  |  Q exit",
          20, 671, color=MUTED, scale=0.34, max_width=420)


def _render_research(panel, result, command, diagnostics,
                     telemetry, guard_config):
    _section(panel, "Target / Gate / Inference", 101)
    tid = "NONE" if result.target_id is None else f"T{result.target_id}"
    _text(panel, f"{tid}   /   {result.state.value}", 20, 136,
          color=_STATE_COLORS[result.state], scale=0.65, thick=2)
    code = getattr(diagnostics, "reason_code", result.state.value)
    _text(panel, f"Reason: {code}", 20, 164, color=MUTED, scale=0.44)
    _text(panel, f"Motion: {'AUTHORIZED' if result.may_follow else 'HOLD'}",
          20, 188, color=GREEN if result.may_follow else AMBER, scale=0.47, thick=2)

    _section(panel, "Frozen-reference checks", 220)
    distance = getattr(diagnostics, "reference_distance", None)
    margin = getattr(diagnostics, "competitor_margin", None)
    overlap = getattr(diagnostics, "candidate_overlap", None)
    threshold = getattr(guard_config, "track_distance", None)
    recover_threshold = getattr(guard_config, "recovery_distance", None)
    overlap_limit = getattr(guard_config, "max_overlap", None)
    margin_limit = getattr(guard_config, "ambiguity_margin", None)
    metrics = (
        ("Cosine distance (lower better)", distance,
         f"track <= {threshold:.2f} / recover <= {recover_threshold:.2f}" if threshold is not None else "N/A",
         2.0, GREEN),
        ("Runner-up distance margin", margin,
         f"min >= {margin_limit:.2f}" if margin_limit is not None else "N/A",
         1.0, BLUE),
        ("Bounding-box overlap", overlap,
         f"max <= {overlap_limit:.2f}" if overlap_limit is not None else "N/A",
         1.0, AMBER),
    )
    for i, (label, value, hint, bound, color) in enumerate(metrics):
        yy = 249 + i * 74
        _text(panel, label, 20, yy, color=TEXT, scale=0.45)
        _text(panel, "N/A" if value is None else f"{value:.3f}",
              355, yy, color=color, scale=0.50, thick=2, max_width=70)
        _meter(panel, 20, yy + 12, SIDEBAR_WIDTH - 40,
               None if value is None else value / bound, color)
        _text(panel, hint, 20, yy + 33, color=MUTED, scale=0.38)

    hits = getattr(diagnostics, "candidate_hits", None)
    required = getattr(diagnostics, "candidate_required", None)
    track = getattr(diagnostics, "trusted_track_id", None)
    candidate = getattr(diagnostics, "candidate_track_id", None)
    _section(panel, "Recovery & command", 483)
    _text(panel, f"Trusted B: {track if track is not None else '-'}     "
          f"Candidate B: {candidate if candidate is not None else '-'}",
          20, 512, color=TEXT, scale=0.45)
    _text(panel, f"Evidence: {hits or 0}/{required or 0}     "
          f"Mock: F {command.forward:+.2f} / T {command.turn:+.2f}",
          20, 536, color=MUTED, scale=0.44)

    _timeline(panel, telemetry, 566)
    if telemetry.events:
        frame_no, event = telemetry.events[0]
        _text(panel, f"LAST: #{frame_no:04d} {event}", 20, 648,
              color=MUTED, scale=0.39, max_width=410)
    _text(panel, "TAB presentation  |  CLICK lock  |  A add view  |  R reset  |  SPACE stop  |  Q exit",
          20, 671, color=MUTED, scale=0.34, max_width=420)


def render_dashboard_frame(
    frame: np.ndarray,
    detections: Sequence[Mapping],
    result: LockResult,
    command: MotionCommand,
    model_label: str,
    *,
    diagnostics=None,
    telemetry: DashboardTelemetry | None = None,
    view_mode: str = "presentation",
    reference_crop: np.ndarray | None = None,
    guard_config=None,
) -> np.ndarray:
    """Return BGR canvas suitable for imshow / VideoWriter. Input is untouched."""
    if view_mode not in ("presentation", "research"):
        raise ValueError("view_mode must be 'presentation' or 'research'")
    h, w = frame.shape[:2]
    ch = max(MIN_CANVAS_HEIGHT, h)
    canvas = np.full((ch, w + SIDEBAR_WIDTH, 3), BG, dtype=np.uint8)
    _view_camera(canvas, frame, detections, result, show_debug=view_mode == "research")
    info = telemetry if telemetry is not None else DashboardTelemetry()
    _camera_footer(canvas, h, w, ch, result, model_label, view_mode)
    panel = canvas[:, w:]
    panel[:] = PANEL
    _header(panel, view_mode)
    if view_mode == "presentation":
        _render_presentation(panel, result, command, diagnostics, info, reference_crop)
    else:
        _render_research(panel, result, command, diagnostics, info, guard_config)
    # Runtime values, not model predictions; placement scales with frame size.
    _rectangle(panel, 0, ch - 41, SIDEBAR_WIDTH - 1, ch - 1, BG)
    fps = info.processing_fps
    _text(panel, f"FRAME {info.frame_id:05d}  |  {info.people} PEOPLE  |  "
          f"PROCESS {fps:.1f} FPS" if fps is not None else
          f"FRAME {info.frame_id:05d}  |  {info.people} PEOPLE  |  FPS N/A",
          20, ch - 15, color=MUTED, scale=0.43, max_width=420)
    return canvas


def fit_display_frame(frame: np.ndarray, max_width: int, max_height: int) -> tuple[np.ndarray, float]:
    """Downscale a rendered dashboard for a monitor without changing recorded pixels."""
    h, w = frame.shape[:2]
    scale = min(1.0, max_width / w, max_height / h)
    if scale >= 1:
        return frame, 1.0
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA), scale


def map_display_click(x: int, y: int, scale: float) -> tuple[int, int]:
    """Convert a click on the scaled window to original detector coordinates."""
    if not 0 < scale <= 1:
        raise ValueError("display scale must be in (0,1]")
    return int(x / scale), int(y / scale)
