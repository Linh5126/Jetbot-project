"""Frame-wise PC follow trace and *optional* MOT17 GT interpretation.

GT is used only for reporting after manual target selection, never to select
or influence an inference/robot command. This is a target-following diagnostic,
not an official MOTChallenge evaluation or a physical-robot safety score.
"""
from __future__ import annotations

import configparser
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Mapping, Sequence

from src.data.mot17 import read_mot_gt
from src.following.controller import MotionCommand
from src.following.target_lock import LockResult

TRACE_COLUMNS = (
    "frame", "timestamp_s", "session", "gt_target_id", "event",
    "target_id", "state", "reason_code", "reason", "identity_locked",
    "motion_authorized", "trusted_track_id", "candidate_track_id",
    "candidate_hits", "candidate_clear_streak", "candidate_required",
    "reference_distance", "competitor_margin", "candidate_overlap",
    "observed_track_id", "observed_target_id", "x1", "y1", "x2", "y2",
    "forward", "turn",
)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-12, area_a + area_b - intersection)


class GroundTruthIndex:
    """Read-only MOT17 GT for trace attribution; never used by the follower."""

    def __init__(self, sequence_dir: str | Path, *, min_visibility: float = 0.20) -> None:
        if not 0 <= min_visibility <= 1:
            raise ValueError("min_visibility must be in [0, 1]")
        sequence_dir = Path(sequence_dir)
        gt_path = sequence_dir / "gt" / "gt.txt"
        if not gt_path.is_file():
            raise FileNotFoundError(f"GT unavailable: {gt_path}")
        self.sequence_dir = sequence_dir
        self.by_frame: dict[int, list[tuple[int, tuple[float, ...]]]] = defaultdict(list)
        for frame, gt_id, bbox in read_mot_gt(gt_path, min_visibility=min_visibility):
            self.by_frame[frame].append((gt_id, tuple(float(v) for v in bbox)))
        seq_info = configparser.ConfigParser()
        seq_info.read(sequence_dir / "seqinfo.ini", encoding="utf-8-sig")
        self.width = seq_info.getint("Sequence", "imWidth", fallback=0)
        self.height = seq_info.getint("Sequence", "imHeight", fallback=0)
        self.frame_rate = seq_info.getfloat("Sequence", "frameRate", fallback=0.0)

    def validate_frame_size(self, width: int, height: int) -> None:
        if self.width and self.height and (width, height) != (self.width, self.height):
            raise ValueError(
                f"Video {width}x{height} != MOT17 GT {self.width}x{self.height}; "
                "disable --gt-sequence or use an aligned original-resolution video"
            )

    def visible(self, frame: int, identity: int) -> bool:
        return any(gt_id == identity for gt_id, _ in self.by_frame.get(frame, ()))

    def match_box(self, frame: int, bbox: Sequence[float], *, min_iou: float = 0.5,
                  min_margin: float = 0.05) -> tuple[int | None, str]:
        """Return (GT ID, attribution status); ties remain AMBIGUOUS, not WRONG."""
        if not 0 < min_iou <= 1 or not 0 <= min_margin <= 1:
            raise ValueError("Invalid GT attribution thresholds")
        scores = sorted(((box_iou(bbox, gt_box), gt_id)
                         for gt_id, gt_box in self.by_frame.get(frame, ())), reverse=True)
        if not scores or scores[0][0] < min_iou:
            return None, "UNMATCHED"
        if len(scores) > 1 and scores[0][0] - scores[1][0] < min_margin:
            return None, "AMBIGUOUS"
        return scores[0][1], "MATCHED"


class FollowTraceWriter:
    """Append frame records to CSV with deterministic, stable column names."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=TRACE_COLUMNS)
        self._writer.writeheader()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def write(self, frame: int, fps: float, session: int | None, gt_target_id: int | None,
              result: LockResult, command: MotionCommand, diagnostics=None,
              *, event: str = "") -> None:
        observation = result.observation
        bbox = observation.bbox if observation else (None, None, None, None)
        row = {
            "frame": frame, "timestamp_s": f"{(frame - 1) / fps:.4f}",
            "session": session, "gt_target_id": gt_target_id, "event": event,
            "target_id": result.target_id, "state": result.state.value,
            "reason_code": getattr(diagnostics, "reason_code", result.state.value),
            "reason": result.reason,
            "identity_locked": int(getattr(diagnostics, "identity_locked", result.target_id is not None)),
            "motion_authorized": int(result.may_follow),
            "trusted_track_id": getattr(diagnostics, "trusted_track_id", None),
            "candidate_track_id": getattr(diagnostics, "candidate_track_id", None),
            "candidate_hits": getattr(diagnostics, "candidate_hits", None),
            "candidate_clear_streak": getattr(diagnostics, "candidate_clear_streak", None),
            "candidate_required": getattr(diagnostics, "candidate_required", None),
            "reference_distance": getattr(diagnostics, "reference_distance", None),
            "competitor_margin": getattr(diagnostics, "competitor_margin", None),
            "candidate_overlap": getattr(diagnostics, "candidate_overlap", None),
            "observed_track_id": observation.track_id if observation else None,
            "observed_target_id": observation.target_id if observation else None,
            **dict(zip(("x1", "y1", "x2", "y2"), bbox)),
            "forward": command.forward, "turn": command.turn,
        }
        self._writer.writerow({key: "" if value is None else value
                               for key, value in row.items()})


def _nullable_int(value: str | None) -> int | None:
    return None if value in (None, "") else int(value)


def evaluate_follow_trace(trace_csv: str | Path, gt: GroundTruthIndex, *,
                          iou_threshold: float = 0.5, frame_offset: int = 0) -> dict:
    """Evaluate only explicit GT-labeled manual selection sessions.

    A trace without a verified selection_gt_id is never silently scored.
    'Hold while visible' is NOT automatically a false positive: a labeled
    person may be too small or occluded to distinguish safely.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0,1]")
    with Path(trace_csv).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"frame", "session", "gt_target_id", "motion_authorized", "x1", "y1", "x2", "y2"}
        if missing := required - set(reader.fieldnames or ()):
            raise ValueError(f"Missing trace columns: {sorted(missing)}")
        rows = list(reader)
    by_session: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        session, identity = _nullable_int(row["session"]), _nullable_int(row["gt_target_id"])
        if session is not None and identity is not None:
            by_session[session].append(row)

    totals = defaultdict(int)
    recovery_latencies: list[int] = []
    session_summaries = []
    for session_id, events in sorted(by_session.items()):
        selected_ids = {_nullable_int(row["gt_target_id"]) for row in events}
        if len(selected_ids) != 1:
            raise ValueError(f"Session {session_id}: GT target ID changed without reselection")
        gt_id = next(iter(selected_ids))
        session_counts = defaultdict(int)
        hold_start: int | None = None
        for row in events:
            frame = int(row["frame"]) + frame_offset
            session_counts["evaluated_frames"] += 1
            visible = gt.visible(frame, gt_id)
            authorized = row["motion_authorized"] == "1"
            moving = abs(float(row.get("forward") or 0)) > 1e-8 or abs(float(row.get("turn") or 0)) > 1e-8
            session_counts["gt_visible_frames"] += int(visible)
            session_counts["held_frames"] += int(not authorized)
            session_counts["hold_while_gt_visible_frames"] += int(visible and not authorized)
            if not authorized:
                if hold_start is None:
                    hold_start = frame
                if moving:
                    raise ValueError(f"Unsafe trace: unauthorized motion at frame {frame}")
                continue

            session_counts["authorized_frames"] += 1
            session_counts["nonzero_command_frames"] += int(moving)
            if not all(row.get(key) not in (None, "") for key in ("x1", "y1", "x2", "y2")):
                raise ValueError(f"Authorized frame {frame} has no observed bounding box")
            box = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))
            owner, status = gt.match_box(frame, box, min_iou=iou_threshold)
            if status == "AMBIGUOUS":
                session_counts["ambiguous_authorizations"] += 1
            elif status == "UNMATCHED":
                session_counts["unmatched_authorizations"] += 1
            elif owner == gt_id:
                session_counts["correct_authorizations"] += 1
            else:
                session_counts["wrong_person_authorizations"] += 1
                if moving:
                    session_counts["wrong_person_nonzero_commands"] += 1

            if hold_start is not None:
                if owner == gt_id:
                    session_counts["correct_recoveries"] += 1
                    recovery_latencies.append(frame - hold_start)
                elif status == "MATCHED":
                    session_counts["wrong_person_recoveries"] += 1
                hold_start = None

        session_summaries.append({"session": session_id, "gt_target_id": gt_id, **dict(session_counts)})
        for key, value in session_counts.items():
            totals[key] += value

    verified = totals["correct_authorizations"] + totals["wrong_person_authorizations"]
    visible = totals["gt_visible_frames"]
    return {
        "protocol": "PC target-follow trace + MOT17 GT (not MOTChallenge official)",
        "trace_csv": str(trace_csv),
        "sequence": gt.sequence_dir.name,
        "frame_offset": frame_offset,
        "sessions_with_gt": len(session_summaries),
        **dict(totals),
        "authorized_wrong_person_rate_known_gt": (
            totals["wrong_person_authorizations"] / verified if verified else None),
        "correct_authorization_per_visible_gt_frame": (
            totals["correct_authorizations"] / visible if visible else None),
        "hold_while_gt_visible_rate": (
            totals["hold_while_gt_visible_frames"] / visible if visible else None),
        "recovery_latency_frames_mean": mean(recovery_latencies) if recovery_latencies else None,
        "recovery_latency_frames_median": median(recovery_latencies) if recovery_latencies else None,
        "sessions": session_summaries,
        "warning": ("No confirmed GT-linked selection: select a person in original-resolution "
                    "aligned MOT17 video with --gt-sequence and ensure IoU >= 0.5")
                   if not session_summaries else None,
    }
