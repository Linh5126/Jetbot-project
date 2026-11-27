"""The two on-board following modes. They differ only in how the target box is chosen;
both drive it with the same StagedFollowController.

guarded  hand-raise selection, enrolment, Re-ID verified target (GuardedTargetLock)
nearest  baseline without identity: whoever is nearest the image centre, every frame
"""
import logging

import numpy as np
import yaml

from onboard.bytetrack import ByteTracker
from onboard.models import OsnetTrt
from onboard.selection import HandRaiseSelector, hands_raised
from src.following.controller import StagedControllerConfig, StagedFollowController
from src.following.guarded_lock import GuardConfig, GuardedTargetLock
from src.following.target_lock import FollowState, TargetObservation
from src.reid.crop_quality import score_person_crop_quality

LOG = logging.getLogger("onboard")


def add_identity_features(frame, detections, reid):
    """Same fields as PersonIdentityPipeline(guard_features=True), target_id = track_id."""
    h, w = frame.shape[:2]
    crops, chosen = [], []
    for d in detections:
        d["target_id"] = d["track_id"]
        x1, y1, x2, y2 = [int(round(v)) for v in d["bbox"]]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if y2 - y1 < 28 or d["confidence"] < 0.35 or x2 <= x1:
            continue
        crop = frame[y1:y2, x1:x2]
        d["crop_quality"] = score_person_crop_quality(crop, d["bbox"], frame.shape, d["confidence"])["score"]
        crops.append(crop)
        chosen.append(d)
    for d, vector in zip(chosen, reid(crops)):
        d["embedding"] = vector
    return detections


def nearest_to_centre(boxes, frame_w, frame_h):
    """Index of the box whose centre is nearest the image centre (normalised), or None."""
    if len(boxes) == 0:
        return None
    b = np.asarray(boxes, np.float32)
    dx = (b[:, 0] + b[:, 2]) / (2.0 * frame_w) - 0.5
    dy = (b[:, 1] + b[:, 3]) / (2.0 * frame_h) - 0.5
    return int(np.argmin(dx * dx + dy * dy))


class GuardedMode(object):
    """WAIT (raise both hands) -> ENROLL (turn around, views added) -> FOLLOW (verified target)."""

    def __init__(self, cfg, fps, frame_h, root):
        ob = cfg["onboard"]
        self.ob, self.runtime = ob, cfg["runtime"]
        self.reid = OsnetTrt(str(root / ob["osnet_engine"]))
        self.tracker = ByteTracker(**yaml.safe_load((root / "configs" / "bytetrack_custom.yaml").read_text()))
        self.guard = GuardedTargetLock(GuardConfig(
            max_lost_frames=max(1, int(cfg["runtime"]["max_lost_seconds"] * fps)), **dict(cfg.get("guard") or {})))
        self.controller = StagedFollowController(StagedControllerConfig(**ob["follow"]))
        self.selector = HandRaiseSelector(ob["select_hold_frames"], frame_h, ob["select_hand_margin"])
        self.phase, self.enroll_until, self.lost_since, self.next_view_at = "WAIT", None, None, 0.0

    def step(self, frame, frame_id, boxes, scores, kpts, now):
        """-> (MotionCommand wanted, log fields). The caller decides whether to arm the motors."""
        detections = add_identity_features(frame, self.tracker.update(boxes, scores, kpts), self.reid)
        guard = self.guard
        result = guard.update(frame_id, detections)
        if self.phase == "WAIT":
            chosen = self.selector.update(detections)
            if chosen is not None:
                try:
                    result = guard.select(TargetObservation.from_detection(chosen), frame_id, detections)
                    self.phase, self.enroll_until = "ENROLL", now + self.ob["enroll_seconds"]
                    LOG.info("Selected track %d; turn around slowly for %.0f s",
                             chosen["track_id"], self.ob["enroll_seconds"])
                except ValueError as exc:
                    LOG.warning("Cannot select: %s", exc)
        elif self.phase == "ENROLL":
            # Robot is still: a new verified look of the target becomes a view.
            dist = guard.diagnostics.reference_distance
            # add_view itself refuses a target that overlaps someone else.
            if (result.may_follow and dist is not None
                    and dist >= self.ob["enroll_new_view_distance"] and now >= self.next_view_at
                    and guard.view_count < guard.config.max_views):
                try:
                    LOG.info("View %d added (d=%.3f)", guard.add_view(frame_id, detections), dist)
                    self.next_view_at = now + self.ob["enroll_min_interval_s"]
                except ValueError as exc:
                    LOG.debug("View not added: %s", exc)
            if now >= self.enroll_until:
                self.phase = "FOLLOW"
                self.controller.reset()
                LOG.warning("FOLLOW with %d views", guard.view_count)
        elif self.phase == "FOLLOW":
            holding = result.state in (FollowState.LOST, FollowState.VERIFYING)
            self.lost_since = (self.lost_since or now) if holding else None
            if result.state == FollowState.STOPPED or (
                    self.lost_since is not None and now - self.lost_since > self.runtime["max_lost_seconds"]):
                result = guard.stop("Target lost timeout")
                guard.reset()
                self.phase, self.lost_since = "WAIT", None
                LOG.warning("Target lost; WAIT for a new selection")

        command = self.controller.command(result, frame.shape[1], frame.shape[0])
        if self.phase != "FOLLOW":
            command = type(command)()  # the robot stays still while selecting and enrolling
        dist = guard.diagnostics.reference_distance
        fields = {"phase": self.phase, "state": result.state.value,
                  "reason": self.controller.phase if self.phase == "FOLLOW" else guard.diagnostics.reason_code,
                  "views": guard.view_count, "people": len(detections), "hands_up": self.selector.raised,
                  "dist": "" if dist is None else "%.3f" % dist,
                  "bbox": None if result.observation is None else result.observation.bbox}
        return command, fields

    def status(self):
        return " | not selected: " + self.selector.why if self.phase == "WAIT" else ""


class NearestMode(object):
    """Baseline: follow whoever is nearest the image centre. No tracker, no identity."""

    def __init__(self, cfg, frame_h):
        ob = cfg["onboard"]
        self.controller = StagedFollowController(StagedControllerConfig(**ob["follow"]))
        self.h, self.margin = frame_h, ob["select_hand_margin"]

    def step(self, frame, frame_id, boxes, scores, kpts, now):
        i = nearest_to_centre(boxes, frame.shape[1], frame.shape[0])
        bbox = None if i is None else [float(v) for v in boxes[i]]
        command = self.controller.command_for_box(bbox, frame.shape[1], frame.shape[0])
        fields = {"phase": "FOLLOW", "state": "NO_PERSON" if i is None else "NEAREST",
                  "reason": self.controller.phase, "views": "", "people": len(boxes),
                  "hands_up": sum(1 for k in kpts if hands_raised(k, self.h, self.margin)),
                  "dist": "", "bbox": bbox}
        return command, fields

    def status(self):
        return ""
