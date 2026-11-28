"""Target selection without a screen: the person raising both hands claims the robot."""
import numpy as np

SHOULDERS = (5, 6)  # COCO-17 keypoint order, as produced by YOLOv8-pose
WRISTS = (9, 10)


def hand_lift(kpts, frame_h, min_conf=0.5):
    """kpts 17x3 (x, y, conf) in pixels -> [left, right]: how far each wrist is above its
    shoulder, as a fraction of frame height (negative = below); None for a side whose
    shoulder or wrist has conf < min_conf. [None, None] without keypoints."""
    if kpts is None or len(kpts) < 17:
        return [None, None]
    k = np.asarray(kpts, np.float32)
    lifts = []
    for s, w in zip(SHOULDERS, WRISTS):
        seen = k[s, 2] >= min_conf and k[w, 2] >= min_conf
        lifts.append(float(k[s, 1] - k[w, 1]) / frame_h if seen else None)  # image y grows downwards
    return lifts


def hands_raised(kpts, frame_h, margin=0.05, min_conf=0.5):
    """True if each wrist is above its shoulder by more than `margin` x frame height,
    all four keypoints seen with conf >= min_conf."""
    return all(v is not None and v > margin for v in hand_lift(kpts, frame_h, min_conf))


class HandRaiseSelector(object):
    """WAIT phase: select the one person raising both hands, held for hold_frames on one track.

    Others may be in view; two people raising their hands select nobody.
    """

    def __init__(self, hold_frames, frame_h, margin=0.05):
        self.hold, self.h, self.margin = hold_frames, frame_h, margin
        self.track, self.frames = None, 0
        self.raised = 0
        self.why = "nobody raising both hands"  # last reason for no selection, for the periodic log

    def update(self, detections):
        raised = [d for d in detections if hands_raised(d.get("kpts"), self.h, self.margin)]
        self.raised = len(raised)
        d = raised[0] if len(raised) == 1 else None
        if d is None:
            self.why = ("nobody raising both hands" if not raised
                        else "%d people raising hands" % len(raised))
        elif d.get("embedding") is None or d.get("crop_quality", 0.0) < 0.30:
            self.why, d = "hands up but crop too poor for Re-ID (closer, apart from others)", None
        if d is None:
            self.track, self.frames = None, 0
            return None
        self.frames = self.frames + 1 if d["track_id"] == self.track else 1
        self.track = d["track_id"]
        self.why = "hands up %d/%d frames" % (self.frames, self.hold)
        return d if self.frames >= self.hold else None
