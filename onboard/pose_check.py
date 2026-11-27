"""Pose check before any following: sit (or stand) in front of the camera and raise both hands.

    python3 -m onboard.pose_check [--seconds 90] [--hold 5]

Reads the agent's camera stream and runs the YOLOv8n-pose engine only. It never sends a
motor command, needs no Re-ID engine and no enrolment, so the robot cannot move.

About twice a second it prints every person seen: box height (fraction of the frame) and,
for each hand, how far the wrist is above its shoulder (fraction of frame height,
'-' = shoulder or wrist not seen well enough). RAISED = both hands above select_hand_margin,
the rule the guarded mode selects with. PASS (exit 0) once exactly one person stays RAISED
for --hold frames in a row; FAIL (exit 1) at --seconds with the most frequent reason.
A snapshot with the keypoints drawn goes to logs/pose_check.jpg (the PASS frame, else the
last frame with a person in it).
"""
import argparse
import collections
import sys
import time
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from onboard.models import YoloTrt  # noqa: E402
from onboard.selection import SHOULDERS, WRISTS, hand_lift  # noqa: E402
from src.runtime.frame_source import MjpegHttpSource, build_stream_url  # noqa: E402

MIN_SCORE = 0.25  # same as the tracker's high-score threshold (configs/bytetrack_custom.yaml)


def describe(boxes, scores, kpts, frame_h, margin):
    """-> (people [(height, [left, right], raised)], reason the frame does not select anyone or None)."""
    people = []
    for box, score, k in zip(boxes, scores, kpts):
        if score < MIN_SCORE:
            continue
        lifts = hand_lift(k, frame_h)
        raised = all(v is not None and v > margin for v in lifts)
        people.append(((box[3] - box[1]) / float(frame_h), lifts, raised))
    n_raised = sum(1 for p in people if p[2])
    if n_raised == 1:
        return people, None
    if n_raised > 1:
        return people, "%d people raising hands" % n_raised
    if not people:
        return people, "no person seen"
    if all(None in p[1] for p in people):
        return people, "shoulder/wrist not seen (sit further back, hands inside the frame, more light)"
    return people, "wrists not above shoulders by %.2f of the frame height" % margin


def _fmt(v):
    return "-" if v is None else "%+.2f" % v


def draw(frame, boxes, scores, kpts):
    view = frame.copy()
    for box, score, k in zip(boxes, scores, kpts):
        if score < MIN_SCORE:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 255), 1)
        for i, colour in [(s, (255, 0, 0)) for s in SHOULDERS] + [(w, (0, 0, 255)) for w in WRISTS]:
            if k[i, 2] >= 0.5:
                cv2.circle(view, (int(k[i, 0]), int(k[i, 1])), 4, colour, -1)
    return view


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=str(ROOT / "configs" / "jetbot.yaml"))
    parser.add_argument("--host", default="127.0.0.1", help="agent host (on the Nano: 127.0.0.1)")
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--hold", type=int, default=None, help="default: onboard.select_hold_frames")
    parser.add_argument("--snapshot", default=str(ROOT / "logs" / "pose_check.jpg"))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ob = cfg["onboard"]
    hold = args.hold or ob["select_hold_frames"]
    margin = ob["select_hand_margin"]
    robot = dict(cfg["robot"], host=args.host)
    detector = YoloTrt(str(ROOT / ob["yolo_engine"]), ob["detector_confidence"], ob["detector_iou"])
    source = MjpegHttpSource(build_stream_url(robot, cfg["stream"]),
                             cfg["stream"]["connect_timeout_s"], cfg["stream"]["read_timeout_s"])
    Path(args.snapshot).parent.mkdir(parents=True, exist_ok=True)
    print("Pose check %.0f s (motors untouched): sit 1.5-2 m in front of the camera, raise both hands."
          % args.seconds, flush=True)
    reasons, streak, frames, last_seen = collections.Counter(), 0, 0, None
    start = next_print = time.monotonic()
    try:
        while time.monotonic() - start < args.seconds:
            packet = source.read(cfg["runtime"]["no_frame_timeout_s"])
            if packet is None:
                continue
            frame = packet.frame
            frames += 1
            boxes, scores, kpts = detector(frame)
            people, why = describe(boxes, scores, kpts, frame.shape[0], margin)
            if people:
                last_seen = (frame, boxes, scores, kpts)
            streak = streak + 1 if why is None else 0
            reasons[why or "RAISED"] += 1
            now = time.monotonic()
            if now >= next_print or streak >= hold:
                next_print = now + 0.5
                text = "  ".join("[h %.2f L %s R %s%s]" % (h, _fmt(l[0]), _fmt(l[1]), " RAISED" if r else "")
                                 for h, l, r in people) or "[nobody]"
                print("%5.1fs %s%s" % (now - start, text, "" if why is None else "  <- " + why), flush=True)
            if streak >= hold:
                cv2.imwrite(args.snapshot, draw(frame, boxes, scores, kpts))
                print("PASS: one person raised both hands for %d frames in a row (%.1f s). Snapshot %s"
                      % (hold, now - start, args.snapshot), flush=True)
                return 0
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
    if last_seen is not None:
        cv2.imwrite(args.snapshot, draw(*last_seen))
    seconds = max(time.monotonic() - start, 1e-6)
    print("FAIL after %d frames (%.1f FPS). Frames per reason: %s. Snapshot %s"
          % (frames, frames / seconds, dict(reasons.most_common()), args.snapshot if last_seen is not None else "none"),
          flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
