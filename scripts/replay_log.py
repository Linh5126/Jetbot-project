"""Replay a recorded demo_pc run offline: same detector, tracker, Re-ID, Guard and controller.

    python scripts/replay_log.py logs/pc_20261127_192720 --device cuda:0
    python scripts/replay_log.py logs/pc_20261127_192720 --variant frozen --variant robot

Reads <prefix>_raw.mp4 (frames as the robot camera sent them) and <prefix>_trace.csv (when the
operator selected, and which box). The person is re-selected at the same frames, so Guard and
controller changes can be compared on real footage without the robot. Variants:
  frozen  guard as before 27/11: one frozen look, no continuity, no blind frames
  robot   configs/jetbot.yaml guard section as it is now
The video is lossy, so tracker IDs and distances differ slightly from the live run.
"""
import argparse
import collections
import sys
from pathlib import Path

import cv2
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.following.controller import StagedControllerConfig, StagedFollowController  # noqa: E402
from src.following.guarded_lock import GuardConfig, GuardedTargetLock  # noqa: E402
from src.following.target_lock import TargetObservation  # noqa: E402
from src.runtime.person_pipeline import PersonIdentityPipeline  # noqa: E402

FROZEN = dict(continuity_distance=0.0, auto_views=0, blind_frames=0)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


def detect(prefix, cfg, device):
    """-> list of detection lists, one per video frame (the expensive part, done once)."""
    pipeline = PersonIdentityPipeline(cfg, mode="v5", strategy="adaptive_fusion", fps=14.0,
                                      device=device, detector_confidence=0.10, guard_features=True)
    cap, out = cv2.VideoCapture(str(prefix) + "_raw.mp4"), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(pipeline.process(frame, len(out) + 1))
    cap.release()
    return out


def replay(frames, selections, guard_cfg, follow_cfg, size):
    lock, ctrl = GuardedTargetLock(guard_cfg), StagedFollowController(follow_cfg)
    rows = []
    for frame_id, dets in enumerate(frames, 1):
        if frame_id in selections:
            box = selections[frame_id]
            best = max(dets, key=lambda d: _iou(d["bbox"], box), default=None)
            if best is not None and _iou(best["bbox"], box) > 0.3:
                try:
                    lock.select(TargetObservation.from_detection(best), frame_id, dets)
                except ValueError as exc:
                    print("frame %d: cannot select: %s" % (frame_id, exc))
            if frame_id > 1:  # select() set the frame counter; update needs the next frame
                rows.append((frame_id, "SELECTED", "MANUAL_SELECTION", "IDLE", 0.0, 0.0))
                continue
        result = lock.update(frame_id, dets)
        command = ctrl.command(result, *size)
        rows.append((frame_id, result.state.value, lock.diagnostics.reason_code, ctrl.phase,
                     command.forward, command.turn))
    return pd.DataFrame(rows, columns=["frame", "state", "reason", "phase", "forward", "turn"]), lock


def summary(name, df, lock, start):
    df = df[df.frame >= start]
    ok = df.state.isin(["TRACKING", "REACQUIRED"])
    runs = (~ok).astype(int).groupby(ok.cumsum()).sum()
    lost = runs[runs > 0]
    print("%-7s followable %5.1f%%  loss episodes %3d  longest %3d frames  views %d+%d  reasons %s" % (
        name, 100 * ok.mean(), len(lost), lost.max() if len(lost) else 0, lock.view_count,
        len(getattr(lock, "_auto_anchors", [])),
        dict(collections.Counter(df.reason[~ok]).most_common(4))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("prefix", help="e.g. logs/pc_20261127_192720")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--robot-config", default=str(ROOT / "configs" / "jetbot.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", default=None, help="detector weights instead of configs/config.yaml's")
    parser.add_argument("--variant", choices=("frozen", "robot"), action="append")
    parser.add_argument("--csv", default=None, help="write the per-frame replay of the last variant")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.weights:
        cfg["detector"]["weights"] = args.weights
    robot = yaml.safe_load(Path(args.robot_config).read_text(encoding="utf-8"))
    trace = pd.read_csv(str(args.prefix) + "_trace.csv")
    # The click happens between two frames: the SELECT event lands on the next trace row.
    selections = {int(r.frame) - 1: (r.x1, r.y1, r.x2, r.y2)
                  for r in trace[trace.event == "SELECT"].itertuples()}
    frames = detect(args.prefix, cfg, args.device)
    fps = float(robot["runtime"]["expected_fps"])
    base = dict(confirm_frames=5, same_track_confirm_frames=2, min_final_clear_frames=2,
                candidate_grace_frames=2, max_lost_frames=max(1, int(robot["runtime"]["max_lost_seconds"] * fps)),
                min_confidence=0.35, min_crop_quality=0.30, ambiguity_margin=0.08, max_overlap=0.35)
    base.update(robot.get("guard") or {})
    follow = StagedControllerConfig(**robot["onboard"]["follow"])
    size = (robot["stream"]["width"], robot["stream"]["height"])
    start = min(selections) if selections else 1
    for name in args.variant or ("frozen", "robot"):
        guard = GuardConfig(**dict(base, **(FROZEN if name == "frozen" else {})))
        df, lock = replay(frames, selections, guard, follow, size)
        summary(name, df, lock, start)
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
