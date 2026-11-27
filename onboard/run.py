"""On-board person following on the Jetson Nano: no PC, no WiFi needed.

    python3 -m onboard.run [--mode guarded|nearest] [--dry-run]

Camera and motors go through the local course agent (127.0.0.1:8100). Both modes share
one pipeline: YOLOv8n-pose TensorRT engine -> target box -> StagedFollowController
(src/following/controller.py) -> optional camera obstacle check -> JetBotHttpDriver.
They differ only in how the target box is chosen (onboard/modes.py):

--mode guarded (default): ByteTrack -> OSNet -> GuardedTargetLock. V5 fusion is not used:
target_id = track_id. With no screen, selection is a ritual:
  WAIT    stand in front of the robot and raise BOTH hands; others may be in view,
          but only one person may raise their hands
  ENROLL  the robot stays still; turn around slowly once (front, side, back),
          views of that one person are added to the reference
  FOLLOW  the robot follows; target lost > max_lost_seconds -> back to WAIT

--mode nearest: baseline without identity, follows whoever is nearest the image centre.

Obstacle check (collision_engine, if built): "blocked" -> no forward motion, turning kept.

Stop: lift the robot / power switch, Ctrl+C, or `systemctl stop` (sends /stop).
"""
import argparse
import csv
import logging
import signal
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from onboard.models import CollisionTrt, YoloTrt  # noqa: E402
from onboard.modes import GuardedMode, NearestMode  # noqa: E402
from src.following.controller import MotionCommand  # noqa: E402
from src.following.jetbot_driver import DriveConfig, JetBotHttpDriver  # noqa: E402
from src.runtime.frame_source import MjpegHttpSource, build_stream_url  # noqa: E402

LOG = logging.getLogger("onboard")
STOP = MotionCommand()
COLUMNS = ["frame", "t_s", "infer_ms", "mode", "phase", "state", "reason", "views", "people",
           "hands_up", "cx", "h", "dist", "p_blocked", "forward", "turn", "left", "right"]


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=str(ROOT / "configs" / "jetbot.yaml"))
    parser.add_argument("--mode", choices=("guarded", "nearest"), default="guarded",
                        help="guarded: hand-raise selection + Re-ID; nearest: person nearest the centre")
    parser.add_argument("--host", default="127.0.0.1", help="agent host (on the Nano: 127.0.0.1)")
    parser.add_argument("--log-csv", default=None, help="per-frame CSV (default logs/onboard_<time>.csv)")
    parser.add_argument("--dry-run", action="store_true", help="never arm the motors")
    parser.add_argument("--agent-wait-s", type=float, default=120.0, help="retry the agent this long")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # urllib3 1.22 (Nano) logs a harmless traceback for every multipart MJPEG response.
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ob = cfg["onboard"]
    robot = dict(cfg["robot"], host=args.host)
    base = "http://%s:%s" % (robot["host"], robot["port"])
    fps = float(ob["expected_fps"])
    frame_h = cfg["stream"]["height"]

    # SIGTERM (systemctl stop) -> same clean path as Ctrl+C, so the driver sends /stop.
    signal.signal(signal.SIGTERM, _raise_interrupt)
    collision = None
    if ob.get("collision_engine") and (ROOT / ob["collision_engine"]).exists():
        collision = CollisionTrt(str(ROOT / ob["collision_engine"]))
    else:
        LOG.warning("No collision engine (%s): obstacle check OFF", ob.get("collision_engine"))
    detector = YoloTrt(str(ROOT / ob["yolo_engine"]), ob["detector_confidence"], ob["detector_iou"])
    mode = GuardedMode(cfg, fps, frame_h, ROOT) if args.mode == "guarded" else NearestMode(cfg, frame_h)
    driver = JetBotHttpDriver(base, DriveConfig(**cfg["drive"]))
    deadline = time.monotonic() + args.agent_wait_s  # at boot the agent's container starts late
    while True:
        try:
            battery = (driver.preflight() or {}).get("battery") or {}
            if isinstance(battery.get("percent"), (int, float)) and battery["percent"] < 40:
                LOG.warning("Battery %s %%: wheels may creep then jump; charge before following", battery["percent"])
            break
        except Exception as exc:
            if time.monotonic() > deadline:
                driver.close()
                raise
            LOG.info("Waiting for the agent at %s: %s", base, exc)
            time.sleep(2)
    source = MjpegHttpSource(build_stream_url(robot, cfg["stream"]),
                             cfg["stream"]["connect_timeout_s"], cfg["stream"]["read_timeout_s"])

    log_path = Path(args.log_csv or ROOT / "logs" / time.strftime("onboard_%Y%m%d_%H%M%S.csv"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", newline="")
    log = csv.writer(log_file)
    log.writerow(COLUMNS)
    LOG.info("Ready (%s mode%s): %s, log %s.%s", args.mode, ", dry run: motors stay off" if args.dry_run else "",
             base, log_path, " Raise both hands in front of the robot to be selected."
             if args.mode == "guarded" else "")

    frame_id, t0 = 0, time.monotonic()
    try:
        while True:
            packet = source.read(cfg["runtime"]["no_frame_timeout_s"])
            if packet is None:
                driver.send(STOP)  # no fresh image -> stop
                continue
            frame, start = packet.frame, time.monotonic()
            frame_id += 1
            boxes, scores, kpts = detector(frame)
            p_blocked = collision(frame) if collision is not None else None
            now = time.monotonic()
            wanted, fields = mode.step(frame, frame_id, boxes, scores, kpts, now)
            if p_blocked is not None and p_blocked >= ob["collision_threshold"] and wanted.forward > 0:
                wanted, fields["reason"] = MotionCommand(turn=wanted.turn), "BLOCKED"

            fresh = now - packet.t_recv <= cfg["runtime"]["stale_frame_s"]
            command = wanted if fresh and not args.dry_run else STOP
            driver.send(command)

            bbox = fields.pop("bbox")
            cx = h = ""
            if bbox is not None:
                cx = "%.3f" % ((bbox[0] + bbox[2]) / (2.0 * frame.shape[1]))
                h = "%.3f" % ((bbox[3] - bbox[1]) / float(frame.shape[0]))
            left, right = driver.last_wheels
            fields.update(frame=frame_id, t_s="%.3f" % (now - t0), infer_ms="%.1f" % (1000 * (now - start)),
                          mode=args.mode, cx=cx, h=h, p_blocked="" if p_blocked is None else "%.2f" % p_blocked,
                          forward="%.3f" % wanted.forward, turn="%.3f" % wanted.turn,
                          left="%.3f" % left, right="%.3f" % right)
            log.writerow([fields[c] for c in COLUMNS])
            if frame_id % int(10 * fps) == 0:
                log_file.flush()
                LOG.info("frame %d, %.1f FPS, phase %s, state %s%s", frame_id, frame_id / (now - t0),
                         fields["phase"], fields["state"], mode.status())
    except KeyboardInterrupt:
        LOG.info("Stopping")
    finally:
        driver.close()  # always ends with /stop
        source.close()
        log_file.close()


if __name__ == "__main__":
    main()
