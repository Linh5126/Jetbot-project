"""Collect free/blocked images for the camera obstacle classifier (run on the PC, robot on WiFi).

    python scripts/collect_collision.py

Place the robot by hand, look at the window, then press:
  b  blocked: an obstacle ~20-30 cm in front (wall, chair leg, box, feet up close)
  f  free:    the way ahead is clear, INCLUDING a person standing 1-2 m ahead
              (the follower must not treat its own target as an obstacle)
  q  quit
Aim for 100+ of each, many places and lighting. Images: data/collision/{blocked,free}/.
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runtime.frame_source import MjpegHttpSource, build_stream_url  # noqa: E402

KEYS = {ord("b"): "blocked", ord("f"): "free"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--robot-config", default=str(ROOT / "configs" / "jetbot.yaml"))
    parser.add_argument("--out", default=str(ROOT / "data" / "collision"))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.robot_config).read_text(encoding="utf-8"))
    out = Path(args.out)
    for name in KEYS.values():
        (out / name).mkdir(parents=True, exist_ok=True)
    source = MjpegHttpSource(build_stream_url(cfg["robot"], cfg["stream"]),
                             cfg["stream"]["connect_timeout_s"], cfg["stream"]["read_timeout_s"])
    frame = None
    try:
        while True:
            packet = source.read(0.5)
            if packet is not None:
                frame = packet.frame
            if frame is None:
                if cv2.waitKey(50) & 0xFF == ord("q"):
                    break
                continue
            counts = {n: len(list((out / n).glob("*.jpg"))) for n in KEYS.values()}
            view = frame.copy()
            cv2.putText(view, "b blocked %(blocked)d   f free %(free)d   q quit" % counts,
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("collect_collision", view)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key in KEYS:
                path = out / KEYS[key] / f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
                cv2.imwrite(str(path), frame)  # the raw stream frame, same as the robot sees
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
