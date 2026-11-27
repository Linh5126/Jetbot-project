"""JetBot connection check and motor calibration helpers (agent HTTP API on :8100).

  python scripts/jetbot_tool.py check
  python scripts/jetbot_tool.py pulse --left 0.2 --right 0.2 --seconds 1   # ramped like the follower
  python scripts/jetbot_tool.py ramp --wheel both      # Ctrl+C when the wheel starts turning

Keep the wheels OFF the ground while using pulse/ramp.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.following.jetbot_driver import _step  # noqa: E402
from src.runtime.frame_source import MjpegHttpSource, build_stream_url  # noqa: E402

MAX_PULSE_WHEEL = 0.6
MAX_PULSE_SECONDS = 3.0


def base_url(cfg: dict) -> str:
    return f"http://{cfg['robot']['host']}:{cfg['robot']['port']}"


def stop(session: requests.Session, base: str) -> None:
    for _ in range(2):
        try:
            session.post(base + "/stop", json={}, timeout=0.5)
        except requests.RequestException as exc:
            print(f"WARNING: /stop failed: {exc}")


def drive_for(session: requests.Session, base: str, left: float, right: float, seconds: float,
              ramp: tuple = None) -> None:
    """ramp = (max_step_per_tick, wheel_deadzone): accelerate like JetBotHttpDriver instead of a step."""
    end = time.monotonic() + seconds
    sent = (0.0, 0.0)
    while time.monotonic() < end:
        sent = (left, right) if ramp is None else (_step(sent[0], left, *ramp), _step(sent[1], right, *ramp))
        session.post(base + "/drive", json={"left": sent[0], "right": sent[1]}, timeout=0.3).raise_for_status()
        time.sleep(0.1)


def battery_warning(session: requests.Session, base: str) -> None:
    battery = (session.get(base + "/status", timeout=2).json() or {}).get("battery") or {}
    print("battery:", battery)
    if isinstance(battery.get("percent"), (int, float)) and battery["percent"] < 50:
        print("WARNING: battery below 50 %: wheels need more power to start (they creep, then jump)."
              " Charge before measuring speed or the dead zone.")


def cmd_check(cfg: dict, args) -> int:
    base = base_url(cfg)
    with requests.Session() as s:
        health = s.get(base + "/health", timeout=2)
        status = s.get(base + "/status", timeout=2)
        health.raise_for_status()
        status.raise_for_status()
        health, status = health.json(), status.json()
        print("/health:", health)
        print("/status:", status)
        if isinstance(status, dict):
            motor = status.get("motor")
            print("motor.ok:", motor.get("ok") if isinstance(motor, dict) else motor)
            print("camera:", status.get("camera"))
            print("imu:", status.get("imu"))
        rtts = []
        for _ in range(10):
            t0 = time.perf_counter()
            s.get(base + "/health", timeout=2)
            rtts.append(1000 * (time.perf_counter() - t0))
        print(f"RTT /health ms: min {min(rtts):.1f}  mean {statistics.mean(rtts):.1f}  max {max(rtts):.1f}")
    url = build_stream_url(cfg["robot"], cfg["stream"])
    print("stream:", url)
    src = MjpegHttpSource(url, float(cfg["stream"]["connect_timeout_s"]),
                          float(cfg["stream"]["read_timeout_s"]))
    frames, last, t_end = 0, None, time.monotonic() + args.seconds
    try:
        t_first = None
        while time.monotonic() < t_end:
            packet = src.read(0.5)
            if packet is not None:
                t_first = t_first or time.monotonic()
                frames += 1
                last = packet.frame
    finally:
        src.close()
    if last is None:
        print("ERROR: no frame received from /stream")
        return 1
    span = max(1e-6, time.monotonic() - t_first)
    print(f"stream FPS: {frames / span:.1f}  frame size: {last.shape[1]}x{last.shape[0]}")
    out = ROOT / "outputs" / "jetbot_check.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), last)
    print("saved last frame:", out)
    return 0


def cmd_pulse(cfg: dict, args) -> int:
    if abs(args.left) > MAX_PULSE_WHEEL or abs(args.right) > MAX_PULSE_WHEEL or not 0 < args.seconds <= MAX_PULSE_SECONDS:
        print(f"Refused: need |left|,|right| <= {MAX_PULSE_WHEEL} and 0 < seconds <= {MAX_PULSE_SECONDS}")
        return 2
    base = base_url(cfg)
    drive = cfg["drive"]
    ramp = None if args.no_ramp else (drive["max_step_per_tick"], drive["wheel_deadzone"])
    with requests.Session() as s:
        battery_warning(s, base)
        try:
            print(f"drive left={args.left} right={args.right} for {args.seconds}s"
                  + (" (step, no ramp)" if ramp is None else " (ramped from the dead zone, as the follower)"))
            drive_for(s, base, args.left, args.right, args.seconds, ramp)
        finally:
            stop(s, base)
    return 0


def cmd_ramp(cfg: dict, args) -> int:
    base = base_url(cfg)
    level = 0.02  # course agent min_duty=0: wheels already turn at 0.10 on this bot
    print("Ramp 0.02 -> 0.40. Press Ctrl+C as soon as the wheel starts turning;"
          " the last printed level is drive.wheel_deadzone.")
    with requests.Session() as s:
        battery_warning(s, base)
        try:
            while level <= 0.40 + 1e-9:
                left = level if args.wheel in ("left", "both") else 0.0
                right = level if args.wheel in ("right", "both") else 0.0
                print(f"level {level:.2f}", flush=True)
                drive_for(s, base, left, right, 0.8)
                level = round(level + 0.02, 2)
            print("Reached 0.40 without Ctrl+C.")
        except KeyboardInterrupt:
            print(f"\nStopped. wheel_deadzone ~ {level:.2f}")
        finally:
            stop(s, base)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-config", default=str(ROOT / "configs" / "jetbot.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="health/status, RTT, 5 s stream FPS, save a frame")
    check.add_argument("--seconds", type=float, default=5.0)
    pulse = sub.add_parser("pulse", help="drive both wheels briefly, then /stop")
    pulse.add_argument("--left", type=float, required=True)
    pulse.add_argument("--right", type=float, required=True)
    pulse.add_argument("--seconds", type=float, required=True)
    pulse.add_argument("--no-ramp", action="store_true", help="jump straight to the wheel values")
    ramp = sub.add_parser("ramp", help="find the wheel dead zone")
    ramp.add_argument("--wheel", choices=("left", "right", "both"), default="both")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.robot_config).read_text(encoding="utf-8"))
    return {"check": cmd_check, "pulse": cmd_pulse, "ramp": cmd_ramp}[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
