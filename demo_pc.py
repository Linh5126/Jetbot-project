"""Interactive JetBot person-following demo (AI runs on the PC).

Click a person to lock their persistent identity. With --driver mock (default)
commands go to a MOCK driver; with --driver jetbot they are sent to the real
robot over HTTP. The real robot always starts SAFE: press M to arm motors.

Examples:
  python demo_pc.py --mode baseline --source 0
  python demo_pc.py --mode v5_1 --source videos/crossing.mp4 --record-video demo.mp4
  python demo_pc.py --source jetbot --driver mock --device cuda:0
  python demo_pc.py --source jetbot --driver jetbot --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import cv2
import yaml

from src.following.controller import ControllerConfig, FollowController, MotionCommand
from src.following.driver import MockRobotDriver
from src.following.jetbot_driver import DriveConfig, JetBotHttpDriver
from src.following.target_lock import FollowState, TargetLockManager, TargetObservation
from src.following.guarded_lock import GuardConfig, GuardedTargetLock
from src.evaluation.follow_trace import FollowTraceWriter, GroundTruthIndex
from src.runtime.frame_source import build_stream_url, open_source
from src.runtime.person_pipeline import PersonIdentityPipeline
from src.runtime.overlay import render_demo_frame
from src.runtime.dashboard import (DashboardTelemetry, render_dashboard_frame,
                                   fit_display_frame, map_display_click)
from src.identity.manager_v5 import IdentityManagerV5
from src.reid.model_config import REID_CHECKPOINTS, select_reid_model

LOG = logging.getLogger("jetbot-pc-demo")


def parse_source(raw: str):
    return int(raw) if raw.isdecimal() else raw


def select_at_point(detections: list[dict], x: int, y: int):
    """Select the smallest box under the cursor, ignoring nested large boxes."""
    candidates = [det for det in detections if
                  det["bbox"][0] <= x <= det["bbox"][2] and
                  det["bbox"][1] <= y <= det["bbox"][3]]
    return min(candidates, key=lambda d: (d["bbox"][2] - d["bbox"][0]) *
               (d["bbox"][3] - d["bbox"][1])) if candidates else None


def draw_frame(frame, detections, result, command: MotionCommand, mode: str, diagnostics=None):
    """Backward-compatible entrypoint for the PC overlay renderer."""
    return render_demo_frame(frame, detections, result, command, mode, diagnostics=diagnostics)


def run(args) -> None:
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg = select_reid_model(cfg, args.reid_model)
    if args.mode != "baseline":
        reid = cfg["reid"]["v5"]
        LOG.info("Re-ID: %s | checkpoint=%s", reid["model_name"], reid["checkpoint"])
    robot_cfg = None
    if args.driver == "jetbot" or args.source == "jetbot":
        robot_cfg = yaml.safe_load(Path(args.robot_config).read_text(encoding="utf-8"))
    raw_source = (build_stream_url(robot_cfg["robot"], robot_cfg["stream"])
                  if args.source == "jetbot" else args.source)
    source = open_source(raw_source, robot_cfg["stream"] if robot_cfg else None)
    writer = raw_writer = robot_log_file = robot_log = None
    jetbot = args.driver == "jetbot"
    if jetbot:
        base = f"http://{robot_cfg['robot']['host']}:{robot_cfg['robot']['port']}"
        driver = JetBotHttpDriver(base, DriveConfig(**robot_cfg["drive"]))
        try:
            LOG.info("JetBot status: %s", driver.preflight())
        except Exception:
            driver.close()
            source.close()
            raise
        controller = FollowController(ControllerConfig(**robot_cfg["controller"]))
    else:
        driver = MockRobotDriver()
        controller = FollowController(ControllerConfig())
    armed = not jetbot  # the real robot always starts SAFE
    max_lost_seconds = (float(robot_cfg["runtime"]["max_lost_seconds"])
                        if robot_cfg else args.max_lost_seconds)
    window = "JetBot LIVE (REAL MOTORS, M = arm)" if jetbot else "JetBot PC dry run (MOCK, no motors)"
    runtime = {"detections": [], "frame_id": 0, "lock_result": None,
               "session": None, "session_count": 0, "gt_target_id": None,
               "pending_event": "", "frame": None, "reference_crop": None,
               "display_scale": 1.0}
    trace = FollowTraceWriter(args.trace_csv) if args.trace_csv else None
    gt = None
    if args.gt_sequence:
        seq_path = Path(args.gt_sequence)
        if not (seq_path / "gt" / "gt.txt").exists():
            seq_path = Path(args.gt_root) / args.gt_sequence
        gt = GroundTruthIndex(seq_path, min_visibility=args.gt_min_visibility)
    if source.live:
        fps = float(robot_cfg["runtime"]["expected_fps"]) if robot_cfg else 15.0
    else:
        fps = (source.nominal_fps if source.nominal_fps and 1.0 <= source.nominal_fps <= 120.0
               else 30.0)
    if gt is not None and 1 <= gt.frame_rate <= 120:
        fps = gt.frame_rate  # Image sequences otherwise default to arbitrary 25 FPS.
    guarded = args.identity_guard == "on" and args.mode != "baseline"
    lock = (
        GuardedTargetLock(GuardConfig(
            confirm_frames=args.confirm_frames,
            same_track_confirm_frames=args.same_track_confirm_frames,
            min_final_clear_frames=args.guard_final_clear_frames,
            candidate_grace_frames=args.guard_grace_frames,
            same_track_recovery_distance=args.guard_same_track_recovery_distance,
            max_lost_frames=max(1, int(max_lost_seconds * fps)),
            min_confidence=args.min_target_confidence,
            min_crop_quality=args.guard_min_quality,
            track_distance=args.guard_track_distance,
            recovery_distance=args.guard_recovery_distance,
            ambiguity_margin=args.guard_margin,
            max_overlap=args.guard_overlap,
        ))
        if guarded else TargetLockManager(
            recovery_confirm_frames=args.confirm_frames,
            max_lost_frames=max(1, int(max_lost_seconds * fps)),
            min_confidence=args.min_target_confidence,
        )
    )
    dashboard = DashboardTelemetry()
    view_mode = args.dashboard_view

    def click(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # imshow receives a scaled copy; map clicks back to raw camera pixels.
        cam_x, cam_y = map_display_click(x, y, runtime["display_scale"])
        selected = select_at_point(runtime["detections"], cam_x, cam_y)
        if selected is not None:
            try:
                obs = TargetObservation.from_detection(selected)
                runtime["lock_result"] = (
                    lock.select(obs, runtime["frame_id"], runtime["detections"])
                    if guarded else lock.select(obs, runtime["frame_id"])
                )
                runtime["session_count"] += 1
                runtime["session"] = runtime["session_count"]
                runtime["pending_event"] = "SELECT"
                runtime["gt_target_id"] = None
                # UI-only snapshot; the Identity Guard keeps its own immutable
                # embedding anchor. This crop is NEVER fed back to inference.
                current = runtime["frame"]
                if current is not None:
                    x1, y1, x2, y2 = (int(round(v)) for v in selected["bbox"])
                    hh, ww = current.shape[:2]
                    x1, x2 = max(0, x1), min(ww, x2)
                    y1, y2 = max(0, y1), min(hh, y2)
                    runtime["reference_crop"] = (current[y1:y2, x1:x2].copy()
                                                  if x2 > x1 and y2 > y1 else None)
                dashboard.add_event(runtime["frame_id"], "TARGET SELECTED")
                if gt is not None:
                    gt_identity, quality = gt.match_box(
                        runtime["frame_id"] + args.gt_frame_offset, selected["bbox"])
                    runtime["gt_target_id"] = gt_identity
                    if gt_identity is None:
                        LOG.warning("Selected box has no unique GT match (%s); "
                                    "this session cannot be GT-scored", quality)
                    else:
                        LOG.info("Selection matched to MOT17 GT person %s", gt_identity)
                LOG.info("Selected target_id=%s", selected["target_id"])
            except ValueError as exc:
                LOG.warning("Cannot select target: %s", exc)

    try:
        pipeline = PersonIdentityPipeline(
            cfg, mode=args.mode, strategy=args.ablation, fps=fps,
            device=args.device, detector_confidence=args.confidence,
            guard_features=guarded,
        )
        if trace is not None:
            # Keep model settings alongside the frame-wise trace for reproducible A/B runs.
            details = {
                "protocol": "JetBot PC target-follow dry run (not official MOTChallenge)",
                "source": args.source,
                "mode": args.mode,
                "strategy": args.ablation,
                "detector_weights": cfg["detector"]["weights"],
                "detector_confidence": args.confidence,
                "reid_model": cfg["reid"]["v5"]["model_name"] if args.mode != "baseline" else None,
                "reid_checkpoint": cfg["reid"]["v5"]["checkpoint"] if args.mode != "baseline" else None,
                "identity_guard": guarded,
                "guard_track_distance": args.guard_track_distance,
                "guard_recovery_distance": args.guard_recovery_distance,
                "guard_overlap": args.guard_overlap,
                "gt_sequence": args.gt_sequence,
                "gt_frame_offset": args.gt_frame_offset,
            }
            if robot_cfg is not None:  # keep mock+file run.json unchanged
                details.update(driver=args.driver, robot_config=robot_cfg)
            trace.path.with_suffix(".run.json").write_text(
                json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, click)
        if args.robot_log:
            path = Path(args.robot_log)
            path.parent.mkdir(parents=True, exist_ok=True)
            robot_log_file = path.open("w", newline="", encoding="utf-8")
            robot_log = csv.writer(robot_log_file)
            robot_log.writerow(["frame", "t_wall_s", "frame_age_ms", "loop_ms", "state", "gate",
                                "armed", "forward", "turn", "left", "right", "link_ok"])
        frame_timeout = (robot_cfg["runtime"]["no_frame_timeout_s"]
                         if (robot_cfg and source.live) else 5.0)
        stale_s = robot_cfg["runtime"]["stale_frame_s"] if robot_cfg else 0.35
        lost_since = None
        last_time = time.perf_counter()
        processed = 0
        while True:
            frame_start = time.perf_counter()
            packet = source.read(frame_timeout)
            if packet is None:
                if source.ended or not source.live:
                    LOG.info("Video source ended/unavailable; requested STOP")
                    break
                driver.send(MotionCommand())  # no fresh image -> stop
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
                continue
            frame = packet.frame
            if args.record_raw and raw_writer is None:
                path = Path(args.record_raw)
                path.parent.mkdir(parents=True, exist_ok=True)
                raw_writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (frame.shape[1], frame.shape[0]))
                if not raw_writer.isOpened():
                    raise RuntimeError(f"Cannot write video: {path}")
            if raw_writer is not None:
                raw_writer.write(frame)  # before any rendering touches the frame
            processed += 1
            runtime["frame_id"] = processed
            runtime["frame"] = frame
            if gt is not None and processed == 1:
                gt.validate_frame_size(frame.shape[1], frame.shape[0])
            detections = pipeline.process(frame, processed)
            runtime["detections"] = detections
            result = lock.update(processed, detections)
            old_result = runtime["lock_result"]
            if old_result is None or old_result.state != result.state:
                LOG.info("Target state=%s reason=%s", result.state.value, result.reason)
            runtime["lock_result"] = result
            command = controller.command(result, frame.shape[1], frame.shape[0])
            # Safety gate; the trace/dashboard below must see the gated command.
            now = time.monotonic()
            gate = ""
            if source.live:
                holding = result.state in (FollowState.LOST, FollowState.VERIFYING)
                lost_since = (lost_since or now) if holding else None
                if lost_since is not None and now - lost_since > max_lost_seconds:
                    result = lock.stop("Target lost timeout (wall clock)")
                    runtime["lock_result"] = result
                    lost_since = None
                    gate = "LOST_TIMEOUT"
                    LOG.info("Target state=%s reason=%s", result.state.value, result.reason)
                if now - packet.t_recv > stale_s:
                    gate = gate or "STALE_FRAME"
            if not armed:
                gate = gate or "SAFE"
            if gate:
                command = MotionCommand()
            driver.send(command)
            diagnostics = getattr(lock, "diagnostics", None)
            if trace is not None:
                trace.write(processed, fps, runtime["session"], runtime["gt_target_id"],
                            result, command, diagnostics,
                            event=runtime["pending_event"])
                runtime["pending_event"] = ""
            model_label = (args.mode if args.mode == "baseline" else
                           f"{args.mode} / {cfg['reid']['v5']['model_name']}")
            if args.ui == "dashboard" and args.mode != "baseline":
                model_label += f" / {args.ablation}"
            if args.ui == "dashboard":
                dashboard.observe(
                    frame_id=processed, result=result, people=len(detections),
                    elapsed_s=time.perf_counter() - frame_start,
                )
                rendered = render_dashboard_frame(
                    frame, detections, result, command, model_label,
                    diagnostics=diagnostics, telemetry=dashboard,
                    view_mode=view_mode, reference_crop=runtime["reference_crop"],
                    guard_config=lock.config if guarded else None,
                )
            else:
                rendered = draw_frame(frame, detections, result, command, model_label, diagnostics)
            if args.record_video and writer is None:
                path = Path(args.record_video)
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (rendered.shape[1], rendered.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot write video: {path}")
            if writer is not None:
                writer.write(rendered)
            shown, runtime["display_scale"] = fit_display_frame(
                rendered, args.display_width, args.display_height)
            age_ms = 1000 * (time.monotonic() - packet.t_recv)
            if jetbot:
                # Drawn on the display copy only: recordings and dashboard stay unchanged.
                left, right = driver.last_wheels
                text = (f"{'ARMED' if armed else 'SAFE'} | link {'OK' if driver.link_ok else 'DOWN'} | "
                        f"age {age_ms:.0f} ms | L {left:+.2f} R {right:+.2f} | {gate}")
                color = (0, 0, 255) if armed else (0, 200, 0)
                cv2.rectangle(shown, (0, 0), (shown.shape[1], 26), (0, 0, 0), -1)
                cv2.putText(shown, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            cv2.imshow(window, shown)
            if robot_log is not None:
                wheels = driver.last_wheels if jetbot else ("", "")
                robot_log.writerow([
                    processed, f"{time.perf_counter() - last_time:.3f}", f"{age_ms:.1f}",
                    f"{1000 * (time.perf_counter() - frame_start):.1f}", result.state.value, gate,
                    int(armed), command.forward, command.turn, wheels[0], wheels[1],
                    int(driver.link_ok) if jetbot else ""])
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m") and jetbot:
                armed = not armed
                LOG.warning("MOTORS %s", "ARMED" if armed else "SAFE")
                dashboard.add_event(processed, "MOTORS ARMED" if armed else "MOTORS SAFE")
                if not armed:
                    driver.send(MotionCommand())
            if key in (9, ord("d")) and args.ui == "dashboard":
                view_mode = "research" if view_mode == "presentation" else "presentation"
                dashboard.add_event(processed, "VIEW " + view_mode.upper())
            elif key == ord("r"):
                lock.reset()
                runtime["reference_crop"] = None
                dashboard.add_event(processed, "RESET")
                runtime["session"] = runtime["gt_target_id"] = None
                runtime["pending_event"] = "RESET"
                armed = armed and not jetbot
                driver.send(MotionCommand())
            elif key == 32:
                lock.stop("Operator emergency stop (reset or click to reselect)")
                dashboard.add_event(processed, "MANUAL STOP")
                armed = armed and not jetbot
                driver.send(MotionCommand())
        elapsed = time.perf_counter() - last_time
        LOG.info("Frames: %d, mean pipeline FPS: %.1f, last command: %s",
                 processed, processed / elapsed if elapsed else 0, driver.last_command)
    finally:
        driver.close()  # Always request zero on normal exit / exception.
        if trace is not None:
            trace.close()
        source.close()
        if writer is not None:
            writer.release()
        if raw_writer is not None:
            raw_writer.release()
        if robot_log_file is not None:
            robot_log_file.close()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive JetBot follow demo (AI on PC; "
                                                 "--driver jetbot drives REAL motors)")
    parser.add_argument("--source", default="0",
                        help="Webcam index, local video path, http(s) MJPEG URL, or 'jetbot' "
                             "(stream URL built from --robot-config)")
    parser.add_argument("--driver", choices=("mock", "jetbot"), default="mock",
                        help="mock = no motors (default); jetbot = real robot over HTTP")
    parser.add_argument("--robot-config", default="configs/jetbot.yaml")
    parser.add_argument("--record-raw", default=None, metavar="PATH",
                        help="Record raw camera frames (no dashboard) for offline replay")
    parser.add_argument("--robot-log", default=None, metavar="PATH",
                        help="Per-frame robot diagnostics CSV")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", choices=("baseline", "v5", "v5_1"), default="v5")
    parser.add_argument("--ablation", choices=sorted(IdentityManagerV5.STRATEGIES),
                        default="adaptive_fusion")
    parser.add_argument("--device", default="cpu", help="e.g. cpu, cuda:0 (default CPU)")
    parser.add_argument("--reid-model", choices=tuple(REID_CHECKPOINTS), default=None,
                        help="V5/V5.1 only: choose a matching OSNet MSMT17 checkpoint; "
                             "omitted = use the configured baseline x0.5")
    parser.add_argument("--confidence", type=float, default=0.10,
                        help="YOLO confidence threshold; use >= tracker low threshold")
    parser.add_argument("--confirm-frames", type=int, default=5,
                        help="Clear confirmations required for a NEW track ID")
    parser.add_argument("--same-track-confirm-frames", type=int, default=2,
                        help="Clear confirmations to restore original trusted track")
    parser.add_argument("--guard-final-clear-frames", type=int, default=2,
                        help="Consecutive unambiguous frames required at completion")
    parser.add_argument("--guard-grace-frames", type=int, default=2,
                        help="Max uncertain frames allowed in pending confirmation")
    parser.add_argument("--max-lost-seconds", type=float, default=5.0)
    parser.add_argument("--min-target-confidence", type=float, default=0.35)
    parser.add_argument("--record-video", default=None, metavar="PATH")
    parser.add_argument("--ui", choices=("dashboard", "compact"), default="dashboard",
                        help="Dashboard visualization or original compact sidebar")
    parser.add_argument("--dashboard-view", choices=("presentation", "research"),
                        default="presentation", help="Initial dashboard view; TAB/D toggles live")
    parser.add_argument("--display-width", type=int, default=1600,
                        help="Limit displayed image width (recording stays full resolution)")
    parser.add_argument("--display-height", type=int, default=900,
                        help="Limit displayed image height (recording stays full resolution)")
    parser.add_argument("--identity-guard", choices=("on", "off"), default="on",
                        help="Frozen-reference Re-ID safety gate (default on for V5/V5.1)")
    parser.add_argument("--guard-track-distance", type=float, default=0.32)
    parser.add_argument("--guard-recovery-distance", type=float, default=0.26)
    parser.add_argument("--guard-same-track-recovery-distance", type=float, default=0.28)
    parser.add_argument("--guard-margin", type=float, default=0.08)
    parser.add_argument("--guard-overlap", type=float, default=0.35)
    parser.add_argument("--guard-min-quality", type=float, default=0.30)
    parser.add_argument("--trace-csv", default=None,
                        help="Write frame-wise safety/identity/controller diagnostics")
    parser.add_argument("--gt-sequence", default=None,
                        help="MOT17 sequence name/path for optional GT attribution")
    parser.add_argument("--gt-root", default="data/datasets/MOT17/train")
    parser.add_argument("--gt-min-visibility", type=float, default=0.20,
                        help="Minimum MOT17 GT visibility for evaluation/selection")
    parser.add_argument("--gt-frame-offset", type=int, default=0,
                        help="Shift video frame indices to GT indices (usually 0)")
    args = parser.parse_args()
    if args.mode == "baseline" and args.reid_model is not None:
        parser.error("--reid-model requires --mode v5 or v5_1")
    if args.driver == "jetbot" and args.gt_sequence:
        parser.error("--gt-sequence cannot be used with --driver jetbot")
    if args.gt_sequence and not args.trace_csv:
        parser.error("--gt-sequence requires --trace-csv")
    if args.display_width < 640 or args.display_height < 480:
        parser.error("--display-width >= 640 and --display-height >= 480 are required")
    if not 0 <= args.confidence <= 1 or args.max_lost_seconds <= 0:
        parser.error("--confidence must be in [0,1]; --max-lost-seconds must be > 0")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run(args)


if __name__ == "__main__":
    main()
