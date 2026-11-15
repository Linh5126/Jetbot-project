"""Historical Baseline/V1-V4 MOT17 benchmark; retained for research comparison."""
from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import cv2

from src.identity.manager import IdentityManager
from src.identity.manager_v4 import IdentityManagerV4
from src.evaluation.metrics import evaluate_mot
from src.reid.osnet_embedder import OSNetEmbedder
from src.runtime.person_pipeline import extract_tracks
from .benchmark_common import get_fps, write_mot_row

LOG = logging.getLogger("jetbot-mot17")


def run_sequence(seq_dir: Path, mode: str, cfg: dict, output_root: Path,
                 max_frames: int | None = None, save_video: bool = False) -> dict:
    seq_name = seq_dir.name
    image_dir = seq_dir / "img1"
    frame_paths = sorted(
        list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")),
        key=lambda p: int(p.stem)
    )
    if not frame_paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh trong: {image_dir}")
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Không đọc được ảnh: {frame_paths[0]}")
    height, width = first.shape[:2]
    fps = get_fps(seq_dir, cfg["dataset"]["fps_fallback"])

    out_dir = output_root / mode / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "annotated.mp4"
    video = None
    if save_video:
        video = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not video.isOpened():
            raise RuntimeError(f"Không tạo được video: {video_path}")

    detector_cfg = cfg["detector"]
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu Ultralytics. Cài bằng: python -m pip install ultralytics"
        ) from exc
    model = YOLO(detector_cfg["weights"])
    tracker_yaml = str(Path(detector_cfg["tracker"]).resolve())

    identity_cfg = cfg["identity"]
    if mode == "v4":
        v4_cfg = dict(identity_cfg["v4"])
        manager = IdentityManagerV4(fps=fps, **v4_cfg)
    else:
        manager = IdentityManager(
            mode=mode,
            fps=fps,
            max_lost_seconds=identity_cfg["max_lost_seconds"][mode],
            max_center_distance=identity_cfg["max_center_distance"],
            max_height_ratio=identity_cfg["max_height_ratio"],
            appearance_threshold=identity_cfg["appearance_threshold"],
            motion_weight=identity_cfg["motion_weight"],
            appearance_weight=identity_cfg["appearance_weight"],
            max_motion_speed=identity_cfg["max_motion_speed"],
        )

    embedder = None
    if mode in ("v2", "v3", "v4"):
        embedder = OSNetEmbedder(
            model_name=cfg["reid"]["model_name"],
            device=cfg["reid"]["device"],
            image_size=tuple(cfg["reid"]["image_size"]),
        )

    with (
        (out_dir / "track_results.txt").open("w", newline="", encoding="utf-8") as raw_f,
        (out_dir / "target_results.txt").open("w", newline="", encoding="utf-8") as target_f,
        (out_dir / "identity_log.csv").open("w", newline="", encoding="utf-8") as log_f,
    ):
        raw_writer = csv.writer(raw_f)
        target_writer = csv.writer(target_f)
        log_writer = csv.DictWriter(log_f, fieldnames=[
            "frame", "track_id", "target_id", "association", "status",
            "confidence", "x1", "y1", "x2", "y2"
        ])
        log_writer.writeheader()

        started = time.perf_counter()
        processed = 0
        for path in frame_paths:
            frame_id = int(path.stem)
            frame = cv2.imread(str(path))
            if frame is None:
                LOG.warning("Bỏ qua frame không đọc được: %s", path)
                continue

            result = model.track(
                source=frame,
                persist=True,
                tracker=tracker_yaml,
                classes=detector_cfg["classes"],
                conf=float(detector_cfg["confidence"]),
                iou=float(detector_cfg["iou"]),
                imgsz=int(detector_cfg["image_size"]),
                device=detector_cfg["device"],
                verbose=False,
            )[0]
            detections = extract_tracks(result)

            embeddings = {}
            if embedder is not None and detections:
                crops, crop_track_ids = [], []
                interval = max(1, int(cfg["reid"]["embedding_interval"]))
                for det in detections:
                    track_id = det["track_id"]
                    # Always embed a new ByteTrack ID; refresh established tracks periodically.
                    if frame_id % interval == 0 or not manager.has_track(track_id):
                        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size:
                            crops.append(crop)
                            crop_track_ids.append(track_id)
                if crops:
                    feats = embedder(crops)
                    for tid, feat in zip(crop_track_ids, feats):
                        embeddings[tid] = feat

            assignments = manager.update(frame_id, detections, embeddings)
            for det in detections:
                track_id = det["track_id"]
                target_id = assignments[track_id]
                bbox = det["bbox"]
                conf = det["confidence"]
                write_mot_row(raw_writer, frame_id, track_id, bbox, conf)
                write_mot_row(target_writer, frame_id, target_id, bbox, conf)

                if video is not None:
                    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 210, 70), 2)
                    label = f"track:{track_id} target:{target_id}"
                    cv2.putText(frame, label, (x1, max(20, y1 - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 210, 70), 2)
                log_writer.writerow({
                    "frame": frame_id,
                    "track_id": track_id,
                    "target_id": target_id,
                    "association": manager.association_of(track_id),
                    "status": manager.status_of(target_id),
                    "confidence": f"{conf:.4f}",
                    "x1": f"{bbox[0]:.2f}", "y1": f"{bbox[1]:.2f}",
                    "x2": f"{bbox[2]:.2f}", "y2": f"{bbox[3]:.2f}",
                })

            if video is not None:
                cv2.putText(frame, f"{seq_name} | {mode.upper()} | frame {frame_id}",
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                video.write(frame)
            processed += 1
            if processed % 200 == 0:
                LOG.info("%s/%s: %d/%d frames", mode, seq_name, processed, len(frame_paths))

        elapsed = time.perf_counter() - started

    if video is not None:
        video.release()
    metric_result = {}
    gt_file = seq_dir / "gt" / "gt.txt"
    if gt_file.exists():
        metric_result = evaluate_mot(
            gt_file,
            out_dir / "target_results.txt",
            out_dir / "track_results.txt",
            max_frame=int(frame_paths[-1].stem),
        )

    summary = {
        "sequence": seq_name,
        "mode": mode,
        "frames_processed": processed,
        "last_frame_id": int(frame_paths[-1].stem),
        "elapsed_seconds": round(elapsed, 3),
        "fps": round(processed / elapsed, 3) if elapsed > 0 else None,
        "video": str(video_path) if save_video else None,
        "metrics": metric_result,
        "identity_diagnostics": manager.diagnostics() if hasattr(manager, "diagnostics") else None,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOG.info("Hoàn tất %s/%s: %d frames, %.2f FPS",
             mode, seq_name, processed, summary["fps"] or 0.0)
    return summary

