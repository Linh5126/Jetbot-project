"""V5/V5.1 research ablations with shared detector and Re-ID inference."""
from __future__ import annotations

import csv
import json
import logging
import time
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from src.identity.manager_v5 import IdentityManagerV5
from src.identity.manager_v51 import IdentityManagerV51
from src.evaluation.metrics import evaluate_mot, evaluate_mot_collection
from src.evaluation.target_metrics import evaluate_target_centric, aggregate_target_centric
from src.reid.osnet_embedder import OSNetEmbedder
from src.reid.crop_quality import score_person_crop_quality
from src.reid.model_config import checkpoint_hint
from src.runtime.person_pipeline import extract_tracks
from .benchmark_common import get_fps, sha256_file, write_mot_row

LOG = logging.getLogger("jetbot-mot17")


def _macro_metrics(sequence_summaries: list[dict]) -> dict:
    metric_names = ["idf1", "mota", "motp", "num_switches", "num_false_positives", "num_misses"]
    result = {}
    for side in ("identity_manager_ids", "bytetrack_raw_ids"):
        values = {key: [] for key in metric_names}
        for item in sequence_summaries:
            metrics = item.get("metrics", {}).get(side, {})
            for key in metric_names:
                value = metrics.get(key)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    values[key].append(float(value))
        result[side] = {
            key: (float(np.mean(items)) if items else None)
            for key, items in values.items()
        }
    return result



def run_v5_ablation_sequence(
    seq_dir: Path,
    cfg: dict,
    output_root: Path,
    strategies: list[str],
    max_frames: int | None = None,
    save_video: bool = False,
    experiment_name: str = "v5",
    quality_aware: bool = False,
) -> dict:
    """Run requested V5/V5.1 ablations on one shared detector/tracker pass.

    Every strategy receives exactly the same ByteTrack detections and Re-ID
    embeddings in each frame, controlling inference variability in the ablation.
    """
    seq_name = seq_dir.name
    image_dir = seq_dir / "img1"
    frame_paths = sorted(
        list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")),
        key=lambda p: int(p.stem),
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

    detector_cfg = cfg["detector"]
    reid_cfg = cfg["reid"]["v5"]
    checkpoint_path = Path(reid_cfg["checkpoint"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size < 500_000:
        raise FileNotFoundError(
            f"Thiếu checkpoint Person Re-ID: {checkpoint_path}. "
            "Chạy: " + checkpoint_hint(str(reid_cfg["model_name"]))
        )
    embedder = OSNetEmbedder(
        model_name=reid_cfg["model_name"],
        model_path=checkpoint_path,
        require_checkpoint=True,
        device=reid_cfg.get("device", cfg["reid"].get("device", "cuda")),
        image_size=tuple(reid_cfg.get("image_size", [256, 128])),
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    import torch
    runtime_info = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Thiếu Ultralytics. Cài bằng: python -m pip install ultralytics") from exc
    model = YOLO(detector_cfg["weights"])
    tracker_yaml = str(Path(detector_cfg["tracker"]).resolve())

    identity_cfg = dict(cfg["identity"]["v5"])
    if quality_aware:
        manager_cfg = dict(identity_cfg)
        v51_options = dict(cfg.get("identity", {}).get("v5_1", {}))
        for key in ("target_min_visibility", "target_lost_gap_frames", "target_max_reacquisition_frames"):
            v51_options.pop(key, None)
        manager_cfg.update(v51_options)
        manager_class = IdentityManagerV51
    else:
        manager_cfg = identity_cfg
        manager_class = IdentityManagerV5
    managers = {
        strategy: manager_class(fps=fps, strategy=strategy, **manager_cfg)
        for strategy in strategies
    }
    output_dirs = {
        strategy: output_root / experiment_name / strategy / seq_name
        for strategy in strategies
    }
    for out_dir in output_dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        handles = {}
        for strategy, out_dir in output_dirs.items():
            raw_f = stack.enter_context((out_dir / "track_results.txt").open("w", newline="", encoding="utf-8"))
            target_f = stack.enter_context((out_dir / "target_results.txt").open("w", newline="", encoding="utf-8"))
            log_f = stack.enter_context((out_dir / "identity_log.csv").open("w", newline="", encoding="utf-8"))
            handles[strategy] = {
                "raw_writer": csv.writer(raw_f),
                "target_writer": csv.writer(target_f),
                "log_writer": csv.DictWriter(log_f, fieldnames=[
                    "frame", "track_id", "target_id", "association", "status",
                    "confidence", "x1", "y1", "x2", "y2"
                ] + (["crop_quality"] if quality_aware else [])),
                "video": None,
            }
            handles[strategy]["log_writer"].writeheader()
            if save_video:
                video_path = out_dir / "annotated.mp4"
                writer = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Không tạo được video: {video_path}")
                handles[strategy]["video"] = writer

        started = time.perf_counter()
        processed = 0
        last_processed_frame = None
        for path in frame_paths:
            frame_id = int(path.stem)
            frame = cv2.imread(str(path))
            if frame is None:
                LOG.warning("Bỏ qua frame không đọc được: %s", path)
                continue
            last_processed_frame = frame_id

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
            crop_quality_by_track = {}
            if detections:
                crops, crop_track_ids, crop_detections = [], [], []
                interval = max(1, int(reid_cfg.get("embedding_interval", 3)))
                min_h = float(manager_cfg.get("min_box_height", 28))
                min_conf = float(manager_cfg.get("min_embedding_confidence", 0.35))
                for det in detections:
                    track_id = det["track_id"]
                    should_embed = (
                        frame_id % interval == 0
                        or any(not manager.has_track(track_id) for manager in managers.values())
                    )
                    if not should_embed:
                        continue
                    x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    if (y2 - y1) < min_h or float(det["confidence"]) < min_conf:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        crops.append(crop)
                        crop_track_ids.append(track_id)
                        crop_detections.append(det)
                if crops:
                    feats = embedder(crops)
                    for crop, det, tid, feat in zip(crops, crop_detections, crop_track_ids, feats):
                        embeddings[tid] = feat
                        if quality_aware:
                            quality = score_person_crop_quality(
                                crop, det["bbox"], frame.shape, det.get("confidence", 1.0)
                            )
                            crop_quality_by_track[int(tid)] = quality
                if quality_aware:
                    for det in detections:
                        quality = crop_quality_by_track.get(int(det["track_id"]))
                        if quality is not None:
                            det["crop_quality"] = quality["score"]

            for strategy, manager in managers.items():
                assignments = manager.update(frame_id, detections, embeddings)
                item = handles[strategy]
                annotated = frame.copy() if item["video"] is not None else None
                for det in detections:
                    track_id = int(det["track_id"])
                    target_id = assignments[track_id]
                    bbox = det["bbox"]
                    conf = float(det["confidence"])
                    write_mot_row(item["raw_writer"], frame_id, track_id, bbox, conf)
                    write_mot_row(item["target_writer"], frame_id, target_id, bbox, conf)
                    if annotated is not None:
                        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (50, 210, 70), 2)
                        label = f"track:{track_id} target:{target_id}"
                        cv2.putText(annotated, label, (x1, max(20, y1 - 7)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 210, 70), 2)
                    item["log_writer"].writerow({
                        "frame": frame_id,
                        "track_id": track_id,
                        "target_id": target_id,
                        "association": manager.association_of(track_id),
                        "status": manager.status_of(target_id),
                        "confidence": f"{conf:.4f}",
                        "x1": f"{bbox[0]:.2f}", "y1": f"{bbox[1]:.2f}",
                        "x2": f"{bbox[2]:.2f}", "y2": f"{bbox[3]:.2f}",
                        **({"crop_quality": f"{float(det.get('crop_quality', 0.0)):.4f}"} if quality_aware else {}),
                    })
                if annotated is not None:
                    cv2.putText(annotated, f"{seq_name} | {experiment_name.upper()} {strategy} | frame {frame_id}",
                                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    item["video"].write(annotated)

            processed += 1
            if processed % 200 == 0:
                LOG.info("%s/%s: %d/%d frames (shared inference)", experiment_name, seq_name, processed, len(frame_paths))

        elapsed = time.perf_counter() - started
        for item in handles.values():
            if item["video"] is not None:
                item["video"].release()

    summaries = {}
    gt_file = seq_dir / "gt" / "gt.txt"
    for strategy, manager in managers.items():
        out_dir = output_dirs[strategy]
        metric_result = {}
        if gt_file.exists():
            metric_result = evaluate_mot(
                gt_file,
                out_dir / "target_results.txt",
                out_dir / "track_results.txt",
                max_frame=last_processed_frame,
            )
        target_metrics = None
        if gt_file.exists():
            target_metrics = evaluate_target_centric(
                gt_file,
                out_dir / "identity_log.csv",
                iou_threshold=0.5,
                min_visibility=float(cfg.get("identity", {}).get("v5_1", {}).get("target_min_visibility", 0.0)),
                lost_gap_frames=int(cfg.get("identity", {}).get("v5_1", {}).get("target_lost_gap_frames", 5)),
                max_reacquisition_frames=int(cfg.get("identity", {}).get("v5_1", {}).get("target_max_reacquisition_frames", 90)),
                max_frame=last_processed_frame,
                include_target_details=True,
            )
        summary = {
            "sequence": seq_name,
            "mode": experiment_name,
            "ablation": strategy,
            "frames_processed": processed,
            "last_frame_id": last_processed_frame,
            "elapsed_seconds_shared": round(elapsed, 3),
            "shared_pipeline_fps": round(processed / elapsed, 3) if elapsed > 0 else None,
            "fps_scope": "All selected ablation managers share one detector/tracker/Re-ID inference pass.",
            "video": str(out_dir / "annotated.mp4") if save_video else None,
            "metrics": metric_result,
            "identity_diagnostics": manager.diagnostics(),
            "detector_config": {
                "weights": detector_cfg["weights"],
                "confidence": detector_cfg["confidence"],
                "iou": detector_cfg["iou"],
                "image_size": detector_cfg["image_size"],
                "device": detector_cfg["device"],
                "tracker": detector_cfg["tracker"],
            },
            "reid": {
                "model_name": reid_cfg["model_name"],
                "checkpoint": str(checkpoint_path),
                "checkpoint_type": "Person Re-ID trained on MSMT17; cross-domain Torchreid Model Zoo checkpoint",
                "checkpoint_sha256": checkpoint_sha256,
                "runtime": runtime_info,
            },
        }
        summary["target_centric_metrics"] = target_metrics
        summary["crop_quality_enabled"] = bool(quality_aware)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summaries[strategy] = summary

    return summaries



def run_v5_experiment(
    sequences: list[Path],
    cfg: dict,
    output_root: Path,
    strategies: list[str],
    max_frames: int | None = None,
    save_video: bool = False,
    experiment_name: str = "v5",
    quality_aware: bool = False,
) -> dict:
    per_strategy = {strategy: [] for strategy in strategies}
    for seq_dir in sequences:
        sequence_result = run_v5_ablation_sequence(
            seq_dir, cfg, output_root, strategies, max_frames=max_frames, save_video=save_video,
            experiment_name=experiment_name, quality_aware=quality_aware,
        )
        for strategy in strategies:
            per_strategy[strategy].append(sequence_result[strategy])

    aggregate_by_strategy = {}
    for strategy in strategies:
        records = []
        sequence_rows = per_strategy[strategy]
        for item in sequence_rows:
            seq_dir = next(s for s in sequences if s.name == item["sequence"])
            max_frame = item.get("last_frame_id")
            out_dir = output_root / experiment_name / strategy / item["sequence"]
            records.append({
                "sequence": item["sequence"],
                "gt_path": seq_dir / "gt" / "gt.txt",
                "target_result_path": out_dir / "target_results.txt",
                "raw_result_path": out_dir / "track_results.txt",
                "max_frame": max_frame,
            })
        aggregate_by_strategy[strategy] = {
            "macro_mean": _macro_metrics(sequence_rows),
            "pooled_metrics": evaluate_mot_collection(records),
            "target_centric_pooled": aggregate_target_centric([
                item.get("target_centric_metrics") or {} for item in sequence_rows
            ]),
        }

    result = {
        "mode": experiment_name,
        "ablation_protocol": (
            "All ablations receive identical per-frame ByteTrack detections and OSNet embeddings; "
            + ("V5.1 adds crop-quality scoring and quality/diversity-aware gallery admission. " if quality_aware else "")
            + "Only the identity-association strategy changes between ablations."
        ),
        "strategies": {
            strategy: {
                "sequences": per_strategy[strategy],
                **aggregate_by_strategy[strategy],
            }
            for strategy in strategies
        },
    }
    result["target_centric_protocol"] = (
        "Protocol v2, all V5/V5.1 ablations: simulate GT target selection at first IoU match; "
        "report mismatch separately from locked-ID wrong-person risk; right-censor terminal losses."
    )
    mode_dir = output_root / experiment_name
    mode_dir.mkdir(parents=True, exist_ok=True)
    (mode_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    csv_path = mode_dir / "ablation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        base_header = ["strategy", "sequence", "idf1", "mota", "motp", "num_switches", "fps_shared"]
        target_header = ["target_identity_retention", "tracking_coverage", "locked_id_wrong_person_rate", "reacquisition_success_rate"]
        writer.writerow(base_header + target_header)
        for strategy in strategies:
            for item in per_strategy[strategy]:
                identity = item.get("metrics", {}).get("identity_manager_ids", {})
                row = [
                    strategy, item["sequence"], identity.get("idf1"), identity.get("mota"),
                    identity.get("motp"), identity.get("num_switches"), item.get("shared_pipeline_fps"),
                ]
                target = item.get("target_centric_metrics") or {}
                row.extend([
                    target.get("target_identity_retention_rate"), target.get("tracking_coverage"),
                    target.get("locked_id_wrong_person_rate"), target.get("reacquisition_success_rate"),
                ])
                writer.writerow(row)
            pooled = aggregate_by_strategy[strategy]["pooled_metrics"].get("identity_manager_ids", {})
            pooled_row = [strategy, "POOLED", pooled.get("idf1"), pooled.get("mota"), pooled.get("motp"),
                          pooled.get("num_switches"), ""]
            target_pooled = aggregate_by_strategy[strategy].get("target_centric_pooled", {})
            pooled_row.extend([
                target_pooled.get("target_identity_retention_rate"), target_pooled.get("tracking_coverage"),
                target_pooled.get("locked_id_wrong_person_rate"), target_pooled.get("reacquisition_success_rate"),
            ])
            writer.writerow(pooled_row)
    LOG.info("Đã lưu kết quả ablation %s: %s", experiment_name.upper(), mode_dir / "run_summary.json")
    return result

