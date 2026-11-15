"""YOLO/ByteTrack + optional OSNet + existing persistent identity managers."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.identity.manager_v5 import IdentityManagerV5
from src.identity.manager_v51 import IdentityManagerV51
from src.reid.crop_quality import score_person_crop_quality


def extract_tracks(result) -> list[dict]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0 or boxes.id is None:
        return []
    bboxes = boxes.xyxy.detach().cpu().numpy()
    tids = boxes.id.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy()
    return [
        {"track_id": int(tid), "bbox": [float(x) for x in box], "confidence": float(conf)}
        for tid, box, conf in zip(tids, bboxes, confidences)
        if box[2] > box[0] and box[3] > box[1]
    ]


class PersonIdentityPipeline:
    """Live inference adapter. 'baseline' is an explicit no-ReID dry run mode."""

    def __init__(self, cfg: dict, mode: str, strategy: str, fps: float,
                 device: str, detector_confidence: float, guard_features: bool = False) -> None:
        if mode not in {"baseline", "v5", "v5_1"}:
            raise ValueError("mode must be baseline, v5 or v5_1")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install project dependencies: pip install -r requirements.txt") from exc
        self.mode = mode
        self.device = device
        self.confidence = detector_confidence
        self.guard_features = bool(guard_features and mode != "baseline")
        self.detector_cfg = cfg["detector"]
        self.reid_cfg = cfg["reid"]["v5"]
        self.tracker_yaml = str(Path(self.detector_cfg["tracker"]).resolve())
        self.model = YOLO(self.detector_cfg["weights"])
        self.embedder = None
        if mode == "baseline":
            from src.identity.manager import IdentityManager
            self.manager = IdentityManager(mode="baseline", fps=fps)
        else:
            manager_cfg = dict(cfg["identity"]["v5"])
            if mode == "v5_1":
                extra = dict(cfg["identity"]["v5_1"])
                for key in ("target_min_visibility", "target_lost_gap_frames", "target_max_reacquisition_frames"):
                    extra.pop(key, None)
                manager_cfg.update(extra)
            manager_class = IdentityManagerV51 if mode == "v5_1" else IdentityManagerV5
            self.manager = manager_class(fps=fps, strategy=strategy, **manager_cfg)
            from src.reid.osnet_embedder import OSNetEmbedder
            self.embedder = OSNetEmbedder(
                model_name=self.reid_cfg["model_name"],
                model_path=self.reid_cfg["checkpoint"],
                require_checkpoint=True,
                device=device,
                image_size=tuple(self.reid_cfg.get("image_size", [256, 128])),
            )
        self.interval = max(1, int(self.reid_cfg.get("embedding_interval", 3)))

    def process(self, frame: np.ndarray, frame_id: int) -> list[dict]:
        height, width = frame.shape[:2]
        prediction = self.model.track(
            source=frame, persist=True, tracker=self.tracker_yaml,
            classes=self.detector_cfg["classes"], conf=self.confidence,
            iou=float(self.detector_cfg["iou"]),
            imgsz=int(self.detector_cfg["image_size"]), device=self.device,
            verbose=False,
        )[0]
        detections = extract_tracks(prediction)
        features = {}
        if self.embedder is not None:
            crops, selected = [], []
            for det in detections:
                tid = det["track_id"]
                if not getattr(self, "guard_features", False) and frame_id % self.interval and self.manager.has_track(tid):
                    continue
                x1, y1, x2, y2 = [int(round(x)) for x in det["bbox"]]
                x1, x2 = max(0, x1), min(width, x2)
                y1, y2 = max(0, y1), min(height, y2)
                if (y2-y1) < self.manager.min_box_height or det["confidence"] < self.manager.min_embedding_confidence:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crops.append(crop)
                selected.append(det)
                if self.mode == "v5_1" or getattr(self, "guard_features", False):
                    det["crop_quality"] = score_person_crop_quality(
                        crop, det["bbox"], frame.shape, det["confidence"]
                    )["score"]
            if crops:
                vectors = self.embedder(crops)
                for det, vector in zip(selected, vectors):
                    tid = det["track_id"]
                    # Fresh per-frame evidence for the guarded lock, *not* a
                    # continually refreshed/poisonable reference gallery.
                    if getattr(self, "guard_features", False):
                        det["embedding"] = np.asarray(vector, dtype=np.float32).copy()
                    # Preserve V5/V5.1 gallery-update cadence in live mode.
                    if frame_id % self.interval == 0 or not self.manager.has_track(tid) or not getattr(self, "guard_features", False):
                        features[tid] = vector
        ids = self.manager.update(frame_id, detections, features)
        for det in detections:
            track_id = det["track_id"]
            det["target_id"] = int(ids[track_id])
            det["association"] = self.manager.association_of(track_id)
        return detections
