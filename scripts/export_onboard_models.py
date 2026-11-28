"""Export the PC models to ONNX for TensorRT 7.1 on the Jetson Nano (JetPack 4.5).

Run on the PC (needs ultralytics, torchreid, onnx):
    .venv/Scripts/python scripts/export_onboard_models.py

Writes models/onboard/:
    yolov8n_pose_288x480.onnx  static 1x3x288x480 (the 480x270 stream letterboxed), opset 11,
                               output 1x56xN: box, person score, 17 COCO keypoints (x, y, conf)
    osnet_x0_5.onnx        dynamic batch Nx3x256x128 -> N x 512, opset 11
    parity.npz             PC reference outputs to check the Nano engines (onboard/check_engines.py)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "models" / "onboard"
YOLO_HW = (288, 480)
YOLO_WEIGHTS = "yolov8n-pose.pt"  # keypoints let a person claim the robot by raising both hands
REID_HW = (256, 128)
OPSET = 11  # TensorRT 7.1.3's ONNX parser


def export_yolo() -> Path:
    from ultralytics import YOLO
    onnx = Path(YOLO(str(ROOT / YOLO_WEIGHTS)).export(
        format="onnx", imgsz=list(YOLO_HW), opset=OPSET, simplify=True, dynamic=False))
    target = OUT / "yolov8n_pose_288x480.onnx"
    shutil.move(str(onnx), target)
    return target


def build_osnet():
    from torchreid.models import build_model
    from torchreid.utils import load_pretrained_weights
    model = build_model("osnet_x0_5", num_classes=1, pretrained=False)
    load_pretrained_weights(model, str(ROOT / "checkpoints" / "osnet_x0_5_msmt17.pth.tar"))
    return model.eval()


def export_osnet(model) -> Path:
    target = OUT / "osnet_x0_5.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, *REID_HW), str(target), opset_version=OPSET,
                      input_names=["input"], output_names=["feat"],
                      dynamic_axes={"input": {0: "n"}, "feat": {0: "n"}}, dynamo=False)
    return target


def parity(model) -> None:
    """Same preprocessing as onboard/models.py; PC outputs become the Nano's expected values."""
    from ultralytics import YOLO
    from onboard.models import letterbox, reid_batch
    image_path = ROOT / ".venv" / "Lib" / "site-packages" / "ultralytics" / "assets" / "bus.jpg"
    frame = cv2.resize(cv2.imread(str(image_path)), (480, 270))
    with torch.no_grad():
        pred = YOLO(str(ROOT / YOLO_WEIGHTS)).predict(frame, imgsz=list(YOLO_HW), conf=0.25,
                                                      verbose=False)[0]
        boxes = pred.boxes.xyxy.cpu().numpy().astype(np.float32)
        kpts = pred.keypoints.data.cpu().numpy().astype(np.float32)  # N x 17 x (x, y, conf)
        crops = [frame[int(y1):int(y2), int(x1):int(x2)] for x1, y1, x2, y2 in boxes]
        feats = model(torch.from_numpy(reid_batch(crops))).numpy()
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    np.savez(OUT / "parity.npz", frame=frame, letterboxed=letterbox(frame, YOLO_HW)[0],
             boxes=boxes, kpts=kpts, feats=feats.astype(np.float32))
    print(f"parity: {len(boxes)} people on bus.jpg")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("YOLO ->", export_yolo())
    model = build_osnet()
    print("OSNet ->", export_osnet(model))
    parity(model)


if __name__ == "__main__":
    main()
