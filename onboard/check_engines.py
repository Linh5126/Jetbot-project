"""On the Nano: compare the TensorRT engines with the PC reference (models/onboard/parity.npz).

    python3 -m onboard.check_engines
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from onboard.models import OsnetTrt, YoloTrt  # noqa: E402

MODELS = ROOT / "models" / "onboard"


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def timed(fn, *args):
    fn(*args)  # warm-up
    start = time.time()
    for _ in range(20):
        out = fn(*args)
    return out, 1000 * (time.time() - start) / 20


def main():
    ref = np.load(str(MODELS / "parity.npz"))
    frame = ref["frame"]
    yolo = YoloTrt(str(MODELS / "yolov8n_pose_288x480_fp16.engine"), conf=0.25)
    (boxes, _, kpts), yolo_ms = timed(yolo, frame)
    matched, kpt_err = 0, 0.0
    for b, k in zip(ref["boxes"], ref["kpts"]):
        overlaps = [iou(b, x) for x in boxes]
        if overlaps and max(overlaps) > 0.9:
            matched += 1
            mine, seen = kpts[int(np.argmax(overlaps))], k[:, 2] > 0.5
            kpt_err = max(kpt_err, float(np.abs(mine[seen, :2] - k[seen, :2]).max()) if seen.any() else 0.0)
    print("YOLO-pose %.1f ms: %d boxes, %d/%d PC boxes matched (IoU > 0.9), keypoint error max %.1f px"
          % (yolo_ms, len(boxes), matched, len(ref["boxes"]), kpt_err))
    reid = OsnetTrt(str(MODELS / "osnet_x0_5_fp16.engine"))
    crops = [frame[int(y1):int(y2), int(x1):int(x2)] for x1, y1, x2, y2 in ref["boxes"]]
    feats, reid_ms = timed(reid, crops)
    cos = (feats * ref["feats"]).sum(axis=1)
    print("OSNet %.1f ms for %d crops: cosine to PC min %.4f (distance %.4f)"
          % (reid_ms, len(crops), cos.min(), 1 - cos.min()))
    ok = matched == len(ref["boxes"]) and kpt_err < 3.0 and cos.min() > 0.99  # FP16: a few px
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
