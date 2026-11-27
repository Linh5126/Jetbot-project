"""TensorRT YOLOv8n + OSNet for the Jetson Nano (Python 3.6, TensorRT 7.1, no pycuda).

Pre/post-processing is plain numpy/OpenCV, so the PC can import this module
(without TensorRT) to produce reference outputs and run the unit tests.
"""
import ctypes

import cv2
import numpy as np

REID_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
REID_STD = np.array([0.229, 0.224, 0.225], np.float32)


def letterbox(frame, size):
    """Centred letterbox, same as the PC detector input -> (image, scale, (pad_x, pad_y))."""
    h, w = frame.shape[:2]
    out_h, out_w = size
    scale = min(out_h / h, out_w / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    if (new_w, new_h) != (w, h):
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (out_w - new_w) / 2, (out_h - new_h) / 2
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    image = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT,
                               value=(114, 114, 114))
    return image, scale, (left, top)


def yolo_input(letterboxed):
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def yolo_decode(output, scale, pad, frame_shape, conf=0.10, iou=0.50):
    """Raw YOLOv8-pose head (1, 4+1+17*3, N) -> person boxes Nx4 xyxy, scores N, keypoints Nx17x3.

    Keypoints are (x, y, conf) in frame pixels, COCO-17 order. A detection-only
    head (1, 4+80, N) also works: class 0 is person and keypoints come back empty (Nx0x3).
    """
    pred = output.reshape(output.shape[-2], output.shape[-1])
    pose = pred.shape[0] == 56
    scores = pred[4]  # pose: the single person score; detect: class 0 = person
    keep = scores >= conf
    if not np.any(keep):
        return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                np.zeros((0, 17 if pose else 0, 3), np.float32))
    cx, cy, bw, bh = pred[:4, keep]
    scores = scores[keep]
    if pose:
        kpts = pred[5:, keep].T.reshape(-1, 17, 3).copy()
        kpts[:, :, 0] = (kpts[:, :, 0] - pad[0]) / scale
        kpts[:, :, 1] = (kpts[:, :, 1] - pad[1]) / scale
    else:
        kpts = np.zeros((len(scores), 0, 3), np.float32)
    x1 = (cx - bw / 2 - pad[0]) / scale
    y1 = (cy - bh / 2 - pad[1]) / scale
    boxes = np.stack([x1, y1, x1 + bw / scale, y1 + bh / scale], axis=1)
    h, w = frame_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
    idx = cv2.dnn.NMSBoxes(xywh, [float(s) for s in scores], conf, iou)
    idx = np.array(idx, dtype=np.int64).reshape(-1)
    return boxes[idx].astype(np.float32), scores[idx].astype(np.float32), kpts[idx].astype(np.float32)


def reid_batch(crops, size=(256, 128)):
    """Same preprocessing as the PC embedder (src/reid/osnet_embedder.py): RGB, resize, ImageNet normalise."""
    out = np.zeros((len(crops), 3, size[0], size[1]), np.float32)
    for i, crop in enumerate(crops):
        if crop is None or crop.size == 0:
            rgb = np.zeros((size[0], size[1], 3), np.uint8)
        else:
            rgb = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (size[1], size[0]))
        out[i] = ((rgb.astype(np.float32) / 255.0 - REID_MEAN) / REID_STD).transpose(2, 0, 1)
    return out


class _Cuda(object):
    """The few CUDA runtime calls TensorRT needs, via ctypes (pycuda is not installed)."""

    def __init__(self):
        self.lib = ctypes.CDLL("libcudart.so")
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]

    def _check(self, code, what):
        if code != 0:
            raise RuntimeError("%s failed with CUDA error %d" % (what, code))

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self._check(self.lib.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
        return ptr.value

    def to_device(self, dst, array):
        self._check(self.lib.cudaMemcpy(dst, array.ctypes.data, array.nbytes, 1), "cudaMemcpy H2D")

    def to_host(self, array, src):
        self._check(self.lib.cudaMemcpy(array.ctypes.data, src, array.nbytes, 2), "cudaMemcpy D2H")

    def free(self, ptr):
        self.lib.cudaFree(ptr)


class TrtEngine(object):
    """One input, one output engine; the input may have a dynamic batch dimension."""

    def __init__(self, path):
        import tensorrt as trt
        self.cuda = _Cuda()
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Cannot load TensorRT engine %s (built by another TensorRT?)" % path)
        self.context = self.engine.create_execution_context()
        if self.engine.num_bindings != 2 or not self.engine.binding_is_input(0):
            raise RuntimeError("Expected one input then one output binding in %s" % path)
        shape = tuple(self.engine.get_binding_shape(0))
        self.dynamic = -1 in shape
        if self.dynamic:
            shape = tuple(self.engine.get_profile_shape(0, 0)[2])  # (min, opt, max) -> max
            self.context.set_binding_shape(0, shape)
        self.max_input = shape
        self.max_batch = shape[0]
        self.max_output = tuple(self.context.get_binding_shape(1))
        self.buffers = [self.cuda.malloc(int(np.prod(s)) * 4) for s in (self.max_input, self.max_output)]

    def __call__(self, batch):
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        if batch.shape[1:] != self.max_input[1:] or not 1 <= batch.shape[0] <= self.max_batch:
            raise ValueError("Input %s does not fit engine %s" % (batch.shape, self.max_input))
        if self.dynamic:
            self.context.set_binding_shape(0, batch.shape)
        out = np.empty(tuple(self.context.get_binding_shape(1)), np.float32)
        self.cuda.to_device(self.buffers[0], batch)
        if not self.context.execute_v2(self.buffers):
            raise RuntimeError("TensorRT execute_v2 failed")
        self.cuda.to_host(out, self.buffers[1])
        return out

    def close(self):
        for ptr in self.buffers:
            self.cuda.free(ptr)
        self.buffers = []


class YoloTrt(object):
    def __init__(self, path, conf=0.10, iou=0.50):
        self.engine = TrtEngine(path)
        self.size = tuple(self.engine.max_input[2:])  # (h, w)
        self.conf, self.iou = conf, iou

    def __call__(self, frame):
        image, scale, pad = letterbox(frame, self.size)
        output = self.engine(yolo_input(image))
        return yolo_decode(output, scale, pad, frame.shape, self.conf, self.iou)


class OsnetTrt(object):
    def __init__(self, path):
        self.engine = TrtEngine(path)
        self.size = tuple(self.engine.max_input[2:])

    def __call__(self, crops):
        if not crops:
            return np.zeros((0, 512), np.float32)
        batch = reid_batch(crops, self.size)
        n = self.engine.max_batch
        feats = np.concatenate([self.engine(batch[i:i + n]) for i in range(0, len(batch), n)])
        return feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)


class CollisionTrt(object):
    """Camera free/blocked classifier (scripts/train_collision.py) -> P(blocked)."""

    def __init__(self, path):
        self.engine = TrtEngine(path)
        self.size = tuple(self.engine.max_input[2:])

    def __call__(self, frame):
        logits = self.engine(reid_batch([frame], self.size))[0]  # same RGB + ImageNet normalise
        e = np.exp(logits - logits.max())
        return float(e[0] / e.sum())  # ImageFolder class order: blocked, free
