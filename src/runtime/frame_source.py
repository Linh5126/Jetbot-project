"""Unified frame sources: video file / image pattern, webcam, JetBot MJPEG stream."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import requests

LOG = logging.getLogger(__name__)
MAX_BUFFER_BYTES = 4_000_000


@dataclass(frozen=True)
class FramePacket:
    frame: np.ndarray
    seq: int          # increasing, starts at 1
    t_recv: float     # time.monotonic() when the PC finished receiving the frame


class FileSource:
    """Reads every frame in order (no dropping) so offline results stay reproducible."""
    live = False

    def __init__(self, raw: str) -> None:
        self._capture = cv2.VideoCapture(raw)
        if not self._capture.isOpened():
            raise RuntimeError(f"Cannot open video source: {raw}")
        self.nominal_fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self.ended = False
        self._seq = 0

    def read(self, timeout: float) -> FramePacket | None:
        if self.ended:
            return None
        ok, frame = self._capture.read()
        if not ok:
            self.ended = True
            return None
        self._seq += 1
        return FramePacket(frame, self._seq, time.monotonic())

    def close(self) -> None:
        self._capture.release()


class _LatestFrameSource:
    """Background thread keeps only the newest frame; read() never returns a frame twice."""
    live = True
    nominal_fps = None
    ended = False

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._seq = 0
        self._last_returned = 0
        self._packet: FramePacket | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        raise NotImplementedError

    def _publish(self, frame: np.ndarray) -> None:
        with self._cond:
            self._seq += 1
            self._packet = FramePacket(frame, self._seq, time.monotonic())
            self._cond.notify_all()

    def read(self, timeout: float) -> FramePacket | None:
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > self._last_returned, timeout):
                return None
            self._last_returned = self._seq
            return self._packet

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)


class LatestCaptureSource(_LatestFrameSource):
    def __init__(self, index: int, capture=None) -> None:
        super().__init__()
        self._capture = capture if capture is not None else cv2.VideoCapture(index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Cannot open video source: {index}")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if ok:
                self._publish(frame)
            else:
                self._stop.wait(0.01)

    def close(self) -> None:
        super().close()
        self._capture.release()


def extract_jpegs(buf: bytearray) -> list[bytes]:
    """Pop every complete JPEG (FFD8..FFD9) from buf; keep a partial tail for the next chunk."""
    out = []
    while True:
        start = buf.find(b"\xff\xd8")
        if start < 0:
            del buf[:-1]          # keep 1 byte in case a marker was split across chunks
            return out
        end = buf.find(b"\xff\xd9", start + 2)
        if end < 0:
            del buf[:start]
            return out
        out.append(bytes(buf[start:end + 2]))
        del buf[:end + 2]


class MjpegHttpSource(_LatestFrameSource):
    """Parses multipart MJPEG with requests (no dependence on OpenCV's FFMPEG backend)."""

    def __init__(self, url: str, connect_timeout: float = 3.0, read_timeout: float = 2.0) -> None:
        super().__init__()
        self.url = url
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with requests.get(self.url, stream=True,
                                  timeout=(self.connect_timeout, self.read_timeout)) as resp:
                    resp.raise_for_status()
                    buf = bytearray()
                    for chunk in resp.iter_content(chunk_size=8192):
                        if self._stop.is_set():
                            return
                        buf += chunk
                        for jpg in extract_jpegs(buf):
                            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                self._publish(frame)
                        if len(buf) > MAX_BUFFER_BYTES:
                            buf.clear()
            except requests.RequestException as exc:
                LOG.warning("Stream error (%s); retrying in 1 s", exc)
            self._stop.wait(1.0)


def open_source(raw: str, stream_cfg: dict | None = None):
    if raw.isdecimal():
        return LatestCaptureSource(int(raw))
    if raw.startswith(("http://", "https://")):
        cfg = stream_cfg or {}
        return MjpegHttpSource(raw, float(cfg.get("connect_timeout_s", 3.0)),
                               float(cfg.get("read_timeout_s", 2.0)))
    return FileSource(raw)


def build_stream_url(robot_cfg: dict, stream_cfg: dict) -> str:
    return (f"http://{robot_cfg['host']}:{robot_cfg['port']}/stream"
            f"?w={stream_cfg['width']}&h={stream_cfg['height']}&fps={stream_cfg['fps']}"
            f"&q={stream_cfg['quality']}&flip={stream_cfg['flip']}")
