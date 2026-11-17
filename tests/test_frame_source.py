from __future__ import annotations

import time

import cv2
import numpy as np

from src.runtime import frame_source as fs


def _jpeg(value: int) -> bytes:
    ok, enc = cv2.imencode(".jpg", np.full((16, 16, 3), value, np.uint8))
    assert ok
    return enc.tobytes()


def test_extract_jpegs_with_garbage_and_split_chunks():
    a, b = _jpeg(40), _jpeg(200)
    stream = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + a + b"\r\n--frame\r\n\r\n" + b + b"\r\n"
    cut = len(stream) - len(b) // 2  # split the second image mid-chunk
    buf, found = bytearray(), []
    for chunk in (stream[:7], stream[7:cut], stream[cut:]):
        buf += chunk
        found += fs.extract_jpegs(buf)
    assert found == [a, b]
    for jpg, value in zip(found, (40, 200)):
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and abs(int(img.mean()) - value) < 5


class FakeCapture:
    def __init__(self, n: int) -> None:
        self.frames = [np.full((4, 4, 3), i, np.uint8) for i in range(1, n + 1)]
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def get(self, prop):
        return 0.0

    def release(self):
        self.released = True


def test_latest_capture_returns_newest_and_never_repeats():
    capture = FakeCapture(3)
    src = fs.LatestCaptureSource(0, capture=capture)
    try:
        deadline = time.monotonic() + 2
        while src._seq < 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        packet = src.read(1.0)
        assert packet.seq == 3 and int(packet.frame[0, 0, 0]) == 3
        assert src.read(0.05) is None
        assert src.live and not src.ended
    finally:
        src.close()
    assert capture.released


def test_file_source_reads_all_frames_in_order(tmp_path):
    path = tmp_path / "five.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    for i in range(5):
        writer.write(np.full((32, 32, 3), 40 * i + 20, np.uint8))
    writer.release()
    src = fs.FileSource(str(path))
    packets = [src.read(1.0) for _ in range(5)]
    assert [p.seq for p in packets] == [1, 2, 3, 4, 5]
    means = [int(p.frame.mean()) for p in packets]
    assert means == sorted(means)
    assert src.read(1.0) is None and src.ended is True
    assert not src.live and src.nominal_fps == 10
    src.close()


def test_open_source_picks_class(monkeypatch):
    monkeypatch.setattr(fs.cv2, "VideoCapture", lambda raw: FakeCapture(0))
    monkeypatch.setattr(fs.MjpegHttpSource, "_run", lambda self: None)
    sources = [fs.open_source("0"), fs.open_source("http://x/stream"), fs.open_source("a.mp4")]
    assert [type(s) for s in sources] == [fs.LatestCaptureSource, fs.MjpegHttpSource, fs.FileSource]
    for s in sources:
        s.close()


def test_build_stream_url():
    url = fs.build_stream_url({"host": "10.0.0.5", "port": 8100},
                              {"width": 640, "height": 480, "fps": 15, "quality": 80, "flip": 0})
    assert url == "http://10.0.0.5:8100/stream?w=640&h=480&fps=15&q=80&flip=0"
