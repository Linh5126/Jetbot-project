#!/usr/bin/env python3
"""JetBot agent: HTTP API on :8100 for the off-board follow pipeline.

Runs ON the JetBot (JetPack 4.x, Python 3.6+). Standard library + cv2 + numpy only.

  GET  /health                          {"ok": true}
  GET  /status                          {"motor": {"ok": ..}, "camera": {..}, "imu": null, ..}
  GET  /stream?w=640&h=480&fps=15&q=80&flip=0   multipart MJPEG (flip=1 -> rotate 180)
  POST /drive  {"left": l, "right": r}  wheel speeds in [-1, 1], capped at --max-wheel
  POST /stop                            motors 0

Watchdog: no /drive for --watchdog-s seconds -> motors 0.

  python3 agent.py                                  # real robot (jetbot lib or Adafruit MotorHAT)
  python3 agent.py --motor fake --camera video.mp4  # PC dry run
"""
import argparse
import json
import logging
import math
import signal
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

LOG = logging.getLogger("jetbot-agent")

# IMX219 1640x1232 mode keeps the full sensor field of view; override with --camera "<pipeline>".
CSI_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1640, height=1232, "
    "framerate=30/1 ! nvvidconv ! video/x-raw, width={w}, height={h}, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
)


# ---------------------------------------------------------------- motors
class FakeMotor:
    name = "fake"

    def __init__(self):
        self.history = []

    def set(self, left, right):
        self.history.append((time.monotonic(), left, right))


class JetbotLibMotor:
    """NVIDIA jetbot package (handles Adafruit MotorHAT and Waveshare boards)."""
    name = "jetbot"

    def __init__(self):
        from jetbot import Robot
        self._robot = Robot()

    def set(self, left, right):
        self._robot.set_motors(left, right)


class AdafruitMotor:
    """Direct Adafruit MotorHAT (I2C bus 1, left = M1, right = M2)."""
    name = "adafruit"

    def __init__(self, i2c_bus=1):
        from Adafruit_MotorHAT import Adafruit_MotorHAT
        self._hat = Adafruit_MotorHAT(i2c_bus=i2c_bus)
        self._cls = Adafruit_MotorHAT
        self._motors = (self._hat.getMotor(1), self._hat.getMotor(2))

    def set(self, left, right):
        for motor, value in zip(self._motors, (left, right)):
            motor.setSpeed(int(round(min(1.0, abs(value)) * 255)))
            if value > 0:
                motor.run(self._cls.FORWARD)
            elif value < 0:
                motor.run(self._cls.BACKWARD)
            else:
                motor.run(self._cls.RELEASE)


def open_motor(kind):
    """Returns (motor or None, error string). Never raises: a dead motor must not kill the agent."""
    order = {"auto": (JetbotLibMotor, AdafruitMotor), "jetbot": (JetbotLibMotor,),
             "adafruit": (AdafruitMotor,), "fake": (FakeMotor,)}[kind]
    errors = []
    for cls in order:
        try:
            motor = cls()
            motor.set(0.0, 0.0)
            return motor, ""
        except Exception as exc:  # ImportError, I2C OSError, ...
            errors.append("%s: %s" % (cls.name, exc))
    return None, "; ".join(errors)


class MotorController:
    """All motor writes go through one lock (HTTP threads + watchdog share the I2C bus)."""

    def __init__(self, motor, error, max_wheel, watchdog_s, clock=time.monotonic):
        self.motor = motor
        self.error = error
        self.max_wheel = max_wheel
        self.watchdog_s = watchdog_s
        self._clock = clock
        self._lock = threading.Lock()
        self.left = self.right = 0.0
        self.last_drive = None
        self.watchdog_trips = 0

    @property
    def ok(self):
        return self.motor is not None and not self.error

    def _write(self, left, right):
        try:
            self.motor.set(left, right)
            self.left, self.right = left, right
            self.error = ""
        except Exception as exc:
            self.error = "write failed: %s" % exc
            raise

    def drive(self, left, right):
        cap = self.max_wheel
        left, right = max(-cap, min(cap, left)), max(-cap, min(cap, right))
        with self._lock:
            self.last_drive = self._clock()
            self._write(left, right)
        return left, right

    def stop(self):
        if self.motor is None:
            return
        with self._lock:
            self._write(0.0, 0.0)

    def watchdog_tick(self):
        with self._lock:
            if self.motor is None or (self.left == 0.0 and self.right == 0.0):
                return
            if self.last_drive is None or self._clock() - self.last_drive > self.watchdog_s:
                self.watchdog_trips += 1
                LOG.warning("Watchdog: no /drive for %.1f s -> stop", self.watchdog_s)
                try:
                    self._write(0.0, 0.0)
                except Exception:
                    LOG.exception("Watchdog stop failed")


# ---------------------------------------------------------------- camera
class Camera:
    """Capture thread keeps only the newest frame. Failure never touches the motor path."""

    def __init__(self, spec, width, height):
        self.spec = spec
        self.width, self.height = width, height
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._t_frame = None
        self.error = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _open(self):
        if self.spec == "csi":
            return cv2.VideoCapture(CSI_PIPELINE.format(w=self.width, h=self.height), cv2.CAP_GSTREAMER), True
        if self.spec.isdigit():
            return cv2.VideoCapture(int(self.spec)), True
        if "!" in self.spec:
            return cv2.VideoCapture(self.spec, cv2.CAP_GSTREAMER), True
        return cv2.VideoCapture(self.spec), False  # video file: loop it at its own fps

    def _run(self):
        while not self._stop.is_set():
            cap, live = self._open()
            if not cap.isOpened():
                self.error = "cannot open camera %r" % self.spec
                LOG.warning("%s; retrying in 2 s", self.error)
                self._stop.wait(2.0)
                continue
            period = 0.0 if live else 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 15.0)
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                self.error = ""
                with self._cond:
                    self._frame, self._seq, self._t_frame = frame, self._seq + 1, time.monotonic()
                    self._cond.notify_all()
                if period:
                    time.sleep(period)
            cap.release()
            if live and not self._stop.is_set():
                self.error = "camera read failed"
                LOG.warning("%s; reopening", self.error)
                self._stop.wait(1.0)

    def wait_newer(self, seq, timeout):
        """Newest frame with seq > given seq, or (None, seq) after timeout."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > seq, timeout)
            if self._seq > seq:
                return self._frame, self._seq
            return None, seq

    def status(self):
        age = None if self._t_frame is None else time.monotonic() - self._t_frame
        return {"ok": age is not None and age < 1.0, "frame_age_s": age, "frames": self._seq,
                "source": self.spec, "error": self.error}

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)


# ---------------------------------------------------------------- HTTP
def _int_param(query, key, default, low, high):
    try:
        value = int(float(query.get(key, [default])[0]))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def parse_wheels(body):
    """Validate a /drive body. Returns (left, right) or raises ValueError (trust boundary)."""
    data = json.loads(body.decode("utf-8") or "{}")
    if not isinstance(data, dict):
        raise ValueError("body must be a JSON object")
    out = []
    for key in ("left", "right"):
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("%s must be a finite number" % key)
        out.append(float(value))
    return out[0], out[1]


class Handler(BaseHTTPRequestHandler):
    server_version = "JetBotAgent/1.0"
    # HTTP/1.0: one request per connection, no keep-alive bookkeeping needed.

    def log_message(self, fmt, *args):
        LOG.debug("%s " + fmt, self.address_string(), *args)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        app = self.server.app
        if url.path == "/health":
            return self._json({"ok": True})
        if url.path == "/status":
            m = app.motors
            return self._json({
                "motor": {"ok": m.ok, "backend": getattr(m.motor, "name", None), "error": m.error,
                          "left": m.left, "right": m.right, "max_wheel": m.max_wheel,
                          "watchdog_s": m.watchdog_s, "watchdog_trips": m.watchdog_trips},
                "camera": app.camera.status(),
                "imu": None,
            })
        if url.path == "/stream":
            return self._stream(parse_qs(url.query))
        self._json({"ok": False, "error": "not found"}, 404)

    def _stream(self, query):
        app = self.server.app
        w = _int_param(query, "w", 640, 64, 1920)
        h = _int_param(query, "h", 480, 48, 1080)
        fps = _int_param(query, "fps", 15, 1, 30)
        quality = _int_param(query, "q", 80, 10, 95)
        flip = _int_param(query, "flip", 0, 0, 1)
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        seq, next_t, idle = 0, 0.0, 0
        try:
            while not app.stopping.is_set():
                frame, seq = app.camera.wait_newer(seq, 1.0)
                if frame is None:
                    idle += 1
                    if idle >= 3:  # no frames for 3 s: end the response so the thread is not leaked
                        return
                    continue
                idle = 0
                now = time.monotonic()
                if now < next_t:
                    continue
                next_t = max(next_t + 1.0 / fps, now)
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                if flip:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not ok:
                    continue
                data = jpg.tobytes()
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_POST(self):
        app = self.server.app
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(min(length, 4096)) if length else b""
        path = urlparse(self.path).path
        if path == "/stop":
            try:
                app.motors.stop()
            except Exception as exc:
                return self._json({"ok": False, "error": str(exc)}, 500)
            return self._json({"ok": True})
        if path == "/drive":
            if app.motors.motor is None:
                return self._json({"ok": False, "error": "motor unavailable: " + app.motors.error}, 503)
            try:
                left, right = parse_wheels(body)
            except ValueError as exc:  # json.JSONDecodeError is a ValueError
                return self._json({"ok": False, "error": str(exc)}, 400)
            try:
                left, right = app.motors.drive(left, right)
            except Exception as exc:
                return self._json({"ok": False, "error": str(exc)}, 500)
            return self._json({"ok": True, "left": left, "right": right})
        self._json({"ok": False, "error": "not found"}, 404)


class Server(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class App:
    def __init__(self, motors, camera):
        self.motors = motors
        self.camera = camera
        self.stopping = threading.Event()

    def watchdog_loop(self):
        while not self.stopping.wait(0.05):
            self.motors.watchdog_tick()


def main(argv=None):
    p = argparse.ArgumentParser(description="JetBot HTTP agent (:8100)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--motor", choices=("auto", "jetbot", "adafruit", "fake"), default="auto")
    p.add_argument("--camera", default="csi",
                   help="'csi' (default), webcam index, video file, or a full GStreamer pipeline")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--max-wheel", type=float, default=0.6)
    p.add_argument("--watchdog-s", type=float, default=1.0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    if not 0 < args.max_wheel <= 1 or not 0.2 <= args.watchdog_s <= 5:
        p.error("--max-wheel in (0, 1], --watchdog-s in [0.2, 5]")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    motor, error = open_motor(args.motor)
    if motor is None:
        LOG.error("Motor unavailable (%s). /drive will return 503.", error)
    else:
        LOG.info("Motor backend: %s", motor.name)
    motors = MotorController(motor, error, args.max_wheel, args.watchdog_s)
    camera = Camera(args.camera, args.width, args.height)
    app = App(motors, camera)
    server = Server((args.host, args.port), Handler)
    server.app = app
    threading.Thread(target=app.watchdog_loop, daemon=True).start()

    def shutdown(signum, frame):
        LOG.info("Signal %s: stopping", signum)
        app.stopping.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    LOG.info("Listening on http://%s:%d (camera=%s)", args.host, args.port, args.camera)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        app.stopping.set()
        try:
            motors.stop()
        except Exception:
            LOG.exception("Final stop failed")
        camera.close()
        server.server_close()
        LOG.info("Agent stopped, motors 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
