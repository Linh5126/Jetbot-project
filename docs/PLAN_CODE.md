# PLAN CODE · Đưa pipeline bám người lên JetBot thật

Bản đặc tả để code theo. Mọi lựa chọn đã chốt; phần nào cần đo trên robot thì nằm trong file config, không nằm trong code.

Phạm vi: nối pipeline hiện có (`demo_pc.py`, đang chạy với `MockRobotDriver`) vào JetBot của trường qua agent HTTP cổng 8100. Không train model, không sửa thuật toán danh tính.

---

## 0. Quyết định đã chốt

| Hạng mục | Chọn | Lý do |
|---|---|---|
| Nơi chạy AI | **PC có GPU (off-board)** | `ultralytics>=8.4` không cài được trên Python 3.6 của JetPack 4.5 |
| Detector | **YOLOv8n** (`yolov8n.pt`), `imgsz=640`, class 0 | Đã có, đã benchmark |
| Tracker | **ByteTrack** (`configs/bytetrack_custom.yaml`) | Giữ nguyên |
| Re-ID | **OSNet-x0.5 MSMT17** (`checkpoints/osnet_x0_5_msmt17.pth.tar`) | Bản chốt của nhóm |
| Identity | `--mode v5 --ablation adaptive_fusion`, Identity Guard **on** | Bản chốt của nhóm |
| Ảnh từ robot | MJPEG `GET /stream?w=640&h=480&fps=15&q=80` | Agent có sẵn |
| Lệnh tới robot | HTTP `POST /drive {left,right}`, 10 Hz, `requests.Session` | Đơn giản, đủ nhanh trong LAN |
| Đọc MJPEG | **Tự parse bằng `requests`**, không dùng `cv2.VideoCapture(url)` | Không phụ thuộc backend FFMPEG của OpenCV |
| Entry point | **Sửa `demo_pc.py`**, thêm `--driver mock\|jetbot` | Dùng lại toàn bộ UI, trace, phím tắt |
| Hành vi khi mất mục tiêu | **Đứng yên** (không xoay tìm) | Giữ nguyên tính fail-closed |

Không làm trong đợt này (on-board/TensorRT làm sau, xem `docs/ONBOARD.md`): xoay tìm mục tiêu, tránh vật cản, khử méo fisheye, WebSocket.

Ràng buộc bắt buộc: với `--driver mock` và nguồn là file/ảnh, hành vi phải **giống hệt hiện tại** (để kết quả MOT17 và 50 test cũ không đổi).

---

## 1. Kiến trúc

```
JetBot (agent :8100)                         PC
────────────────────                         ─────────────────────────────────────
/stream  ── MJPEG ──────────────>  MjpegHttpSource (thread, giữ frame mới nhất)
                                           │ FramePacket(frame, seq, t_recv)
                                           ▼
                                   PersonIdentityPipeline.process()   [không sửa]
                                           ▼
                                   GuardedTargetLock.update()         [không sửa]
                                           ▼
                                   FollowController.command()         [không sửa]
                                           ▼
                                   cổng an toàn trong demo_pc (armed, frame cũ, timeout giây thật)
                                           ▼
/drive   <── POST 10 Hz ────────  JetBotHttpDriver (thread, TTL 0,4 s)
/stop    <── khi thoát / Space
```

Ba lớp dừng xe, độc lập nhau:

1. **Guard**: không chắc danh tính thì `MotionCommand()` bằng 0.
2. **TTL trong driver**: vòng lặp chính không gọi `send()` quá 0,4 s thì driver tự gửi 0.
3. **Watchdog của agent**: PC chết hoặc rớt mạng quá 1 s thì agent tự dừng motor.

Lớp 2 là bắt buộc. Thiếu nó, luồng nền của driver sẽ tiếp tục "vỗ" watchdog bằng lệnh cũ khi vòng lặp chính bị treo, và lớp 3 mất tác dụng.

---

## 2. Danh sách file

| File | Việc |
|---|---|
| `configs/jetbot.yaml` | **Mới.** Toàn bộ tham số robot |
| `src/runtime/frame_source.py` | **Mới.** Nguồn frame thống nhất |
| `src/following/jetbot_driver.py` | **Mới.** Driver thật |
| `scripts/jetbot_tool.py` | **Mới.** Kiểm tra kết nối, thử motor |
| `tests/test_jetbot_driver.py` | **Mới** |
| `tests/test_frame_source.py` | **Mới** |
| `demo_pc.py` | **Sửa** (mục 6) |
| `requirements.txt` | Thêm dòng `requests>=2.31` |
| `.gitignore` | **Mới:** `data/`, `outputs/`, `.venv/`, `__pycache__/`, `*.pyc` |

Không sửa: `src/identity/*`, `src/reid/*`, `src/following/{controller,target_lock,guarded_lock,driver}.py`, `src/runtime/{person_pipeline,dashboard,overlay}.py`, `src/evaluation/*`.

---

## 3. `configs/jetbot.yaml`

```yaml
robot:
  host: 192.168.16.44        # SỬA: IP hiện trên OLED của bot
  port: 8100

stream:
  width: 640
  height: 480
  fps: 15
  quality: 80
  flip: 0                    # đổi thành 1 nếu ảnh bị lộn ngược
  connect_timeout_s: 3.0
  read_timeout_s: 2.0

runtime:
  expected_fps: 12           # SỬA: FPS pipeline đo được ở bước 7.2
  stale_frame_s: 0.35        # frame cũ hơn mức này -> lệnh 0
  no_frame_timeout_s: 0.5    # không có frame mới trong khoảng này -> lệnh 0
  max_lost_seconds: 5.0      # giây thật, sau đó STOPPED

drive:
  forward_to_wheel: 2.0      # forward tối đa 0,15 -> 0,30
  turn_to_wheel: 0.4         # turn tối đa 0,45 -> 0,18
  wheel_deadzone: 0.15       # SỬA: đo ở bước 7.3
  max_wheel: 0.50
  left_trim: 1.0             # SỬA nếu xe đi lệch
  right_trim: 1.0
  invert_left: false
  invert_right: false
  send_hz: 10
  command_ttl_s: 0.4
  http_timeout_s: 0.3
  max_step_per_tick: 0.15    # giới hạn tăng tốc mỗi lần gửi

controller:                  # ghi đè ControllerConfig
  desired_person_height_ratio: 0.55   # SỬA sau khi thử khoảng cách bám
  forward_gain: 0.55
  turn_gain: 0.7
  max_forward: 0.15
  max_turn: 0.45
  horizontal_deadband: 0.07
  height_deadband: 0.06
  min_target_height_ratio: 0.08
```

Bốn dòng đánh dấu `SỬA` là thứ duy nhất phải chỉnh theo robot thật.

---

## 4. `src/runtime/frame_source.py`

### Interface

```python
@dataclass(frozen=True)
class FramePacket:
    frame: np.ndarray
    seq: int          # tăng dần, bắt đầu từ 1
    t_recv: float     # time.monotonic() lúc PC nhận xong frame

class FrameSource(Protocol):
    live: bool                    # True: camera/stream, False: file
    nominal_fps: float | None     # FPS của file; None với nguồn live
    ended: bool                   # True khi file hết; nguồn live không bao giờ True
    def read(self, timeout: float) -> FramePacket | None: ...
    def close(self) -> None: ...

def open_source(raw: str, stream_cfg: dict | None = None) -> FrameSource: ...
def build_stream_url(robot_cfg: dict, stream_cfg: dict) -> str: ...
```

### Ba lớp cài đặt

| Lớp | Dùng khi | Hành vi `read()` |
|---|---|---|
| `FileSource` | đường dẫn file, mẫu `%06d.jpg` | Gọi thẳng `capture.read()`, **không bỏ frame**, `seq` tăng 1. Hết file: `ended=True`, trả `None` |
| `LatestCaptureSource` | `raw.isdecimal()` (webcam) | Thread đọc liên tục, chỉ giữ frame mới nhất |
| `MjpegHttpSource` | `raw` bắt đầu bằng `http://` hoặc `https://` | Thread parse MJPEG, chỉ giữ frame mới nhất, tự kết nối lại |

Quy tắc chung cho hai lớp live: `read(timeout)` chờ tới khi có frame có `seq` lớn hơn frame đã trả lần trước; hết `timeout` thì trả `None`. Không bao giờ trả lại frame cũ.

`open_source`: `raw.isdecimal()` → `LatestCaptureSource(int(raw))`; bắt đầu bằng `http` → `MjpegHttpSource(raw, ...)`; còn lại → `FileSource(raw)`.

`build_stream_url` trả `http://{host}:{port}/stream?w=..&h=..&fps=..&q=..&flip=..`.

### Lõi của `MjpegHttpSource`

```python
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
                    while True:
                        start = buf.find(b"\xff\xd8")
                        if start < 0:
                            del buf[:-1]          # giữ 1 byte phòng marker bị cắt đôi
                            break
                        end = buf.find(b"\xff\xd9", start + 2)
                        if end < 0:
                            del buf[:start]
                            break
                        jpg = bytes(buf[start:end + 2])
                        del buf[:end + 2]
                        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            self._publish(frame)
                    if len(buf) > 4_000_000:
                        buf.clear()
        except requests.RequestException as exc:
            LOG.warning("Stream lỗi (%s); thử lại sau 1 s", exc)
        self._stop.wait(1.0)

def _publish(self, frame) -> None:
    with self._cond:
        self._seq += 1
        self._packet = FramePacket(frame, self._seq, time.monotonic())
        self._cond.notify_all()

def read(self, timeout: float) -> FramePacket | None:
    with self._cond:
        ok = self._cond.wait_for(lambda: self._seq > self._last_returned, timeout)
        if not ok:
            return None
        self._last_returned = self._seq
        return self._packet
```

Thread là `daemon=True`. `close()` đặt `_stop`, `join(timeout=2)`.

---

## 5. `src/following/jetbot_driver.py`

### Hàm trộn bánh (thuần, dễ test)

`MotionCommand.turn` dương là **rẽ trái** (xem `controller.py`), nên bánh phải nhanh hơn:

```python
@dataclass(frozen=True)
class DriveConfig:
    forward_to_wheel: float = 2.0
    turn_to_wheel: float = 0.4
    wheel_deadzone: float = 0.15
    max_wheel: float = 0.50
    left_trim: float = 1.0
    right_trim: float = 1.0
    invert_left: bool = False
    invert_right: bool = False
    send_hz: float = 10.0
    command_ttl_s: float = 0.4
    http_timeout_s: float = 0.3
    max_step_per_tick: float = 0.15

    def __post_init__(self) -> None:
        if not 0 <= self.wheel_deadzone < self.max_wheel <= 1.0:
            raise ValueError("Cần 0 <= wheel_deadzone < max_wheel <= 1")
        if self.send_hz < 2 or not 0.1 <= self.command_ttl_s < 1.0:
            raise ValueError("send_hz >= 2 và command_ttl_s trong [0.1, 1.0)")


def mix(command: MotionCommand, cfg: DriveConfig) -> tuple[float, float]:
    """(forward, turn) -> (left, right) trong [-max_wheel, max_wheel]."""
    if command.forward == 0.0 and command.turn == 0.0:
        return 0.0, 0.0
    f = cfg.forward_to_wheel * command.forward
    t = cfg.turn_to_wheel * command.turn

    def shape(x: float, trim: float, invert: bool) -> float:
        if abs(x) < 1e-3:
            return 0.0
        y = math.copysign(cfg.wheel_deadzone + abs(x), x) * trim
        y = max(-cfg.max_wheel, min(cfg.max_wheel, y))
        return -y if invert else y

    return shape(f - t, cfg.left_trim, cfg.invert_left), shape(f + t, cfg.right_trim, cfg.invert_right)
```

### Lớp driver

```python
class JetBotHttpDriver:                      # thỏa Protocol RobotDriver
    def __init__(self, base_url: str, cfg: DriveConfig, *,
                 post=None, clock=time.monotonic, start_thread: bool = True): ...
    def preflight(self) -> dict: ...         # GET /health + /status
    def send(self, command: MotionCommand) -> None: ...
    def close(self) -> None: ...
    @property
    def link_ok(self) -> bool: ...
    @property
    def last_wheels(self) -> tuple[float, float]: ...
    last_command: MotionCommand              # giữ tương thích với MockRobotDriver
    commands_sent: int
```

Hành vi phải đúng từng điểm sau:

1. `post` mặc định là `lambda path, payload: session.post(base_url + path, json=payload, timeout=cfg.http_timeout_s)`. Tham số `post`, `clock`, `start_thread` chỉ để test.
2. `preflight()`: `/health` phải trả `ok: true`; `/status` phải có `motor.ok == True`. Sai thì `raise RuntimeError` kèm nội dung `/status`. Trả về dict `/status`.
3. `send(cmd)`: dưới lock, lưu `(left, right) = mix(cmd, cfg)` làm mục tiêu và `t_last_send = clock()`. Không gọi mạng trong `send()`. Nếu mục tiêu mới là `(0, 0)` thì `set()` một `Event` để thread gửi ngay.
4. Thread chạy `_tick()` mỗi `1/send_hz` giây (hoặc ngay khi `Event` được set). Trong `_tick()`:
   - Nếu `clock() - t_last_send > command_ttl_s` → mục tiêu coi như `(0, 0)`.
   - Mục tiêu `(0, 0)` → gửi `(0, 0)` ngay, **không** giới hạn bước.
   - Ngược lại, mỗi bánh chỉ được đổi tối đa `max_step_per_tick` so với giá trị đã gửi lần trước.
   - `POST /drive {"left": l, "right": r}`. Lỗi `requests.RequestException` → tăng bộ đếm lỗi, không ném ra ngoài. Thành công → bộ đếm về 0.
5. `link_ok` là `False` khi có từ 3 lỗi liên tiếp.
6. `close()`: đặt mục tiêu 0, dừng thread (`join(timeout=1)`), rồi `POST /stop` hai lần, bỏ qua lỗi.
7. Chưa từng gọi `send()` → thread gửi `(0, 0)`.

---

## 6. Sửa `demo_pc.py`

Giữ nguyên `parse_source`, `select_at_point`, `draw_frame` (test đang import).

### 6.1 Tham số mới

```
--driver {mock,jetbot}      mặc định mock
--robot-config PATH         mặc định configs/jetbot.yaml
--source                    thêm giá trị đặc biệt "jetbot" = dựng URL từ robot-config
--record-raw PATH           ghi video gốc từ camera (không có dashboard), để chạy lại offline
--robot-log PATH            CSV chẩn đoán robot
```

Kiểm tra: `--driver jetbot` mà `--gt-sequence` được đặt → `parser.error` (không chấm GT khi chạy robot).

### 6.2 Khởi tạo (đầu hàm `run`)

```python
robot_cfg = None
if args.driver == "jetbot" or args.source == "jetbot":
    robot_cfg = yaml.safe_load(Path(args.robot_config).read_text(encoding="utf-8"))

raw_source = (build_stream_url(robot_cfg["robot"], robot_cfg["stream"])
              if args.source == "jetbot" else args.source)
source = open_source(raw_source, robot_cfg["stream"] if robot_cfg else None)

if source.live:
    fps = float(robot_cfg["runtime"]["expected_fps"]) if robot_cfg else 15.0
else:
    fps = source.nominal_fps if source.nominal_fps and 1.0 <= source.nominal_fps <= 120.0 else 30.0
# khối "if gt is not None ..." giữ nguyên phía sau

if args.driver == "jetbot":
    base = f"http://{robot_cfg['robot']['host']}:{robot_cfg['robot']['port']}"
    driver = JetBotHttpDriver(base, DriveConfig(**robot_cfg["drive"]))
    LOG.info("JetBot status: %s", driver.preflight())
    controller = FollowController(ControllerConfig(**robot_cfg["controller"]))
else:
    driver = MockRobotDriver()
    controller = FollowController(ControllerConfig())

armed = args.driver == "mock"      # robot thật luôn khởi động ở trạng thái SAFE
max_lost_seconds = (float(robot_cfg["runtime"]["max_lost_seconds"])
                    if robot_cfg else args.max_lost_seconds)
```

`max_lost_frames` của Guard vẫn tính bằng `int(max_lost_seconds * fps)` như cũ. Bỏ `cv2.VideoCapture` và `capture.release()`, thay bằng `source.close()` trong `finally`.

### 6.3 Vòng lặp

Thay đoạn `ok, frame = capture.read()` bằng:

```python
timeout = robot_cfg["runtime"]["no_frame_timeout_s"] if (robot_cfg and source.live) else 5.0
packet = source.read(timeout)
if packet is None:
    if source.ended:
        LOG.info("Video source ended; requested STOP")
        break
    driver.send(MotionCommand())                 # mất hình -> dừng
    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
        break
    continue
frame = packet.frame
```

Sau `command = controller.command(...)`, trước `driver.send(command)`, chèn cổng an toàn (chỉ có tác dụng với nguồn live):

```python
now = time.monotonic()
gate = ""
if source.live:
    holding = result.state in (FollowState.LOST, FollowState.VERIFYING)
    lost_since = (lost_since or now) if holding else None
    if lost_since is not None and now - lost_since > max_lost_seconds:
        result = lock.stop("Target lost timeout (wall clock)")
        runtime["lock_result"] = result
        gate = "LOST_TIMEOUT"
    stale_s = robot_cfg["runtime"]["stale_frame_s"] if robot_cfg else 0.35
    if now - packet.t_recv > stale_s:
        gate = "STALE_FRAME"
if not armed:
    gate = gate or "SAFE"
if gate:
    command = MotionCommand()
driver.send(command)
```

`lost_since = None` khai báo trước vòng lặp. Cần `from src.following.target_lock import FollowState`. Lệnh ghi vào trace phải là `command` **sau** cổng an toàn (nếu không `evaluate_follow_trace` sẽ báo "unauthorized motion").

### 6.4 Phím và hiển thị

- Phím `M`: đảo `armed`, chỉ khi `args.driver == "jetbot"`. Ghi log và `dashboard.add_event(processed, "MOTORS ARMED" / "MOTORS SAFE")`.
- Phím `Space` và `R`: giữ như cũ, thêm `armed = False` khi chạy jetbot.
- Thanh trạng thái: vẽ **lên `shown`** (bản hiển thị, sau `fit_display_frame`), không vẽ lên `rendered`, nên video ghi lại và `dashboard.py` không đổi:

```python
if args.driver == "jetbot":
    l, r = driver.last_wheels
    text = (f"{'ARMED' if armed else 'SAFE'} | link {'OK' if driver.link_ok else 'DOWN'} | "
            f"age {1000 * (time.monotonic() - packet.t_recv):.0f} ms | L {l:+.2f} R {r:+.2f} | {gate}")
    color = (0, 0, 255) if armed else (0, 200, 0)
    cv2.rectangle(shown, (0, 0), (shown.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(shown, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
```

### 6.5 Ghi dữ liệu

- `--record-raw`: `cv2.VideoWriter` thứ hai, ghi `frame` gốc với `fps` ở trên, mở ở frame đầu tiên, `release()` trong `finally`.
- `--robot-log`: CSV với cột `frame, t_wall_s, frame_age_ms, loop_ms, state, gate, armed, forward, turn, left, right, link_ok`. `left`/`right` lấy từ `driver.last_wheels` (với mock ghi rỗng). Ghi mỗi frame, đóng trong `finally`.

---

## 7. `scripts/jetbot_tool.py`

Ba lệnh con, đều đọc `--robot-config configs/jetbot.yaml`:

| Lệnh | Việc |
|---|---|
| `check` | GET `/health`, `/status` (in `motor.ok`, `camera`, `imu`); đo RTT 10 lần trên `/health` (in min/trung bình/max); mở stream 5 giây, in FPS nhận được và kích thước frame; lưu frame cuối vào `outputs/jetbot_check.jpg` |
| `pulse --left L --right R --seconds S` | Gửi `/drive` 10 Hz trong S giây rồi `/stop`, có `try/finally`. Từ chối nếu `abs(L)` hoặc `abs(R)` > 0,6 hoặc S > 3 |
| `ramp --wheel left\|right\|both` | Tăng từ 0,02 tới 0,40, bước 0,02, mỗi mức 0,8 giây, in mức đang chạy. Người dùng nhấn Ctrl+C khi bánh bắt đầu quay; mức in cuối cùng là `wheel_deadzone` |

---

## 8. Test mới

`tests/test_jetbot_driver.py` (dùng `post` giả ghi lại lời gọi, `clock` giả, `start_thread=False`, gọi `_tick()` trực tiếp):

1. `mix(MotionCommand(), cfg) == (0.0, 0.0)`.
2. `turn > 0` → `right > left`; `turn < 0` → `left > right`.
3. Đi thẳng: `left == right > wheel_deadzone`.
4. Không giá trị nào vượt `max_wheel`, kể cả với `forward=1, turn=1`.
5. `invert_left=True` đảo dấu bánh trái; `left_trim=0.9` làm bánh trái nhỏ hơn.
6. Sau `send(lệnh khác 0)`, `_tick()` gửi `/drive` với giá trị khác 0 nhưng không vượt `max_step_per_tick`.
7. Tua `clock` quá `command_ttl_s` rồi `_tick()` → gửi `(0, 0)`.
8. `send(MotionCommand())` → `_tick()` gửi `(0, 0)` ngay, không giảm dần.
9. `post` ném `requests.ConnectionError` ba lần → `link_ok is False`, không có exception thoát ra.
10. `close()` → lời gọi cuối cùng là `/stop`.
11. `preflight()` ném `RuntimeError` khi `/status` trả `motor.ok == False`.
12. `DriveConfig(wheel_deadzone=0.6, max_wheel=0.5)` ném `ValueError`.

`tests/test_frame_source.py`:

1. Tách hàm parse buffer thành `extract_jpegs(buf: bytearray) -> list[bytes]`; nạp hai JPEG hợp lệ (tạo bằng `cv2.imencode`) xen rác và bị cắt ở giữa chunk → trả đúng hai ảnh, decode được.
2. `LatestCaptureSource` với capture giả: publish 3 frame trước khi `read()` → `read()` trả frame thứ 3; `read()` lần nữa với `timeout=0.05` trả `None`.
3. `FileSource` với video 5 frame (ghi bằng `cv2.VideoWriter` vào `tmp_path`) → đọc đủ 5 frame theo thứ tự, sau đó `None` và `ended is True`.
4. `open_source("0")`, `open_source("http://x/stream")`, `open_source("a.mp4")` trả đúng lớp (mock phần mở thiết bị).
5. `build_stream_url` ra đúng chuỗi mong đợi.

Toàn bộ `python -m pytest` phải xanh, gồm cả test cũ.

---

## 9. Thứ tự làm và tiêu chí xong

| Bước | Việc | Xong khi |
|---|---|---|
| 1 | `.gitignore`, `requirements.txt`, `configs/jetbot.yaml` | `pip install -r requirements.txt` chạy được |
| 2 | `frame_source.py` + test | `pytest tests/test_frame_source.py` xanh |
| 3 | `jetbot_driver.py` + test | `pytest tests/test_jetbot_driver.py` xanh |
| 4 | Sửa `demo_pc.py` | `python -m pytest` xanh; `demo_pc.py --source <video MOT17>` chạy như cũ |
| 5 | `jetbot_tool.py` | `--help` của ba lệnh con chạy được |

Bước 1–5 không cần robot.

## 10. Quy trình chạy lần đầu trên robot

Pin sạc đầy, PC và bot cùng mạng. **Kê bánh khỏi mặt đất tới hết bước 10.4.**

1. **Kết nối.** Sửa `robot.host`. Chạy `python scripts/jetbot_tool.py check`. Cần: `motor.ok` là true, RTT trung bình dưới 50 ms, FPS stream từ 10 trở lên. Mở `outputs/jetbot_check.jpg`; ảnh lộn ngược thì đặt `stream.flip: 1`.
2. **Đo FPS pipeline.** `python demo_pc.py --source jetbot --driver mock --device cuda:0`. Đứng trước camera, click chọn mình, nhấn `Q`, đọc dòng log `mean pipeline FPS`. Ghi vào `runtime.expected_fps`. Dưới 8 FPS thì giảm `stream.width/height` xuống 480×360.
3. **Đo vùng chết.** `python scripts/jetbot_tool.py ramp --wheel both`, ghi mức bánh bắt đầu quay vào `drive.wheel_deadzone`. Bánh nào quay ngược thì bật `invert_left`/`invert_right`.
4. **Thử kín, bánh vẫn kê.** `python demo_pc.py --source jetbot --driver jetbot --device cuda:0`. Click chọn mình, nhấn `M`. Kiểm tra lần lượt:
   - Đứng lệch sang **phải** khung hình → bánh **trái** quay nhanh hơn. Ngược lại thì hoán đổi hai cờ `invert` hoặc đổi `stream.flip`.
   - Lùi ra xa → cả hai bánh quay tiến. Tiến sát camera → bánh dừng.
   - Che camera → bánh dừng trong 1 giây.
   - Nhấn `Space` → dừng ngay, thanh trạng thái về `SAFE`.
   - Tắt WiFi của PC → bánh dừng trong khoảng 1 giây.
5. **Thả xuống sàn**, chỗ trống, một người đứng cạnh sẵn sàng nhấc xe. Lặp lại bước 4. Xe đi lệch thì chỉnh `left_trim`/`right_trim`. Xe lắc qua lại khi rẽ thì giảm `turn_to_wheel`. Khoảng cách bám chưa hợp thì chỉnh `desired_person_height_ratio` (tăng lên thì xe bám gần hơn).
6. **Quay dữ liệu.** Thêm `--record-raw outputs/raw_01.mp4 --trace-csv outputs/trace_01.csv --robot-log outputs/robot_01.csv`. Video gốc chạy lại offline được bằng `demo_pc.py --source outputs/raw_01.mp4` để chỉnh ngưỡng Guard mà không cần robot.

---

## 11. Những gì chưa kiểm chứng được

Bản plan này dựa trên mã nguồn hiện tại và tài liệu agent của khóa học. Code theo đúng mục 3–8 sẽ chạy được trên PC; còn các điểm sau chỉ biết khi cắm vào bot thật:

- Định dạng chính xác của `/status` (plan giả định có `motor.ok`). Nếu khác, chỉ cần sửa `preflight()`.
- `/stream` có nhận tham số `h` và `flip` đúng như ví dụ `/snapshot` trong tài liệu hay không.
- `wheel_deadzone = 0.15` là giá trị tạm; con số thật lấy từ bước 10.3.
- Ngưỡng Guard được chọn trên MOT17. Camera JetBot đặt thấp, góc rộng 160°, nên có thể phải nới `--guard-track-distance` / `--guard-recovery-distance` sau khi xem `reference_distance` trong trace thật.
- FPS thực tế trên máy của nhóm khi có nhiều người trong khung hình (Guard trích embedding cho mọi người ở mỗi frame).
