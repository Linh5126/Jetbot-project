# Chạy on-board trên Jetson Nano (TensorRT, không cần PC/WiFi)

Một pipeline chung cho cả hai chế độ:

YOLOv8n-pose (TensorRT FP16, 480x288, box + 17 keypoint) → chọn box mục tiêu → `StagedFollowController`
(`src/following/controller.py`) → `JetBotHttpDriver` → agent `127.0.0.1:8100`.

Hai chế độ chỉ khác nhau ở cách chọn box mục tiêu (`onboard/modes.py`):

- `--mode guarded` (mặc định): ByteTrack (`onboard/bytetrack.py`, bản numpy cho Python 3.6, cho kết quả như tracker trên PC) → OSNet x0.5
  → `GuardedTargetLock`. Chọn người bằng giơ tay, chỉ bám đúng người đã xác minh bằng Re-ID.
- `--mode nearest`: baseline không có danh tính. Mỗi frame bám người gần tâm ảnh nhất, nên dễ bám nhầm người.

Khác bản PC: không có V5 fusion (`target_id = track_id`). Không có màn hình hay chuột, nên chọn người bằng cử chỉ.

## Mô phỏng trên PC (trước khi thử xe thật)

```
python scripts/sim_follow.py            # 8 kịch bản x 24 bộ tham số x 2 mode, ~2 phút
python scripts/sim_follow.py --scenario crossing --mode guarded --seeds 1 -v
```

Chạy đúng code thật (`onboard/modes.py`, `StagedFollowController`, `JetBotHttpDriver`) trong một thế giới giả:
camera pinhole, người đi theo kịch bản, embedding Re-ID khớp khoảng cách đo ngày 25/11, robot có deadzone
và độ trễ motor lấy từ log chạy sàn 25/11. Kịch bản: `stand`, `walk_away`, `turns`, `crossing` (người khác
cắt ngang), `crowd` (lớp học có người đứng xung quanh), `lookalike` (người mặc giống đi cạnh), `sidestep`
(mục tiêu chạy khỏi khung), `approach` (mục tiêu bước lại gần xe). Lỗi bị đánh trượt: xe đâm người,
xe chạy theo người khác, xe tiến khi còn cách dưới 0,9 m, kết thúc quá gần hoặc quá xa (sau khi mọi người
dừng, xe có 30 s để tới, vì xe chậm hơn người đi bộ). Tốc độ robot trong mô phỏng lấy từ lần đo 27/11.
Không mô phỏng: chất lượng ảnh thật, vật cản, trượt bánh, pin yếu. PASS ở đây chưa đảm bảo xe thật chạy đúng.

## Kiểm tra pose trước (xe không chạy)

```
sudo systemctl stop jetbot-follow     # nếu service đang chạy
python3 -m onboard.pose_check         # 90 s; --seconds, --hold
```

Chỉ đọc camera và chạy engine pose, không bao giờ gửi lệnh motor, không cần Re-ID hay đăng ký.
Ngồi (hoặc đứng) cách camera 1,5–2 m, giơ cả hai tay quá vai. Mỗi nửa giây in từng người:
`h` là chiều cao box / khung; `L`, `R` là cổ tay trái / phải cao hơn vai bao nhiêu (phần chiều cao khung,
`-` = không thấy rõ vai hoặc cổ tay); `RAISED` khi cả hai tay > `select_hand_margin`.
`PASS` khi đúng một người `RAISED` liên tục `select_hold_frames` frame, cùng luật chế độ guarded dùng để chọn.
Hết giờ thì in `FAIL` kèm lý do hay gặp nhất. Ảnh có vẽ vai (xanh) và cổ tay (đỏ): `logs/pose_check.jpg`.

## Chọn người (`--mode guarded`)

1. **WAIT**: đứng trước xe, cả người lọt khung (cách khoảng 1–2 m), **giơ cả hai tay lên quá vai** khoảng 0,5 s
   (`select_hold_frames`, `select_hand_margin`; code ở `onboard/selection.py`).
   Người khác vẫn được ở trong khung, miễn là chỉ một người giơ tay. Hai người cùng giơ thì xe không chọn ai.
2. **ENROLL** (8 s, xe đứng yên): hạ tay, xoay người chậm một vòng (trước, nghiêng, sau).
   Mỗi góc nhìn mới được thêm vào reference, tối đa 4 góc.
3. **FOLLOW**: motor tự bật. Mất người quá `max_lost_seconds` thì về WAIT, giơ tay chọn lại.

Log định kỳ in lý do chưa chọn được, ví dụ `not selected: nobody raising both hands` hoặc `2 people raising hands`.
Xem log: `journalctl -u jetbot-follow -f`.

## Cách xe bám (`onboard.follow`, cả hai mode)

Cột `reason` trong log CSV ghi pha hiện tại của `StagedFollowController`:

- `IDLE`: không có mục tiêu (guarded: Guard chưa chắc đúng người). Xe dừng.
- `SETTLING`: box phải ổn định `settle_frames` frame liên tiếp (IoU với frame trước >= `min_continuity_iou`)
  mới được chạy. Box nhảy chỗ thì đếm lại.
- `APPROACH`: tiến với `forward_speed`, vừa tiến vừa xoay về phía người. Còn cách chiều cao dừng dưới
  `slow_band` thì giảm dần tới `min_forward_ratio` x `forward_speed`, để xe không lao tới người.
- `COAST`: đang tiến hết tốc (người còn xa) mà mục tiêu mất tối đa `coast_frames` frame (~0,5 s, thường do
  Re-ID đang xác minh lại) thì xe chạy thẳng tiếp, không dừng rồi ramp lại. Ở gần người thì dừng ngay.
- `PAUSE`: mất mục tiêu ngắn (<= `coast_frames`) lúc không tiến hết tốc: xe đứng yên nhưng nhớ box, người hiện
  lại đúng chỗ thì chạy tiếp ngay, không SETTLING lại.
- `SEARCH`: người biến mất ở rìa ảnh (đi ra khỏi khung sang bên): xoay chậm `search_turn` về phía đó, tối đa
  `search_frames`, một chiều, không tiến. Mất ở giữa ảnh thì không xoay.
- `HOLD`: dừng tiến khi người đủ gần (chiều cao box >= `stop_height_ratio`), chỉ tiến lại khi
  < `stop_height_ratio - resume_margin` để xe không giật. Người lệch quá `max_side_error` sang bên thì xe chỉ xoay tại chỗ.

Lệnh xoay: trong `horizontal_deadband` thì không xoay. Ngoài vùng đó, lệnh được làm mượt theo `turn_smoothing`
để vòng camera chậm không làm xe xoay quá đà.

Tốc độ đo 27/11 (pulse bánh 0,32 trong 2 s: ~30 cm, hơi lệch phải): `forward_speed` 0,21 cho bánh 0,50
(= `drive.max_wheel`), khoảng 0,29 m/s. Người đi bình thường (~0,35 m/s) nhanh hơn xe: khi thử, đi chậm
và đứng chờ cho xe tới. Lệch phải nhẹ không cần chỉnh, vòng camera tự sửa hướng.

Lưu ý an toàn:
- Không có WiFi thì không có nút dừng từ xa. Dừng xe bằng cách nhấc xe hoặc tắt nguồn.
- Ở WAIT, ai giơ hai tay một mình trước xe cũng có thể được chọn.
- `nearest` bám bất kỳ ai gần tâm ảnh. Chỉ dùng để so sánh, ở chỗ vắng.
- Thử lần đầu: dùng `--dry-run` (motor không bao giờ chạy) hoặc kê bánh.
- Camera phải nhìn ngang tầm người. Chĩa lên trần thì không thấy tay.

## Vì sao xe mất dấu, và Guard xử lý thế nào (`guard:` trong configs/jetbot.yaml)

Phân tích log PC 27/11 (`logs/pc_20261127_192720_*`, xem lại bằng `scripts/replay_log.py`):

| Nguyên nhân | Dấu hiệu trong log | Xử lý |
|---|---|---|
| Đổi ánh sáng / nghiêng người, ảnh mẫu chỉ có 1 | cùng `track_id`, `APPEARANCE_MISMATCH`, d 0,38-0,52 | `continuity_distance` 0,45 cho track đang bám liên tục; `auto_views` tự thêm ảnh mẫu (phải khớp ảnh gốc <= `track_distance`) |
| Ngược sáng, khuất nửa: detector kém tự tin, không có embedding | `NO_EVIDENCE`, conf 0,14-0,29 | `blind_frames` 7: giữ track đang bám nếu không ai chồng lên và không ai khác giống |
| YOLO vẽ thêm box nửa trên trùng người | `OCCLUDED` với 2 box chung đỉnh đầu | `drop_part_boxes` |
| Detector không thấy người (ngược sáng mạnh) | không có box | PC: `pc.detector_weights: yolov8s.pt` (phát lại: 79 % -> 88,5 % frame bám được) |
| Đi ra khỏi khung sang bên | box cuối ở rìa ảnh | controller `SEARCH` |
| Track mới (ID khác) | `VERIFYING` | vẫn chặt: `recovery_distance` 0,30, 5 frame xác nhận (chống bám nhầm) |

Kết quả phát lại video thật (bỏ đoạn nhấc xe): bám được 67 % -> 95 % số frame, số lần mất dấu 17 -> 6.
Còn lại: người khuất sau góc tường với đèn chiếu thẳng sau đầu.

## Chạy trên PC và debug (`demo_pc.py --driver jetbot`)

```
ssh -t jetbot@<ip> "sudo systemctl stop jetbot-follow"     # không cho Nano và PC cùng lái
.venv\Scripts\python demo_pc.py --source jetbot --driver jetbot --device cuda:0
python scripts/replay_log.py logs/pc_<giờ> --device cuda:0  # phát lại offline, so Guard cũ / mới
```

Mỗi lần chạy xe thật tự ghi vào `logs/`: `pc_<giờ>_robot.csv` (lệnh, bánh, pha, độ trễ),
`_trace.csv` (Guard), `_dets.csv` (mọi box mỗi frame + khoảng cách Re-ID), `_raw.mp4` (video gốc để phát lại).

## Cài đặt (một lần)

Trên PC (cần ultralytics, torch, torchreid, onnx):

```
python scripts/export_onboard_models.py
```

Tạo `models/onboard/yolov8n_pose_288x480.onnx`, `osnet_x0_5.onnx`, `parity.npz` (box, keypoint, embedding tham chiếu).

Chép sang Nano vào `~/jetbot_follow/` (qua ssh hoặc USB): `onboard/`, `src/` (các module dùng tới),
`configs/jetbot.yaml`, `configs/bytetrack_custom.yaml`, `models/onboard/`.

Nano (JetPack 4.5, Python 3.6) thiếu `dataclasses`: chép `dataclasses.py` (backport 0.8) vào user site
(`python3 -c "import site; print(site.USER_SITE)"`). Không để file này trong repo vì nó che stdlib trên PC.

Build engine trên Nano (engine gắn với đúng bản TensorRT/GPU, phải build tại chỗ, mất ~15–20 phút):

```
cd ~/jetbot_follow && sh onboard/build_engines.sh
python3 -m onboard.check_engines     # cần in "PARITY OK" (box, keypoint, OSNet) và thời gian
```

## Chạy

```
cd ~/jetbot_follow
python3 -m onboard.run --dry-run                   # thử, motor tắt
python3 -m onboard.run                             # thật, --mode guarded
python3 -m onboard.run --mode nearest --dry-run    # baseline gần tâm ảnh
```

Log từng frame: `logs/onboard_<thời gian>.csv`. Dòng log "frame N, X FPS" cho FPS thật:
ghi vào `onboard.expected_fps` trong `configs/jetbot.yaml` (dùng để đổi `max_lost_seconds` ra số frame).

## Tự chạy khi bật nguồn

```
sudo cp ~/jetbot_follow/onboard/jetbot-follow.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now jetbot-follow
journalctl -u jetbot-follow -f
sudo systemctl disable --now jetbot-follow   # tắt, quay lại dùng PC
```

Sửa code hoặc config trên Nano thì service chỉ nhận khi khởi động lại: `sudo systemctl restart jetbot-follow`.

Không chạy `demo_pc.py --driver jetbot` cùng lúc với service: hai bên sẽ tranh nhau gửi lệnh `/drive`.
