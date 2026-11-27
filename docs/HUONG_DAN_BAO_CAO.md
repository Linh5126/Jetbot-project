# Hướng dẫn viết báo cáo tiến độ đồ án (CS532 · JetBot bám theo người)

File này dành cho bạn viết báo cáo. Nó gồm: khung 6 phần, số trang cho từng phần, nội dung cần có,
**số liệu đã kiểm chứng** kèm nguồn (file/lệnh để tái lập), và những chỗ còn trống phải tự điền.
Quy tắc chung: số nào không có nguồn thì không ghi. Chỗ ghi `[ĐIỀN]` là thông tin chưa có trên máy.

## 0. Yêu cầu hình thức

- Báo cáo nhóm, 10–15 trang, đúng 6 phần bên dưới, kèm video demo.
- Trang bìa: tên trường/khoa, môn CS532, tên đề tài, GVHD `[ĐIỀN]`, thành viên + MSSV `[ĐIỀN]`, ngày.
- Thêm bảng phân công thành viên ở cuối phần 3 (ai làm module nào) `[ĐIỀN]`.
- Hình và bảng đánh số, có chú thích nguồn (tên file log, ngày đo).
- Ngày dùng đúng mốc trong log: 16/11, 17/11, 23/11, 25/11, 27/11/2026.

| Phần | Số trang gợi ý |
|---|---|
| 1. Mô tả đề tài & mục tiêu | 1–1,5 |
| 2. Kiến thức / công nghệ | 2,5–3 |
| 3. Tiến độ giữa kì | 1,5–2 |
| 4. Kết quả thực nghiệm | 3–4 |
| 5. Tối ưu & triển khai thiết bị thật | 2–2,5 |
| 6. Kế hoạch giai đoạn sau | 1 |

---

## Phần 1. Mô tả đề tài & mục tiêu

Nội dung cần viết:
- **Bài toán:** robot JetBot (NVIDIA Jetson Nano 4GB, JetPack 4.5) chỉ dùng một camera, bám theo **đúng một người**
  được chọn. Phải giữ đúng người khi có người khác đi ngang, bị che khuất hay đổi tư thế. Không chắc chắn thì dừng.
- **Hai mảng của đồ án** (README.md):
  1. Nghiên cứu: đánh giá detection/tracking/Re-ID trên MOT17 (baseline V1–V5.1, 4 ablation của V5).
  2. Hệ thống: demo trên PC, sau đó đưa lên xe thật.
- **Mục tiêu đo được** (viết thành danh sách):
  - Chọn mục tiêu bằng cách bấm chuột (chạy trên PC) hoặc giơ hai tay (chạy trên Nano).
  - Không bao giờ chạy theo người khác. Không chắc thì dừng (fail-closed).
  - Dừng cách người khoảng 1 m. Không đâm người.
  - Chạy thời gian thực trên Nano (≥ 14 FPS).
  - Có log để debug và tái lập mọi lần chạy.
- **Phạm vi / giới hạn:** chạy trong nhà. Xe chậm hơn người đi bộ (~0,29 m/s). Tránh vật cản cần dữ liệu riêng (xem phần 6).

Gợi ý hình: sơ đồ khối tổng quát: camera, phát hiện người, tracking, Re-ID, Guard, điều khiển, động cơ.

---

## Phần 2. Kiến thức / công nghệ

Viết mỗi mục khoảng 1 đoạn: là gì, vì sao chọn, dùng ở file nào.

| Thành phần | Công nghệ | Dùng ở đâu |
|---|---|---|
| Phát hiện người | YOLOv8n (Nano, PC cũ), YOLOv8s (PC từ 27/11), YOLOv8n-pose (Nano: box + 17 keypoint) | `configs/config.yaml`, `configs/jetbot.yaml` (`pc.detector_weights`, `onboard.yolo_engine`) |
| Tracking | ByteTrack (dùng cả box điểm thấp; `track_buffer 60`, `match_thresh 0.8`) | `configs/bytetrack_custom.yaml`; bản viết lại cho Nano: `onboard/bytetrack.py` |
| Re-ID (ngoại hình) | OSNet-x0.5 pretrained MSMT17, embedding 512 chiều, cosine distance | `src/reid/`, `checkpoints/osnet_x0_5_msmt17.pth.tar` |
| Quản lý danh tính (nghiên cứu) | V5 Adaptive Fusion: trộn ngoại hình + chuyển động, trọng số theo độ bất định; V5.1 thêm gallery theo chất lượng crop | `src/identity/manager_v5.py`, `manager_v51.py` |
| Khóa mục tiêu an toàn | Identity Guard (`GuardedTargetLock`): ảnh mẫu đóng băng, trạng thái TRACKING / LOST / VERIFYING / REACQUIRED / STOPPED, chỉ cho chạy khi đã xác minh | `src/following/guarded_lock.py` |
| Chọn bằng cử chỉ | Hai cổ tay cao hơn vai > 5 % chiều cao khung, giữ 5 frame, chỉ đúng 1 người | `onboard/selection.py` |
| Điều khiển | Controller dừng-và-đi theo pha: SETTLING, APPROACH (giảm tốc gần người), COAST, PAUSE, HOLD; dừng khi box cao ≥ 0,70 khung | `src/following/controller.py` (`StagedFollowController`) |
| Giao tiếp xe | Agent HTTP :8100 trên xe: `/stream` MJPEG, `/drive`, `/stop`, watchdog | `jetbot_agent/agent.py`, `src/following/jetbot_driver.py` |
| Tăng tốc trên Nano | TensorRT 7.1 FP16, input 288×480, build bằng `trtexec` ngay trên Nano | `onboard/models.py`, `onboard/build_engines.sh` |
| Tránh vật cản | ResNet18 phân loại free/blocked (công thức notebook NVIDIA JetBot), xuất ONNX | `scripts/collect_collision.py`, `scripts/train_collision.py` |
| Mô phỏng | Mô phỏng vòng kín trên PC: camera pinhole, người theo kịch bản, embedding giả lập khớp số đo thật, động học xe vi sai | `scripts/sim_follow.py` |

Cần giải thích rõ 3 ý (giảng viên hay hỏi):
1. **Vì sao cần Re-ID, không chỉ tracking:** ByteTrack đổi ID khi người bị che hoặc ra khỏi khung. Re-ID so ngoại hình với ảnh mẫu để nhận lại đúng người.
2. **Cosine distance không phải xác suất.** Ngưỡng được chọn từ số đo thật: cùng người trung vị 0,24; người khác trung vị 0,59 (log 25/11).
3. **Fail-closed:** mất bằng chứng thì lệnh chuyển động bằng 0, không đoán.

Tài liệu tham khảo cần trích: YOLOv8 (Ultralytics), ByteTrack (Zhang et al., ECCV 2022), OSNet (Zhou et al., ICCV 2019),
MOT17 (MOTChallenge), NVIDIA JetBot (github.com/NVIDIA-AI-IOT/jetbot, notebook collision_avoidance), TensorRT.

---

## Phần 3. Tiến độ giữa kì

Viết dạng bảng mốc thời gian (lấy từ git log và log chạy):

| Ngày | Việc đã xong | Bằng chứng |
|---|---|---|
| 16/11 | Bản nghiên cứu + demo PC (V1–V5.1, dashboard, Guard, đánh giá trace với MOT17 GT) | commit `d408507` |
| 17/11 | Driver xe thật qua HTTP; agent chạy trên xe (:8100, watchdog) | commit `3fb8c06`, `b5cd343` |
| 23/11 | Pipeline PC điều khiển xe qua WiFi: 14,8 FPS | ghi chú trong `configs/jetbot.yaml` |
| 25/11 | Chạy thử trên sàn 4 lần, hiệu chỉnh controller và ngưỡng Guard; pipeline trên Nano (TensorRT) 14,3 FPS | `outputs/robot_0[1-4].csv`, `outputs/trace_0[1-4].csv` |
| 27/11 | Engine pose trên Nano, chọn bằng giơ tay, gộp 2 mode (guarded / nearest), mô phỏng PC, đo tốc độ xe, sửa xe "nhích rồi vọt", Guard chịu được nghiêng người / ngược sáng, công cụ phát lại log | `logs/pc_20261127_*`, `docs/ONBOARD.md` |

Thêm một đoạn "khó khăn đã gặp và cách xử lý" (đây là điểm cộng):
- Xe xoay quá đà (25/11): giảm `turn_to_wheel` từ 0,4 xuống 0,25; làm mượt lệnh xoay; phanh tức thì.
- Bánh có vùng chết 0,08: tăng tốc từ mép vùng chết, mỗi tick +0,06.
- Pin yếu (34 %) làm xe "nhích rồi vọt": thêm ramp, giảm tốc khi gần người, cảnh báo pin < 40 %.
- Chọn người kiểu "đứng một mình giữa khung" không dùng được trong quán đông: đổi sang giơ hai tay.
- Nghiêng người là mất dấu (27/11): xem phần 4.4.

Cuối phần: bảng phân công thành viên `[ĐIỀN]`.

---

## Phần 4. Kết quả thực nghiệm

Phần quan trọng nhất. Mỗi bảng ghi rõ điều kiện đo.

### 4.1 Benchmark MOT17 (phần nghiên cứu) `[ĐIỀN]`

Không có trên máy này (`outputs/benchmark_final` không tồn tại). Chạy lại trên máy có dữ liệu MOT17:

```
python scripts/validate_dataset.py --strict-seven
python main.py --mode v5 --all-sequences --ablation all --output-root outputs/benchmark_final
python main.py --mode v5_1 --all-sequences --ablation all --output-root outputs/benchmark_final
```

Bảng nên có: các chỉ số MOT (MOTA, IDF1, IDSW…) cho V5 với 4 ablation (appearance_only, motion_only, fixed_fusion,
adaptive_fusion) và V5.1. Ghi chú: thay đổi Guard ngày 27/11 đều **mặc định tắt**, nên không ảnh hưởng benchmark.

### 4.2 Chạy thử trên sàn 25/11 (AI chạy trên PC, xe qua WiFi)

Nguồn: `outputs/trace_0[1-4].csv`, `outputs/robot_0[1-4].csv`.

| Lần | Thời lượng | FPS | % frame được phép bám | % LOST + VERIFYING | Số lần chọn lại |
|---|---|---|---|---|---|
| 1 | 28 s | 14,5 | 80 % | 19 % | 1 |
| 2 | 57 s | 14,5 | 91 % | 9 % | 2 |
| 3 | 151 s | 14,6 | 52 % | 26 % | 7 |
| 4 | 104 s | 14,6 | 65 % | 22 % | 7 |

Nhận xét: chạy càng lâu và môi trường càng đông thì càng mất dấu nhiều, phải chọn lại liên tục. Đây là lý do của cải tiến 27/11.

### 4.3 Hiệu chỉnh xe thật (27/11)

- Tốc độ: bánh 0,32 trong 2 s (có ramp) đi được khoảng 30 cm. Suy ra ~0,65 m/s cho mỗi đơn vị bánh hiệu dụng.
  Ở `forward_speed` 0,21 (bánh 0,50) xe chạy ~0,29 m/s. Xe hơi lệch phải; vòng camera tự sửa hướng.
- Đáp ứng xoay (log 25/11): 6,25 lần chiều rộng ảnh/s trên mỗi đơn vị chênh lệch bánh, trễ ~4 frame, tương quan 0,93–0,95.
- Pin: 70–72 % ổn định. Ở 34 % xe giật.

### 4.4 Chạy trên PC với xe thật, 27/11 (GPU RTX 3050)

Nguồn: `logs/pc_20261127_185728_*`, `logs/pc_20261127_185940_*`, `logs/pc_20261127_192720_*`.

| Lần | Thiết bị chạy AI | FPS | Thời gian xử lý 1 frame (trung vị) | Độ trễ ảnh (trung vị) |
|---|---|---|---|---|
| 18:57 | CPU (quên `--device cuda:0`) | 4,5 | 208 ms | 250 ms |
| 18:59 | GPU | 13,8 | 57 ms | 32 ms |
| 19:27 | GPU | 14,5 | 57 ms | 32 ms |

Kết quả quan sát: xe bám được và dừng khi người dừng. Nhưng **nghiêng người sang trái/phải là mất dấu**.
Phân tích log từng box (`_dets.csv`) cho ra các nguyên nhân sau:

| Nguyên nhân | Dấu hiệu trong log | Cách sửa |
|---|---|---|
| Nghiêng người / đổi ánh sáng, chỉ có 1 ảnh mẫu | cùng track, khoảng cách Re-ID 0,38–0,52 vượt ngưỡng 0,38 (người khác ≥ 0,44) | `continuity_distance` 0,45 chỉ cho track đang bám liên tục; `auto_views` tự thêm tối đa 3 ảnh mẫu (phải khớp ảnh gốc) |
| Ngược sáng, khuất nửa | detector chỉ tự tin 0,14–0,29, không có embedding | `blind_frames` 7 (~0,5 s) |
| YOLO vẽ thêm box nửa trên trùng người | Guard tưởng bị che | `drop_part_boxes` |
| Detector không thấy người | không có box | PC dùng YOLOv8s |

**Kết quả chính** (phát lại video thật bằng `scripts/replay_log.py`, bỏ đoạn nhấc xe, cùng detector YOLOv8s):

| | Guard cũ | Guard mới |
|---|---|---|
| % frame được phép bám | 67,3 % | **94,9 %** |
| Số lần mất dấu | 17 | 6 |

Riêng việc đổi detector (Guard mới): YOLOv8n bám được 79,1 % → YOLOv8s 88,5 % (tính cả đoạn nhấc xe). YOLOv8m không hơn YOLOv8s.
Cần ghi rõ: đây là **phát lại offline** trên video thật, chưa phải lần chạy xe mới. Phải xác nhận lại trên xe (phần 6).

Tái lập: `python scripts/replay_log.py logs/pc_20261127_192720 --weights yolov8s.pt`.

### 4.5 Mô phỏng vòng kín trên PC

Nguồn: `python scripts/sim_follow.py` (8 kịch bản × 24 bộ tham số vật lý × 2 seed = 48 lượt mỗi kịch bản).
Kịch bản: `stand`, `walk_away`, `turns`, `crossing` (người khác cắt ngang), `crowd`, `lookalike` (người mặc giống),
`sidestep` (mục tiêu chạy khỏi khung), `approach` (mục tiêu bước lại gần xe).
Một lượt bị đánh trượt nếu: đâm người, chạy theo người khác, tiến khi còn cách < 0,9 m, kết thúc quá gần/xa, hoặc mất mục tiêu.

Bảng so sánh nên có (số lượt PASS / 48, cấu hình cuối 27/11):

| Kịch bản | guarded (của nhóm) | nearest (kiểu repo NVIDIA, không Re-ID) |
|---|---|---|
| stand | 48 | 48 |
| walk_away | 48 | 48 |
| turns | 43 | 46 |
| crossing | 48 | **10** (38 lượt chạy theo người khác) |
| crowd | 47 | **15** (có va chạm) |
| lookalike | 48 | **16** (có va chạm) |
| sidestep | 48 | 20 |
| approach | 48 | 48 |

Thông điệp chính: chế độ **không có Re-ID** (giống các repo mẫu) bám nhầm người và đâm người. Chế độ `guarded` của nhóm
**không có lượt nào chạy theo người khác hay đâm người**. Các lượt trượt chỉ nằm ở góc tham số cực đoan
(trễ 0,25 s, đáp ứng xoay 8). Nên chạy lại lệnh trên để lấy số mới nhất trước khi nộp.

Các tối ưu tìm ra nhờ mô phỏng (nên viết thành 1 đoạn):
- `forward_speed` 0,12 → 0,21: bám kịp hơn trong mọi kịch bản.
- Pha COAST: mất mục tiêu ≤ 7 frame khi còn xa thì chạy thẳng tiếp. walk_away tăng từ 16/24 lên 24/24.
- Luật đầu bị cắt khỏi khung: dừng, tránh đâm người.
- Đã thử xoay tìm khi mất người (SEARCH) nhưng **tắt**, vì không có lợi.

Giới hạn cần nói thẳng: mô phỏng không có chất lượng ảnh thật, vật cản, trượt bánh, pin.

### 4.6 Chạy trên Nano (không cần PC)

- Engine TensorRT FP16 build trên Nano (TRT 7.1). So với bản PC: `PARITY OK` (`python3 -m onboard.check_engines`).
- FPS: 14,3 với YOLOv8n detect (25/11); 14,3–14,5 với YOLOv8n-pose (27/11).
- Kiểm tra giơ tay (`python3 -m onboard.pose_check`, xe không chạy): PASS.
- Chọn bằng giơ tay hoạt động trên xe (chọn được track, thu 3 ảnh mẫu khi xoay người). Sau đó mất dấu do APPEARANCE_MISMATCH:
  đúng lỗi đã sửa ở 4.4, nhưng **bản sửa chưa được đưa lên Nano** (phần 6).

### 4.7 Video demo

- File: `docs/media/demo_guard_27-11.mp4` (19,7 MB, 85 s). Hai khung cạnh nhau trên cùng video thật: Guard cũ và Guard mới,
  có tỉ lệ bám tích lũy. Ảnh trích để chèn báo cáo: `docs/media/demo_frame_470.jpg`, `demo_frame_760.jpg`.
- Video gốc từ camera xe: `logs/pc_20261127_192720_raw.mp4`.
- **Lưu ý quyền riêng tư:** đầu video có người khác (khách ngồi bàn) lọt vào khung. Thư mục `docs/media/` đã đưa vào
  `.gitignore`. **Làm mờ mặt người lạ** trước khi đăng công khai (YouTube, GitHub), hoặc cắt bỏ đoạn đó.
- Nên quay thêm video bên ngoài (điện thoại quay xe chạy) sau lần thử sàn mới, ghép với video camera.

---

## Phần 5. Tối ưu & triển khai thiết bị thật

Hai kiến trúc chạy. Nên có hình so sánh.

| | Chạy trên PC (`demo_pc.py --driver jetbot`) | Chạy trên Nano (`onboard/run.py`) |
|---|---|---|
| AI chạy ở | PC (RTX 3050) | Jetson Nano (TensorRT FP16) |
| Cần WiFi | Có (ảnh lên, lệnh bánh xuống) | Không |
| Chọn người | Bấm chuột | Giơ hai tay, rồi xoay người 8 s để thu ảnh mẫu |
| Detector | YOLOv8s | YOLOv8n-pose 288×480 |
| FPS | 13,8–14,5 | 14,3–14,5 |

Các tối ưu cần trình bày:
1. **Băng thông:** stream 480×270, JPEG q60, ~8 KB/frame. Đây là mức tốt nhất đo được trên WiFi yếu (640×360 ~15 KB/frame).
2. **TensorRT FP16** trên Nano. Engine phải build tại chỗ (~15–20 phút). Kiểm tra parity với ONNX trên PC.
3. **An toàn ở tầng driver** (`configs/jetbot.yaml`, `drive:`):
   - Lệnh hết hạn sau 0,4 s. Watchdog trên agent.
   - Frame cũ hơn 0,35 s hoặc mất ảnh 0,5 s thì lệnh = 0.
   - Bánh tối đa 0,50. Tăng tốc từng bước 0,06 từ mép vùng chết 0,08. Phanh tức thì.
   - Xe luôn khởi động ở SAFE: phải bấm `M` mới chạy. `Q` / Ctrl+C luôn gửi `/stop`.
4. **An toàn ở tầng nhận dạng:** Guard fail-closed. Track mới vẫn cần ngưỡng chặt 0,30 và 5 frame xác nhận.
   Chỉ track đang bám liên tục mới được nới ngưỡng.
5. **Tương thích Nano:** Python 3.6 (kiểm tra `ast.parse(feature_version=(3,6))`), backport `dataclasses`,
   service systemd `jetbot-follow` tự chạy khi bật nguồn, có chế độ `--dry-run`.
6. **Hạ tầng debug:** mỗi lần chạy tự ghi `_robot.csv`, `_trace.csv`, `_dets.csv`, `_raw.mp4`. `replay_log.py` phát lại
   offline để sửa thuật toán mà không cần xe. 108 unit test (`python -m pytest -q`).

Hình gợi ý: ảnh xe, sơ đồ luồng dữ liệu của 2 kiến trúc, ảnh dashboard.

---

## Phần 6. Kế hoạch giai đoạn sau

Viết dạng bảng: việc, người phụ trách `[ĐIỀN]`, hạn `[ĐIỀN]`, tiêu chí xong.

1. **Đưa bản sửa 27/11 lên Nano** (`controller.py`, `guarded_lock.py`, `configs/jetbot.yaml`), restart service, thử sàn lại.
   Xong khi: % frame được bám trên xe thật ≥ 90 % (như phát lại offline).
2. **Tránh vật cản:** thu 100–200 ảnh mỗi lớp free/blocked trong phòng demo, train, build engine, bật kiểm tra.
   NVIDIA có model pretrained, nhưng train trên camera và phòng khác, kiến trúc chưa rõ. Chỉ dùng làm mốc so sánh.
3. **Đo lại benchmark MOT17** (mục 4.1) và đưa vào báo cáo cuối kỳ.
4. **Detector trên Nano:** thử YOLOv8s-pose TensorRT. Đổi nếu vẫn ≥ 12 FPS.
5. **Xử lý WiFi rớt** (log 27/11 có 2 lần ~0,5 s): hiện xe đã tự dừng; xem xét chạy hẳn trên Nano khi demo.
6. **Thêm kịch bản mô phỏng** từ log thật (nghiêng người, ngược sáng) để kiểm tra hồi quy.
7. **Video demo cuối kỳ** quay từ bên ngoài, làm mờ người lạ.
8. Hoàn thiện báo cáo cuối kỳ, slide.

---

## Phụ lục: lệnh tái lập nhanh

```
python -m pytest -q                                              # 108 test
python scripts/sim_follow.py                                     # mô phỏng ~10 phút
python scripts/replay_log.py logs/pc_20261127_192720 --weights yolov8s.pt
python demo_pc.py --source jetbot --driver jetbot --device cuda:0
```

Tài liệu chi tiết: `README.md` (nghiên cứu, demo PC), `docs/ONBOARD.md` (Nano, mô phỏng, nguyên nhân mất dấu),
`docs/PLAN_CODE.md` (thiết kế driver xe thật).
