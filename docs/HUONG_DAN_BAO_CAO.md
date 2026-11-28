# Hướng dẫn viết báo cáo tiến độ đồ án (CS532 · JetBot bám theo người)

File này dành cho người viết báo cáo. Nó gồm:
- khung 6 phần và số trang mỗi phần;
- nội dung cần có trong từng phần;
- **số liệu đã kiểm chứng**, kèm nguồn (file log hoặc lệnh để tái lập).

Quy tắc:
- Số nào không có nguồn thì không ghi.
- `[ĐIỀN]` là thông tin chưa có, nhóm tự bổ sung.
- **Phạm vi đồ án là bám theo người.** Tránh vật cản không thuộc phạm vi, không viết vào báo cáo.

## 0. Yêu cầu hình thức

- Báo cáo nhóm, 10–15 trang, đúng 6 phần, kèm video demo.
- Trang bìa ghi: trường/khoa, môn CS532, tên đề tài, GVHD `[ĐIỀN]`, thành viên + MSSV `[ĐIỀN]`, ngày nộp.
- Hình và bảng đánh số, chú thích nguồn (tên file log, ngày đo).
- Mốc thời gian lấy từ git log và log chạy: 16/11, 17/11, 23/11, 25/11, 27/11/2026.

| Phần | Số trang |
|---|---|
| 1. Mô tả đề tài & mục tiêu | 1–1,5 |
| 2. Kiến thức / công nghệ | 2,5–3 |
| 3. Tiến độ giữa kì | 1,5–2 |
| 4. Kết quả thực nghiệm | 3–4 |
| 5. Tối ưu & triển khai thiết bị thật | 2–2,5 |
| 6. Kế hoạch giai đoạn sau | 1 |

---

## Phần 1. Mô tả đề tài & mục tiêu

**Bài toán:** robot JetBot (NVIDIA Jetson Nano 4GB, JetPack 4.5) chỉ có một camera, phải bám theo **đúng một người**
được chọn. Phải giữ đúng người khi có người khác đi ngang, bị che khuất, nghiêng người hay đổi ánh sáng.
Không chắc chắn thì robot dừng.

**Ba phần của đồ án** (xem `README.md`):
1. Nghiên cứu: đánh giá tracking + Re-ID trên MOT17 (V1–V5.1).
2. Bám người với AI trên PC: ảnh từ xe gửi về PC qua WiFi.
3. Bám người với AI chạy ngay trên Nano, không cần PC.

**Mục tiêu đo được** (viết thành danh sách):
- Chọn người bằng bấm chuột (PC) hoặc giơ hai tay (Nano).
- Không bao giờ chạy theo người khác. Không chắc thì dừng (fail-closed).
- Dừng cách người khoảng 1 m, không đâm người.
- Thời gian thực, ≥ 14 FPS.
- Mỗi lần chạy có log để debug và tái lập.

**Giới hạn:** chạy trong nhà. Xe chậm hơn người đi bộ (~0,29 m/s). Không làm tránh vật cản.

Hình gợi ý: sơ đồ khối camera → YOLO → ByteTrack → OSNet → Identity Guard → controller → driver → động cơ.

---

## Phần 2. Kiến thức / công nghệ

Mỗi dòng viết khoảng 1 đoạn: là gì, vì sao chọn, dùng ở file nào.

| Thành phần | Công nghệ | File |
|---|---|---|
| Phát hiện người | YOLO26s (PC; trước 28/11 là YOLOv8s), YOLOv8n-pose (Nano: box + 17 keypoint) | `configs/jetbot.yaml` (`pc.detector_weights`, `onboard.yolo_engine`) |
| Tracking | ByteTrack: ghép cả box điểm thấp, `track_buffer 60` | `configs/bytetrack_custom.yaml`, `onboard/bytetrack.py` |
| Re-ID | OSNet-x0.5 pretrained MSMT17, vector 512 chiều, cosine distance | `src/reid/` |
| Danh tính (nghiên cứu) | V5 Adaptive Fusion (ngoại hình + chuyển động); V5.1 gallery theo chất lượng crop | `src/identity/manager_v5.py`, `manager_v51.py` |
| Khóa mục tiêu an toàn | Identity Guard: ảnh mẫu, trạng thái TRACKING / LOST / VERIFYING / REACQUIRED / STOPPED | `src/following/guarded_lock.py` |
| Chọn bằng cử chỉ | Hai cổ tay cao hơn vai > 5 % chiều cao khung, giữ 5 frame, đúng 1 người | `onboard/selection.py` |
| Điều khiển | Controller theo pha: SETTLING, APPROACH, COAST, PAUSE, HOLD | `src/following/controller.py` |
| Giao tiếp xe | Agent HTTP :8100 trên xe: stream MJPEG, `/drive`, `/stop`, watchdog | `jetbot_agent/agent.py`, `src/following/jetbot_driver.py` |
| Tăng tốc trên Nano | TensorRT 7.1 FP16, input 288×480 | `onboard/models.py`, `onboard/build_engines.sh` |
| Mô phỏng, phát lại | Mô phỏng vòng kín; phát lại video thật qua toàn bộ pipeline | `scripts/sim_follow.py`, `scripts/replay_log.py` |

Ba ý cần giải thích rõ (hay bị hỏi):
1. **Vì sao cần Re-ID, không chỉ tracking:** ByteTrack đổi ID khi người bị che hay ra khỏi khung. Re-ID so ngoại hình
   với ảnh mẫu để nhận lại đúng người.
2. **Cosine distance không phải xác suất.** Ngưỡng chọn từ số đo thật: cùng người trung vị 0,24, người khác 0,59 (log 25/11).
3. **Fail-closed:** mất bằng chứng thì lệnh chạy bằng 0, không đoán.

Tài liệu tham khảo:
- YOLOv8 (Ultralytics).
- ByteTrack (Zhang et al., ECCV 2022).
- OSNet (Zhou et al., ICCV 2019).
- MOT17 (MOTChallenge).
- NVIDIA JetBot (github.com/NVIDIA-AI-IOT/jetbot).
- TensorRT.

---

## Phần 3. Tiến độ giữa kì

Bảng mốc:

| Ngày | Đã xong | Bằng chứng |
|---|---|---|
| 16/11 | Nghiên cứu MOT17 + demo PC (V1–V5.1, dashboard, Identity Guard, chấm trace theo ground truth) | commit `d408507` |
| 17/11 | Driver xe thật qua HTTP, agent trên xe (:8100, watchdog) | commit `3fb8c06`, `b5cd343` |
| 23/11 | PC điều khiển xe qua WiFi, 14,8 FPS | ghi chú trong `configs/jetbot.yaml` |
| 25/11 | 4 lần chạy thử trên sàn. Hiệu chỉnh controller và ngưỡng Guard. Pipeline TensorRT trên Nano đạt 14,3 FPS | `outputs/robot_0[1-4].csv`, `outputs/trace_0[1-4].csv` |
| 27/11 | Engine pose trên Nano, chọn bằng giơ tay, mô phỏng PC, đo tốc độ xe, sửa xe giật, Guard chịu được nghiêng người và ngược sáng, công cụ phát lại log | `logs/pc_20261127_*`, commit `8c828ed` |
| 28/11 | So sánh model mạnh hơn trên video thật (mục 5.3) | `scripts/replay_log.py` |

Đoạn "khó khăn và cách xử lý" (điểm cộng):
- Xe xoay quá đà (25/11): giảm `turn_to_wheel` 0,4 → 0,25, làm mượt lệnh xoay, phanh tức thì.
- Bánh có vùng chết 0,08: tăng tốc từ mép vùng chết, mỗi bước +0,06.
- Pin yếu (34 %) làm xe "nhích rồi vọt": thêm ramp, giảm tốc gần người, cảnh báo pin < 40 %.
- Cách chọn "đứng một mình giữa khung" không dùng được ở nơi đông người: đổi sang giơ hai tay.
- Nghiêng người là mất dấu (27/11): xem mục 4.4.

Cuối phần: bảng phân công thành viên `[ĐIỀN]`.

---

## Phần 4. Kết quả thực nghiệm

Mỗi bảng ghi rõ điều kiện đo.

### 4.1 Benchmark MOT17 `[ĐIỀN]`

Máy này chưa có kết quả. Chạy trên máy có dữ liệu MOT17:

```
python scripts/validate_dataset.py --strict-seven
python main.py --mode v5 --all-sequences --ablation all --output-root outputs/benchmark_final
python main.py --mode v5_1 --all-sequences --ablation all --output-root outputs/benchmark_final
```

Bảng gồm MOTA, IDF1, IDSW cho 4 ablation của V5 (appearance_only, motion_only, fixed_fusion, adaptive_fusion) và V5.1.
Các tùy chọn Guard mới (27/11) mặc định tắt, nên không ảnh hưởng benchmark.

### 4.2 Chạy thử trên sàn 25/11 (AI trên PC, xe qua WiFi)

Nguồn: `outputs/trace_0[1-4].csv`, `outputs/robot_0[1-4].csv`.

| Lần | Thời lượng | FPS | % frame được phép bám | % LOST + VERIFYING | Số lần chọn lại |
|---|---|---|---|---|---|
| 1 | 28 s | 14,5 | 80 % | 19 % | 1 |
| 2 | 57 s | 14,5 | 91 % | 9 % | 2 |
| 3 | 151 s | 14,6 | 52 % | 26 % | 7 |
| 4 | 104 s | 14,6 | 65 % | 22 % | 7 |

Nhận xét: chạy càng lâu, càng đông người thì càng hay mất dấu và phải chọn lại.

### 4.3 Hiệu chỉnh xe thật (27/11)

- Tốc độ: bánh 0,32 trong 2 s đi được ~30 cm. Ở `forward_speed` 0,21 (bánh 0,50) xe chạy ~0,29 m/s.
- Xe hơi lệch phải; vòng camera tự sửa hướng.
- Đáp ứng xoay (log 25/11): 6,25 lần chiều rộng ảnh/s trên mỗi đơn vị chênh bánh, trễ ~4 frame, tương quan 0,93–0,95.

### 4.4 Chạy với xe thật 27/11 (AI trên PC, RTX 3050)

Nguồn: `logs/pc_20261127_185728_*`, `_185940_*`, `_192720_*`.

| Lần | AI chạy trên | FPS | Xử lý 1 frame (trung vị) | Độ trễ ảnh (trung vị) |
|---|---|---|---|---|
| 18:57 | CPU (quên `--device cuda:0`) | 4,5 | 208 ms | 250 ms |
| 18:59 | GPU | 13,8 | 57 ms | 32 ms |
| 19:27 | GPU | 14,5 | 57 ms | 32 ms |

Quan sát: xe đã bám được và dừng khi người dừng. Nhưng **người nghiêng sang trái hay phải là mất dấu**.
Phân tích log từng box (`_dets.csv`) cho ra các nguyên nhân sau:

| Nguyên nhân | Dấu hiệu trong log | Cách sửa |
|---|---|---|
| Nghiêng người, đổi ánh sáng, chỉ có 1 ảnh mẫu | cùng track, khoảng cách 0,38–0,52, vượt ngưỡng 0,38 (người khác ≥ 0,44) | `continuity_distance` 0,45 chỉ cho track đang bám; `auto_views` tự thêm tối đa 3 ảnh mẫu, ảnh mới phải khớp ảnh gốc |
| Ngược sáng, khuất nửa người | detector tự tin 0,14–0,29, không có embedding | `blind_frames` 7 (~0,5 s) |
| YOLO thêm box nửa trên trùng người | Guard tưởng bị che | `drop_part_boxes` |
| Detector bỏ sót người | không có box | PC đổi sang YOLOv8s (27/11), rồi YOLO26s (28/11, mục 5.3) |

**Kết quả chính.** Phát lại video thật (`scripts/replay_log.py`, bỏ đoạn nhấc xe, cùng YOLOv8s):

| | Guard cũ | Guard mới |
|---|---|---|
| % frame được phép bám | 67,3 % | **94,9 %** |
| Số lần mất dấu | 17 | 6 |

Phải ghi rõ: đây là **phát lại offline trên video thật**, chưa phải lần chạy xe mới (xem phần 6).

### 4.5 Mô phỏng vòng kín

Chạy `python scripts/sim_follow.py`: 8 kịch bản, mỗi kịch bản 48 lượt (24 bộ tham số vật lý × 2 seed).

Các kịch bản:

| Kịch bản | Tình huống |
|---|---|
| `stand` | người đứng yên |
| `walk_away` | người đi xa dần |
| `turns` | người đi rẽ |
| `crossing` | có người khác cắt ngang |
| `crowd` | đông người |
| `lookalike` | người mặc giống mục tiêu |
| `sidestep` | mục tiêu chạy khỏi khung |
| `approach` | mục tiêu bước lại gần xe |

Một lượt bị tính trượt khi:
- xe đâm người;
- xe chạy theo người khác;
- xe tiến khi còn cách < 0,9 m;
- xe kết thúc quá gần hoặc quá xa;
- mất hẳn mục tiêu.

| Kịch bản | `guarded` (của nhóm) | `nearest` (không Re-ID, giống các repo mẫu) |
|---|---|---|
| stand | 48 | 48 |
| walk_away | 48 | 48 |
| turns | 43 | 46 |
| crossing | 48 | **10** (38 lượt chạy theo người khác) |
| crowd | 47 | **15** (có va chạm) |
| lookalike | 48 | **16** (có va chạm) |
| sidestep | 48 | 20 |
| approach | 48 | 48 |

Thông điệp chính:
- Không có Re-ID thì xe bám nhầm người và đâm người.
- `guarded` không có lượt nào chạy theo người khác hay đâm người. Các lượt trượt chỉ nằm ở góc tham số cực đoan.
- Chạy lại lệnh trước khi nộp để lấy số mới nhất.

Tối ưu tìm ra nhờ mô phỏng:
- `forward_speed` 0,12 → 0,21.
- Pha COAST: walk_away tăng từ 16/24 lên 24/24.
- Dừng khi đầu bị cắt khỏi khung, để tránh đâm người.

Giới hạn: mô phỏng không có ảnh thật, trượt bánh, pin.

### 4.6 Chạy trên Nano (không cần PC)

- Engine TensorRT FP16 build trên Nano (TRT 7.1). Kiểm tra `PARITY OK` với ONNX trên PC (`python3 -m onboard.check_engines`).
- FPS: 14,3 với YOLOv8n (25/11); 14,3–14,5 với YOLOv8n-pose (27/11).
- Kiểm tra giơ tay (`python3 -m onboard.pose_check`, xe không chạy): PASS.
- Trên xe đã chọn được người và thu được 3 ảnh mẫu. Sau đó mất dấu vì APPEARANCE_MISMATCH: chính là lỗi đã sửa ở 4.4,
  nhưng **bản sửa chưa được đưa lên Nano**.

### 4.7 Video demo

- `docs/media/demo_guard_27-11.mp4` (85 s): hai khung đặt cạnh nhau trên cùng video thật, Guard cũ và Guard mới,
  có tỉ lệ bám tích lũy. Ảnh trích để chèn báo cáo: `docs/media/demo_frame_470.jpg`, `demo_frame_760.jpg`.
- Video gốc từ camera xe: `logs/pc_20261127_192720_raw.mp4`.
- **Quyền riêng tư:** đầu video có người lạ lọt khung. `docs/media/` không được đưa lên git.
  Phải làm mờ mặt người lạ hoặc cắt đoạn đó trước khi đăng.
- Sau lần thử sàn mới, quay thêm cảnh xe chạy bằng điện thoại và ghép với video từ camera xe.

---

## Phần 5. Tối ưu & triển khai thiết bị thật

### 5.1 Hai kiến trúc chạy (nên vẽ hình)

| | AI trên PC (`demo_pc.py --driver jetbot`) | AI trên Nano (`onboard/run.py`) |
|---|---|---|
| Phần cứng AI | RTX 3050 | Jetson Nano, TensorRT FP16 |
| Cần WiFi | Có | Không |
| Chọn người | Bấm chuột | Giơ hai tay, xoay người 8 s để thu ảnh mẫu |
| Detector | YOLO26s | YOLOv8n-pose 288×480 |
| FPS | 13,8–14,5 | 14,3–14,5 |

### 5.2 Các tối ưu đã làm

1. **Băng thông:** stream 480×270, JPEG q60, ~8 KB/frame. Đây là mức tốt nhất đo được trên WiFi yếu.
2. **TensorRT FP16** trên Nano. Engine build tại chỗ (~15–20 phút), có kiểm tra parity.
3. **An toàn ở tầng driver:**
   - Lệnh hết hạn sau 0,4 s; watchdog trên agent.
   - Ảnh cũ hơn 0,35 s thì lệnh bằng 0.
   - Bánh tối đa 0,50. Tăng tốc từ mép vùng chết 0,08, mỗi bước 0,06. Phanh tức thì.
   - Xe khởi động ở SAFE. `Q` / Ctrl+C luôn gửi `/stop`.
4. **An toàn ở tầng nhận dạng:** Guard fail-closed. Track mới vẫn cần ngưỡng chặt 0,30 và 5 frame xác nhận.
5. **Tương thích Nano:** Python 3.6, service systemd tự chạy khi bật nguồn, có `--dry-run`.
6. **Hạ tầng debug:**
   - Mỗi lần chạy tự ghi log và video.
   - `replay_log.py` phát lại offline, sửa thuật toán mà không cần xe.
   - 108 unit test.

### 5.3 Thử model mạnh hơn (28/11)

Cách đo: phát lại cùng video thật 27/11, cùng Guard và controller, chỉ đổi model.
Tái lập: `python scripts/replay_log.py logs/pc_20261127_192720 --weights <model> --reid-model <model>`.

Đoạn đo: frame 17–1096 (sau lần chọn đầu, trước khi nhấc xe). PyTorch trên RTX 3050, thời gian tính cả Re-ID.
"Khoảng cách Re-ID" so với ảnh mẫu đầu tiên: mục tiêu càng thấp và người khác càng cao thì càng dễ tách.

| Detector + Re-ID | ms/frame | % frame được bám | Số lần mất dấu | Mục tiêu p50 / p90 | Người khác p5 / p50 |
|---|---|---|---|---|---|
| YOLOv8s + OSNet-x0.5 (cũ) | ~28 (lượt đầu 61,7 do khởi động CUDA) | 94,9 % | 6 | 0,30 / 0,42 | 0,56 / 0,62 |
| YOLO11s + OSNet-x0.5 | 30,2 | 92,9 % | 7 | 0,30 / 0,41 | 0,55 / 0,61 |
| **YOLO26s + OSNet-x0.5 (chọn)** | 43,0 | **96,9 %** | 6 | 0,29 / 0,41 | 0,54 / 0,60 |
| YOLO11m + OSNet-x0.5 | 49,1 | 96,5 % | 7 | 0,29 / 0,41 | 0,53 / 0,61 |
| YOLOv8s + OSNet-x1.0 | 27,9 | 94,0 % | 5 | 0,39 / 0,51 | 0,58 / 0,64 |

Nhận xét:
- YOLO26s tốt nhất, đã đặt làm mặc định cho chế độ PC (`pc.detector_weights`). Mọi model đều dưới 69 ms/frame,
  nên vẫn theo kịp stream 14,5 FPS.
- Model to hơn (YOLO11m) không hơn YOLO26s.
- OSNet-x1.0 không tốt hơn: khoảng trống giữa mục tiêu p90 và người khác p5 chỉ còn 0,07, so với 0,14 của x0.5.
  Ngưỡng Guard được chỉnh cho x0.5, nên giữ x0.5.
- Giới hạn: chỉ đo trên **một** video (~1080 frame). Chênh 2 % ≈ 22 frame. Cần đo thêm ở lần thử sàn sau.

Bối cảnh model mới (để viết phần lý thuyết):
- YOLO26 (Ultralytics, 2025–2026): bản s đạt mAP COCO 48,6, so với YOLO11s 47,0. Trên GPU dùng TensorRT, tốc độ tương đương YOLO11.
  Một nghiên cứu khác cho thấy ở cỡ n/s, YOLO11 lại tốt hơn trên dữ liệu có người.
- Re-ID: các model nhẹ mới hơn OSNet (LightMBN, CDNet) chỉ hơn ~1–2 % trên MSMT17, và không có checkpoint sẵn để dùng ngay.
  OSNet vẫn là lựa chọn phổ biến trong tracking.
- Ràng buộc của Nano: JetPack 4.5 = TensorRT 7.1, Python 3.6. Model có phép tính mới có thể không chuyển sang TensorRT được.
  Muốn đổi model trên Nano phải thử build engine trước.

---

## Phần 6. Kế hoạch giai đoạn sau

Viết dạng bảng: việc, người phụ trách `[ĐIỀN]`, hạn `[ĐIỀN]`, tiêu chí xong.

1. Đưa bản sửa 27/11 lên Nano, restart service, thử sàn lại.
   Xong khi % frame được bám trên xe thật ≥ 90 %, bằng mức phát lại offline.
2. Áp dụng model tốt nhất ở mục 5.3 cho chế độ PC. Thử build TensorRT bản tương ứng trên Nano; đổi nếu vẫn ≥ 12 FPS.
3. Chạy benchmark MOT17 (mục 4.1) cho báo cáo cuối kỳ.
4. Thêm kịch bản mô phỏng lấy từ log thật (nghiêng người, ngược sáng) để kiểm tra hồi quy.
5. Xử lý WiFi rớt (log 27/11 có 2 lần ~0,5 s). Xe đã tự dừng; khi demo nên chạy hẳn trên Nano.
6. Quay video demo cuối kỳ từ bên ngoài, làm mờ người lạ.
7. Hoàn thiện báo cáo cuối kỳ và slide.

---

## Phụ lục: lệnh tái lập

```
python -m pytest -q
python scripts/sim_follow.py
python scripts/replay_log.py logs/pc_20261127_192720 --weights yolov8s.pt
python demo_pc.py --source jetbot --driver jetbot --device cuda:0
```

Tài liệu chi tiết: `README.md`, `docs/ONBOARD.md`, `docs/PLAN_CODE.md`.
