# CS532 · JetBot Person Following with Identity Re-Identification

**Bản chốt cho đồ án:** YOLOv8n + ByteTrack + **OSNet-x0.5** + **V5 Adaptive Fusion** + Identity Guard v2.4.

Đồ án gồm hai phần độc lập:

- **Nghiên cứu:** đánh giá detection/tracking/Re-ID trên MOT17 với các baseline V1–V5.1 và 4 ablation của V5.
- **Demo PC:** click chọn một người, giữ mẫu ngoại hình gốc, dừng khi danh tính không rõ, xác minh trước khi tái bám, ghi video và trace để đánh giá với MOT17 GT.

> **Giới hạn:** đây là bản **PC-only**, tất cả lệnh tốc độ gửi tới `MockRobotDriver`. Chưa có motor driver, nhận biết vật cản hay chứng nhận an toàn cho robot thật. Không nối lệnh vận tốc vào động cơ JetBot trước khi bổ sung watchdog, giới hạn tốc độ và kiểm thử vật cản.

## 1. Cấu trúc

```text
JetBot-Project/
├── configs/
│   ├── config.yaml              # Cấu hình thuật toán và benchmark
│   └── bytetrack_custom.yaml
├── src/
│   ├── identity/                # Baseline V1–V4, V5, V5.1
│   ├── reid/                    # OSNet, chất lượng crop, chọn checkpoint
│   ├── following/               # Target lock, Identity Guard, controller, mock driver
│   ├── runtime/                 # Inference adapter, compact overlay và dashboard demo
│   ├── evaluation/              # MOT17, target-centric, follow-trace metrics
│   └── data/                    # Đọc MOT17
├── scripts/
│   ├── download_reid_weights.py
│   ├── validate_dataset.py
│   └── evaluate_follow_trace.py
├── main.py                     # Benchmark trên MOT17
├── demo_pc.py                  # Webcam/video, click target và mock follow
├── yolov8n.pt                  # YOLO detector pretrained
├── requirements.txt
└── requirements-reid.txt
```

Dữ liệu, checkpoint Re-ID và file kết quả **không nằm trong gói source**. Giữ nguyên các thư mục đã có trên máy:

```text
data/datasets/MOT17/train/MOT17-11-FRCNN/{img1,gt,seqinfo.ini}
data/videos/MOT17-05-FRCNN-raw.mp4
data/videos/MOT17-11-FRCNN-raw.mp4
checkpoints/osnet_x0_5_msmt17.pth.tar
outputs/
```

## 2. Khởi động trên Windows PowerShell

Mở terminal tại thư mục gốc project (ví dụ `C:\JetBot-Project`):

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu chưa có môi trường thì cài Python/PyTorch phù hợp với GPU trước, sau đó:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-reid.txt
python scripts/download_reid_weights.py --model osnet_x0_5_msmt17
```

Torchreid dùng checkpoint **MSMT17** đúng với kiến trúc OSNet-x0.5; không được nạp checkpoint của x1.0 vào x0.5. Máy hiện tại cần PyTorch CUDA hoạt động nếu chạy `--device cuda:0`.

## 3. Demo V5 Adaptive Fusion

**MOT17-05:**

```powershell
python demo_pc.py --mode v5 --ablation adaptive_fusion --device cuda:0 --source data/videos/MOT17-05-FRCNN-raw.mp4 --record-video outputs/demo_05.mp4
```

**MOT17-11:**

```powershell
python demo_pc.py --mode v5 --ablation adaptive_fusion --device cuda:0 --source data/videos/MOT17-11-FRCNN-raw.mp4 --record-video outputs/demo_11.mp4 --trace-csv outputs/trace_11.csv
```

**Webcam:**

```powershell
python demo_pc.py --mode v5 --ablation adaptive_fusion --device cuda:0 --source 0
```

### Giao diện theo dõi thuật toán (mặc định)

Bản demo sử dụng **JetBot Identity Lab Dashboard** với hai chế độ hiển thị cùng một luồng inference:

- **Presentation** (mặc định): camera, ID đã khóa, ảnh mẫu lúc click, trạng thái TRACKING/LOST/VERIFYING/REACQUIRED, lệnh MOVE/HOLD và lịch sử thay đổi trạng thái.
- **Research**: cosine **distance** tới embedding gốc (càng thấp càng tốt), competitor margin, bounding-box overlap, ngưỡng Guard, trusted/candidate ByteTrack ID, tiến trình xác minh, FPS xử lý và lệnh điều khiển giả lập.

**Click** vào người để khóa; **Tab/D** chuyển chế độ; **R** bỏ khóa; **Space** dừng khẩn cấp; **Q/Esc** thoát. Trạng thái HOLD không cho phép điều khiển giả lập di chuyển. Dashboard chỉ đọc kết quả hiện có, không chạy thêm Re-ID và không thay đổi quyết định thuật toán.

Để vào thẳng Research Mode và lưu video có cả panel:

```powershell
python demo_pc.py --mode v5 --ablation adaptive_fusion --device cuda:0 --source data/videos/MOT17-11-FRCNN-raw.mp4 --ui dashboard --dashboard-view research --record-video outputs/demo_research.mp4 --trace-csv outputs/trace_research.csv
```

Chế độ overlay cũ (nhẹ hơn, cùng thuật toán):

```powershell
python demo_pc.py --mode v5 --device cuda:0 --source data/videos/MOT17-11-FRCNN-raw.mp4 --ui compact
```

Giới hạn cửa sổ hiển thị bằng `--display-width 1600 --display-height 900` (mặc định). Video được ghi theo kích thước dashboard gốc; hệ thống tự quy đổi tọa độ click từ cửa sổ thu nhỏ về frame nguồn. **Cosine distance không phải xác suất danh tính** và chỉ hiển thị khi có embedding hợp lệ. Các metric MOT17 GT vẫn cần chấm riêng bằng trace/benchmark, không hiển thị như độ chính xác live.


Identity Guard lưu embedding lúc chọn người và không thay mẫu gốc bằng các quan sát sau đó. Khi người bị che khuất, khó phân biệt hoặc mô hình không đưa ra bằng chứng mới, trạng thái chuyển `LOST/VERIFYING` và **lệnh chuyển động bằng 0**. Sau khi có nhiều quan sát rõ ràng mới được phép `REACQUIRED`.

Lưu ý: bằng chứng embedding chỉ mang tính xác suất; người mặc đồ giống nhau vẫn có thể gây nhầm ID. Các ngưỡng Guard đang là baseline đã chọn, **không bảo đảm** kết quả trên mọi tình huống.

## 4. Đánh giá bám theo mục tiêu bằng Ground Truth

Dùng chuỗi ảnh MOT17 để đồng bộ frame với `gt.txt`:

```powershell
python demo_pc.py --mode v5 --ablation adaptive_fusion --device cuda:0 --source "data/datasets/MOT17/train/MOT17-11-FRCNN/img1/%06d.jpg" --gt-sequence MOT17-11-FRCNN --trace-csv outputs/trace_11_gt.csv
python scripts/evaluate_follow_trace.py --trace outputs/trace_11_gt.csv --sequence MOT17-11-FRCNN
```

Chỉ chấm phần **sau khi click chọn mục tiêu**. File `outputs/trace_11_gt.metrics.json` chứa correct/wrong authorization, HOLD khi GT target còn hiện diện, recovery latency. Đây là **PC target-follow diagnostic**, không phải MOTChallenge official evaluation. `gt.txt` chỉ dùng để chấm, **không truyền vào thuật toán theo dõi**.

## 5. Benchmark nghiên cứu trên 7 sequence MOT17

Kiểm tra dữ liệu đã có:

```powershell
python scripts/validate_dataset.py --strict-seven
```

Chạy V5 x0.5 trên toàn bộ tập train:

```powershell
python main.py --mode v5 --all-sequences --ablation all --output-root outputs/benchmark_final
```

So sánh với V5.1 (nghiên cứu Quality-aware Gallery):

```powershell
python main.py --mode v5_1 --all-sequences --ablation all --output-root outputs/benchmark_final
```

V1–V4/Baseline vẫn có thể chạy để tái lập thí nghiệm lịch sử (ví dụ `python main.py --mode v3 --all-sequences`). Riêng OSNet-x1.0 là **đối chứng nghiên cứu tùy chọn**, không phải mô hình được chọn cho demo:

```powershell
python main.py --mode v5 --all-sequences --ablation adaptive_fusion --reid-model osnet_x1_0 --output-root outputs/benchmark_x1_0
```

Muốn chạy x1.0 phải tải checkpoint tương ứng bằng `scripts/download_reid_weights.py --model osnet_x1_0_msmt17`.

> **So sánh công bằng:** dùng cùng detection/tracker settings, GT, tập sequence, thời điểm chọn mục tiêu và evaluation protocol. Bộ mặc định trong `configs/config.yaml` giữ detector confidence **0.35** để tái lập benchmark trước; `demo_pc.py` mặc định **0.10** để ByteTrack nhận low-confidence detections. Hai cấu hình này **không phải cùng một phép thử**. FPS của bốn ablation chia sẻ một lượt inference, không phải FPS của robot thật.

## 6. Khi có JetBot của trường

Tách riêng `MockRobotDriver` khỏi mã perception/decision. Trước khi thay bằng driver thật, xác nhận Jetson/JetPack/camera/motor controller và kiểm tra FPS end-to-end, độ trễ, kết nối, watchdog, stop khi mất hình ảnh và vật cản. Không chạy chương trình này trên JetBot chỉ bằng việc thay tên driver.
