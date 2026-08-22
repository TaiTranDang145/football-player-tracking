# Football Player Tracking

Dự án xây dựng hệ thống **multi-class detection và multi-object tracking (MOT)** cho video bóng đá, hướng tới khả năng triển khai trên camera edge.

## Trạng thái hiện tại

Repository hiện chứa data contract và kiến trúc Phase 1/Phase 2. Phần code model sẽ được triển khai lại từ đầu theo roadmap bên dưới.

## Mục tiêu hệ thống

Hệ thống cần trả lời:

1. Object đang nằm ở đâu và thuộc role nào?
2. Object ở frame hiện tại có cùng identity với frame trước không?

Pipeline tổng quát:

```text
Frame
  → Role detector
  → Team classifier
  → Motion prediction
  → Class-aware data association
  → Track lifecycle
  → Track ID output
```

## Architecture

### Phase 1 — Multi-class detection và team classification

Role detector:

```text
0 = ball
1 = player
2 = goalkeeper
3 = referee
```

Team classifier:

```text
0 = team_left
1 = team_right
2 = neutral
```

Pipeline:

```text
Ảnh
  ↓
Role detector
  ↓
Crop player/goalkeeper
  ↓
Team classifier
  ↓
Detection có role_id + team_id + confidence
```

`team_left` và `team_right` là nhãn cục bộ theo sequence/game, chưa phải tên câu lạc bộ cố định trên toàn dataset.

### Phase 2 — Multi-object tracking

```text
Role/team detections
  → Kalman motion prediction
  → Class-aware association
  → Optional appearance/Re-ID
  → Track lifecycle
  → MOT result
```

Association không ghép khác role; team được dùng như tín hiệu hỗ trợ identity association.

## Dataset

Dataset SoccerNet Tracking 2023 được giữ local và bị loại khỏi Git bằng `.gitignore` vì kích thước khoảng 22 GiB.

Đặt dataset tại:

```text
tracking-2023/
```

Một sequence có cấu trúc:

```text
tracking-2023/train/train/SNMOT-060/
├── img1/
├── gt/gt.txt
├── det/det.txt
├── seqinfo.ini
└── gameinfo.ini
```

Trong architecture mới:

- `gt.txt` dùng để tạo label và đánh giá;
- `gameinfo.ini` dùng để ánh xạ `track_id` sang role/team;
- `det.txt` là detection có sẵn và không dùng làm input chính;
- output detector mới sẽ lưu riêng, ví dụ `yolo_det.txt`.

Chi tiết data contract xem tại [`docs/DATA_INPUT.md`](docs/DATA_INPUT.md).

## Training và TensorBoard

Cài các dependency cho training:

```powershell
& .venv\Scripts\python.exe -m pip install -e ".[training]"
```

Chạy baseline YOLO:

```powershell
& .venv\Scripts\python.exe scripts\detection\train_yolo_baseline.py `
  --data yolo\dataset.yaml `
  --epochs 5 `
  --device auto
```

Trong terminal khác, mở TensorBoard:

```powershell
& .venv\Scripts\tensorboard.exe --logdir outputs\runs\detection
```

Sau đó mở `http://localhost:6006`. Ultralytics sẽ ghi các đường loss train,
learning rate và metric validation vào thư mục run tương ứng.

## Data labeling

`gt.txt` theo format MOTChallenge không có `class_id` riêng. Class được suy ra bằng cách kết hợp:

```text
gt.txt + gameinfo.ini
        ↓
track_id → role/team
        ↓
YOLO labels và team-classifier labels
```

YOLO role label có dạng:

```text
class_id center_x center_y width height
```

Tọa độ được chuẩn hóa về `0..1`.

## Roadmap

### Phase 0 — Data preparation

- kiểm tra frame/annotation;
- parse `seqinfo.ini` và `gameinfo.ini`;
- tạo role/team label;
- chia train/validation/test theo game hoặc sequence.

### Phase 1 — Detection

- train role detector 4 class;
- train team classifier cho player/goalkeeper;
- đánh giá Precision, Recall, AP50, AP75, mAP;
- lưu detection output với `role_id` và `team_id`.

### Phase 2 — Tracking

- Kalman motion model;
- class-aware data association;
- track lifecycle;
- đánh giá HOTA, IDF1, MOTA và ID Switches.

### Phase 3 — Robustness

- camera motion compensation;
- appearance/Re-ID;
- long-term track recovery;
- xử lý occlusion và camera zoom/pan.

### Phase 4 — Deployment

- export ONNX;
- TensorRT FP16/INT8;
- tối ưu FPS và latency;
- tích hợp camera stream.

## Evaluation principle

Detection và tracking được đánh giá riêng:

```text
Role/team predictions + ground truth → detection metrics
Track results + ground truth         → MOT metrics
```

Không dùng ground truth làm input cho tracker trong kết quả end-to-end.

## Repository layout

```text
football-player-tracking/
├── docs/
│   └── DATA_INPUT.md
├── tracking-2023/       # local only, ignored by Git
├── .gitignore
└── README.md
```

Dataset lớn nên được quản lý bằng DVC, Git LFS hoặc object storage nếu cần chia sẻ cùng repository.
