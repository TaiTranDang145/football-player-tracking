# SoccerNet Detection và Tracking Pipeline

Runbook triển khai end-to-end cho SoccerNet Tracking 2023. Pipeline dùng một
detector độ phân giải cao cho player/GK/referee và một detector ball riêng với
sliced inference.

Không dùng trực tiếp weight của repo tham khảo để báo cáo kết quả trên
SoccerNet; weight phải được train hoặc fine-tune trên dữ liệu SoccerNet.

## 1. Data contract

Role ID:

```text
0 = ball
1 = player
2 = goalkeeper
3 = referee
```

Team ID là nhãn cục bộ theo game/sequence:

```text
0 = team_left
1 = team_right
2 = neutral
```

Một sequence có dạng:

```text
tracking-2023/<split>/<group>/<sequence>/
├── img1/                 # frame đầu vào
├── gt/gt.txt             # ground truth
├── det/det.txt           # detection có sẵn, chỉ tham khảo
├── seqinfo.ini           # FPS, kích thước, số frame
└── gameinfo.ini          # track_id → role/team
```

Quy tắc cốt lõi:

- `gt.txt` dùng để train/evaluate, không dùng làm input inference.
- `det/det.txt` không phải ground truth mới của project.
- Không ghi đè `gt.txt` hoặc `det.txt`.
- `track_id` dành cho tracking, không ghi vào label YOLO.

Chi tiết format MOT xem tại [`DATA_INPUT.md`](DATA_INPUT.md).

## 2. M1 — Validate dataset

Chọn một sequence nhỏ làm smoke test trước khi chạy toàn bộ dataset, ví dụ
`SNMOT-060`.

Validator cần kiểm tra:

1. Mỗi `frame_id` trong `gt.txt` có ảnh tương ứng trong `img1/`.
2. Frame được sort theo số: `1, 2, 10`, không phải `1, 10, 2`.
3. `seqinfo.ini` khớp kích thước ảnh thực tế.
4. Bbox có `width > 0`, `height > 0` và nằm hợp lý trong ảnh.
5. Mọi `track_id` map được sang role trong `gameinfo.ini`.
6. Không có role không xác định bị gán class ngẫu nhiên.
7. `frame_id` nằm trong `1..seqLength`.

Xuất báo cáo lỗi theo sequence:

```text
sequence, frames, annotations, missing_images,
unknown_track_ids, invalid_boxes, unknown_roles
```

Nếu còn lỗi annotation chưa giải thích, dừng pipeline trước khi train.

## 3. M2 — Parse role/team từ MOT

`gt.txt` có dạng:

```text
frame, track_id, x, y, width, height, mark, x_3d, y_3d, z_3d
```

Cột `mark` không phải class ID. Role/team được lấy bằng cách kết hợp:

```text
gt.txt + gameinfo.ini
        ↓
track_id → role/team
        ↓
annotation nội bộ
```

Annotation nội bộ tối thiểu:

```python
{
    "sequence": "SNMOT-060",
    "frame_id": 1,
    "track_id": 18,
    "role": "ball",
    "class_id": 0,
    "team_id": 2,
    "bbox_xywh": [x, y, width, height],
}
```

Chuyển bbox MOT `[x, y, width, height]` sang `xyxy` bằng:

```python
x1, y1 = x, y
x2, y2 = x + width, y + height
```

## 4. M3 — Split theo game/sequence

Không chia ngẫu nhiên từng frame vì các frame liên tiếp gây leakage.

```text
train: game/sequence A, B, C, ...
val:   game/sequence chưa xuất hiện trong train
test:  game/sequence chỉ dùng ở bước báo cáo cuối
```

Nếu có `gameID`, chia theo `gameID`; nếu không, chia theo sequence. Lưu manifest
cố định:

```text
manifests/train.txt
manifests/val.txt
manifests/test.txt
```

Không thay đổi test split sau khi đã xem metric.

## 5. M4 — Tạo YOLO role labels

Mỗi ảnh có một label cùng tên:

```text
yolo/images/train/SNMOT-060_000001.jpg
yolo/labels/train/SNMOT-060_000001.txt
```

Mỗi dòng label:

```text
class_id center_x center_y width height
```

Với ảnh `W × H` và bbox `[x, y, w, h]`:

```text
center_x = (x + w / 2) / W
center_y = (y + h / 2) / H
width    = w / W
height   = h / H
```

Yêu cầu:

- clamp bbox vào biên ảnh trước khi normalize;
- loại bbox diện tích bằng 0 và ghi log;
- frame không có object vẫn có label file rỗng;
- không ghi `track_id` vào label;
- render một số frame với bbox để kiểm tra bằng mắt.

`yolo/dataset.yaml` phải giữ mapping bốn class hiện tại.

## 6. M5 — Tạo dataset ball riêng

Ball nhỏ hơn player rất nhiều. Không nên chỉ train model bốn class ở `640 × 640`
rồi kỳ vọng ball recall cao.

```text
ball_dataset/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

Quy trình:

1. Lấy frame có annotation ball.
2. Chia frame thành tile `640 × 640` có overlap.
3. Chuyển bbox ball từ tọa độ frame sang tọa độ tile.
4. Giữ tile nếu bbox ball còn hợp lệ và đủ pixel.
5. Thêm tile không có ball làm negative sample.
6. Split tile theo sequence, không split ngẫu nhiên từng tile.

Khi inference phải dùng tile tương tự lúc train, merge về frame gốc rồi mới NMS.

## 7. M6 — Train role detector

Role detector nhận player, goalkeeper và referee; có thể giữ ball như baseline.
Run chính nên dùng pretrained YOLO11s hoặc model tương đương ở `imgsz=1280`
(tối thiểu `960` nếu thiếu VRAM).

Smoke test hiện tại chỉ kiểm tra data path. Run chính không nên giới hạn ở 5
epoch. Ví dụ:

```powershell
& .venv\Scripts\python.exe scripts\detection\train_yolo_baseline.py `
  --model yolo11s.pt `
  --data yolo\dataset.yaml `
  --imgsz 1280 `
  --epochs 100 `
  --batch 4 `
  --device auto `
  --name role_yolo11s_1280
```

Nếu thiếu bộ nhớ, giảm `batch` trước khi giảm `imgsz`. Lưu cùng run: model,
seed, image size, split, config và git commit. Báo cáo metric riêng cho từng
class.

## 8. M7 — Train ball detector

Ball model dùng dataset tile, thường chỉ có một class `0 = ball`. Có thể bắt đầu
bằng YOLO11s ở `640 × 640`.

Kiểm tra riêng ball gần camera, ball xa camera, ball bị che và ball gần biên
tile. Augmentation không được làm ball biến mất hoặc giảm quá mức số pixel.
Chọn confidence threshold từ precision-recall trên validation, không copy nguyên
threshold từ repo khác.

## 9. M8 — Inference detection

Mỗi frame chạy hai nhánh:

```text
frame
 ├─→ role detector, imgsz 1280
 └─→ sliced ball detector, tile 640
```

Sau khi merge tile:

1. bỏ bbox ngoài biên hoặc không hợp lệ;
2. NMS theo class hoặc NMS riêng cho ball;
3. lưu `frame_id`, bbox, `class_id`, confidence;
4. giữ class ID trong memory hoặc sidecar file.

Output:

```text
outputs/detections/<sequence>/yolo_det.txt
outputs/detections/<sequence>/yolo_det_classes.jsonl
```

MOT format không có class ID chuẩn, nên sidecar là bắt buộc nếu muốn evaluate
theo class hoặc truyền class vào tracker.

## 10. M9 — Tune confidence/NMS

Tách rõ:

```text
model confidence → NMS threshold → tracking threshold
```

Điểm bắt đầu có thể là:

```text
role candidate conf: 0.15
role output conf:    0.25
ball output conf:    0.10–0.20
IoU/NMS:             chọn theo validation
```

Đây không phải giá trị cố định. Lập bảng thử nghiệm gồm class, confidence,
NMS IoU, precision, recall, mAP50 và mAP50-95. Với ball, ưu tiên recall và false
positive per frame.

## 11. M10 — Tracking player/GK/referee

Tracker chỉ nhận detection của model:

```text
role detections
  → lọc bbox
  → group theo class
  → motion prediction
  → class-aware association
  → track lifecycle
  → track_result
```

Association tối thiểu phải cùng role:

```text
player     ↔ player
goalkeeper ↔ goalkeeper
referee    ↔ referee
ball       ↔ ball
```

BoT-SORT phù hợp cho player/GK/referee vì hỗ trợ motion, track buffer và global
motion compensation. Tune `track_high_thresh`, `track_low_thresh`,
`new_track_thresh`, `track_buffer`, `match_thresh` và `gmc_method` trên
validation. Re-ID chỉ thêm sau khi baseline có HOTA/IDF1 để chứng minh cần.

## 12. M11 — Tracking ball

Ball không nên mặc định dùng cùng logic với người:

```text
ball detections
  → chọn candidate gần vị trí dự đoán
  → Kalman/velocity prediction khi mất ngắn hạn
  → giới hạn jump bất thường
  → reset sau N frame mất
```

Khi có nhiều candidate, ưu tiên khoảng cách tới prediction, confidence và kích
thước bbox hợp lý. Khi không có detection, đánh dấu `ball_visible=false`; không
biến prediction thành ground-truth detection.

## 13. M12 — Team classification

Team classification chạy sau role detector:

```text
player/GK bbox
  → crop torso
  → team classifier
  → temporal smoothing theo track_id
```

Có thể bắt đầu bằng LAB/K-Means như repo tham khảo, nhưng phải loại grass và
background, xử lý referee/goalkeeper riêng và ngăn team ID flip giữa các frame.
Metric team được báo cáo riêng, không gộp vào mAP detection.

## 14. M13 — Output

Object nội bộ nên có:

```python
{
    "frame_id": 1,
    "track_id": 7,
    "bbox": [x1, y1, x2, y2],
    "role_id": 1,
    "role_confidence": 0.91,
    "team_id": 0,
    "team_confidence": 0.87,
    "visible": True,
}
```

Xuất ít nhất:

```text
outputs/detections/<sequence>/yolo_det.txt
outputs/tracking/<sequence>/track_result.txt
outputs/tracking/<sequence>/tracks.jsonl
```

`track_result.txt` dùng format MOT; `tracks.jsonl` giữ thêm role, team và
confidence để debug/evaluate.

## 15. M14 — Evaluation

Đánh giá detection trước, tracking sau.

```text
model detections + ground truth → Precision / Recall / AP50 / AP50-95
track_result + gt.txt           → HOTA / IDF1 / MOTA / ID switches
```

Báo cáo riêng player, goalkeeper, referee và ball. Với ball thêm visible-ball
recall, small-ball recall, frames-with-ball-detected và false positives per frame.

Không dùng ground truth để sửa output inference. Ground truth chỉ dùng ở train,
validation và evaluation.

## 16. Thứ tự implement

```text
M1  data validator
M2  MOT + gameinfo → YOLO role labels
M3  split manifest và dataset checks
M4  role detector baseline rồi model 1280
M5  ball tile dataset và ball detector
M6  detection inference + yolo_det.txt
M7  human tracker + track_result.txt
M8  ball tracker + visibility state
M9  team classifier + temporal smoothing
M10 detection/tracking evaluation và regression tests
```

## Definition of Done

- annotation và role mapping đã được validate;
- train/val/test không leakage theo game/sequence;
- role detector có metric theo từng class;
- ball detector chạy bằng sliced inference;
- inference không đọc `gt.txt`;
- tracker chỉ nhận detection của model;
- output MOT hợp lệ và có sidecar class metadata;
- có HOTA, IDF1, MOTA và ID-switch report;
- có video debug cho player overlap, camera pan và ball bị mất;
- có thể tái chạy từ manifest, config và checkpoint.

> `gt.txt` là đáp án chuẩn; detector tạo `yolo_det.txt`; tracker nhận output
> detector; metric cuối chạy trên test split chưa dùng để tune.
