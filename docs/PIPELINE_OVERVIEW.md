# Football Tracking Pipeline Overview

Tài liệu này mô tả pipeline tổng quát của hệ thống phát hiện, phân loại team
và tracking cầu thủ bóng đá. Đây là kiến trúc cấp cao; chi tiết về model,
hyperparameter, format file và implementation sẽ được đặc tả ở các tài liệu
riêng sau.

## 1. Mục tiêu hệ thống

Với mỗi frame video, hệ thống cần xác định:

1. Object nằm ở đâu trong ảnh;
2. Object thuộc role nào;
3. Cầu thủ thuộc team nào;
4. Object ở frame hiện tại có cùng track với frame trước hay không.

Pipeline tổng quát:

```text
Video / Frame
     │
     ▼
Data preparation and validation
     │
     ▼
Role detector
     │  ball / player / goalkeeper / referee
     ▼
Detection post-processing
     │
     ▼
Team classifier for player and goalkeeper
     │  team_left / team_right / neutral
     ▼
Enriched detections
     │
     ▼
Multi-object tracker
     │
     ▼
Track results
     │
     ▼
Evaluation and deployment
```

## 2. Các giai đoạn chính

### Giai đoạn A — Data preparation

Input:

```text
img1/*.jpg
gt/gt.txt
gameinfo.ini
seqinfo.ini
```

Nhiệm vụ:

- đọc metadata của sequence;
- ánh xạ `track_id` sang role và team;
- kiểm tra frame, bbox và annotation;
- loại hoặc báo lỗi dữ liệu không hợp lệ;
- chia train/validation/test theo game hoặc sequence;
- sinh label cho role detector;
- lưu manifest để truy vết annotation về frame gốc.

Nguyên tắc quan trọng:

- không chia ngẫu nhiên từng frame nếu các frame thuộc cùng video;
- không để cùng một game xuất hiện ở nhiều split khi đánh giá generalization;
- không dùng `det.txt` làm ground truth cho detector của project;
- giữ riêng dữ liệu gốc, dữ liệu đã chuẩn hóa và output của model.

### Giai đoạn B — Role detection

Detector nhận một frame và trả về các object độc lập:

```text
0 = ball
1 = player
2 = goalkeeper
3 = referee
```

Output tối thiểu của detector:

```python
{
    "frame_id": 1,
    "bbox": [x1, y1, x2, y2],
    "role_id": 1,
    "role_confidence": 0.91,
}
```

Role detector chỉ chịu trách nhiệm trả lời:

> Có object nào trong frame, nằm ở đâu và thuộc role nào?

Nó chưa chịu trách nhiệm nhận diện team hoặc identity cá nhân.

### Giai đoạn C — Detection post-processing

Sau detector cần có các bước hậu xử lý chung:

- confidence threshold;
- non-maximum suppression;
- loại bbox quá nhỏ hoặc không hợp lệ;
- giới hạn bbox trong kích thước ảnh;
- giữ lại `frame_id`, `role_id` và confidence;
- chuẩn hóa format để truyền sang team classifier và tracker.

Đây cũng là nơi có thể thêm các rule riêng cho ball, player, goalkeeper và
referee nếu cần.

### Giai đoạn D — Team classification

Chỉ áp dụng cho:

```text
role_id = player hoặc goalkeeper
```

Luồng xử lý:

```text
role detection
     │
     ▼
player/goalkeeper bbox crop
     │
     ▼
team feature/classifier
     │
     ▼
team_id và team_confidence
```

Team label:

```text
0 = team_left
1 = team_right
2 = neutral
```

`ball` và `referee` mặc định có `team_id=neutral`.

`team_left` và `team_right` là nhãn cục bộ theo game/sequence. Không được
coi chúng là tên câu lạc bộ cố định trên toàn dataset nếu chưa có mapping
riêng.

Output sau khi enrich:

```python
{
    "frame_id": 1,
    "bbox": [x1, y1, x2, y2],
    "role_id": 1,
    "team_id": 0,
    "role_confidence": 0.91,
    "team_confidence": 0.87,
}
```

Team classifier có thể được phát triển độc lập với detector. Khi benchmark,
cần tách hai câu hỏi:

```text
GT bbox → team classifier
detector bbox → team classifier
```

Thí nghiệm đầu tiên đo riêng chất lượng phân loại team; thí nghiệm thứ hai
đo hiệu năng thực tế của pipeline.

### Giai đoạn E — Multi-object tracking

Tracker nhận enriched detections theo từng frame và duy trì các track qua
thời gian.

Luồng cơ bản:

```text
Current detections
     │
     ▼
Motion prediction
     │
     ▼
Candidate matching
     │
     ▼
Class/team-aware association
     │
     ▼
Track lifecycle management
     │
     ▼
Active tracks
```

Tracker có thể sử dụng:

- vị trí và IoU của bbox;
- vận tốc dự đoán;
- role compatibility;
- team compatibility;
- appearance/Re-ID embedding;
- visibility và thời gian mất track.

Nguyên tắc association cơ bản:

- không ghép khác role nếu không có lý do rõ ràng;
- team chỉ là tín hiệu hỗ trợ, không nên là điều kiện duy nhất;
- `track_id` do tracker sinh ra, không phải output của detector;
- không dùng ground truth làm input cho tracker trong inference.

Output track tối thiểu:

```python
{
    "frame_id": 1,
    "track_id": 12,
    "bbox": [x1, y1, x2, y2],
    "role_id": 1,
    "team_id": 0,
    "confidence": 0.88,
}
```

## 3. Phân tách training, inference và evaluation

### Training flow

```text
Raw annotations
     ▼
Validated labels and split
     ▼
Train role detector
     ▼
Train/evaluate team classifier
     ▼
Train/tune tracker
```

### Inference flow

```text
Video frame
     ▼
Role detector
     ▼
Post-processing
     ▼
Team classifier
     ▼
Tracker
     ▼
Track output
```

### Evaluation flow

Các module được đánh giá riêng trước khi đánh giá end-to-end:

```text
Role predictions + ground truth
     ▼
Detection metrics

Team predictions + ground truth
     ▼
Team classification metrics

Track output + ground truth
     ▼
MOT metrics
```

Metric dự kiến:

- detector: Precision, Recall, AP50, AP75, mAP50-95;
- team classifier: Accuracy, Macro-F1, confusion matrix;
- tracker: HOTA, IDF1, MOTA, ID switches;
- deployment: FPS, latency, memory và kích thước model.

## 4. Các data contract chính

### Annotation từ ground truth

```python
{
    "sequence_name": "SNMOT-060",
    "frame_id": 1,
    "track_id": 7,
    "bbox_xywh": [x, y, width, height],
    "role_id": 1,
    "team_id": 0,
}
```

### Detection sau role detector

```python
{
    "frame_id": 1,
    "bbox_xyxy": [x1, y1, x2, y2],
    "role_id": 1,
    "role_confidence": 0.91,
}
```

### Detection sau team classifier

```python
{
    "frame_id": 1,
    "bbox_xyxy": [x1, y1, x2, y2],
    "role_id": 1,
    "team_id": 0,
    "role_confidence": 0.91,
    "team_confidence": 0.87,
}
```

### Track output

```python
{
    "frame_id": 1,
    "track_id": 12,
    "bbox_xyxy": [x1, y1, x2, y2],
    "role_id": 1,
    "team_id": 0,
}
```

## 5. Các nguyên tắc thiết kế

1. Mỗi module có một trách nhiệm chính: detector phát hiện, classifier phân
   loại team, tracker duy trì identity.
2. Ground truth chỉ dùng cho training và evaluation, không dùng trong
   inference.
3. Mọi output cần truy ngược được về sequence, frame và bbox gốc khi debug.
4. Role/team ID phải được giữ nhất quán giữa các module.
5. Có thể thay detector, team classifier hoặc tracker mà không phá vỡ
   interface của các module khác.
6. Đánh giá offline phải tách lỗi detection, lỗi team classification và lỗi
   association.
7. Tối ưu deployment chỉ thực hiện sau khi chất lượng offline ổn định.

## 6. Thứ tự triển khai tổng quát

```text
1. Data contract và validation
2. Dataset/label exporter cho role detection
3. YOLO role detector baseline
4. Detection evaluation và visual QA
5. Team classification trên detection crop
6. Enriched detection interface
7. Baseline tracker
8. Tracking evaluation
9. Appearance/Re-ID cải tiến
10. ONNX/TensorRT và tối ưu edge deployment
```

Các chi tiết về từng bước sẽ được tách thành plan và tài liệu riêng sau khi
kiến trúc tổng quát này được chốt.

