# Data Input Specification — SoccerNet MOT

Tài liệu này mô tả data contract cho architecture mới:

```text
Phase 1: ảnh + ground truth → train detector multi-class
Phase 2: ảnh + output detector → tracking
```

Trong plan mới, `det.txt` không phải input chính. Đây là detection có sẵn của dataset và chỉ giữ lại để tham khảo.


## 1. Cấu trúc một sequence

Ví dụ sequence `SNMOT-060`:

```text
tracking-2023/train/train/SNMOT-060/
├── img1/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── gt/gt.txt
├── det/det.txt
├── seqinfo.ini
└── gameinfo.ini
```

Mỗi sequence là một video độc lập. `track_id=7` ở hai sequence khác nhau không được xem là cùng một người.

## 2. `img1/` — ảnh đầu vào

`img1/` chứa các frame của video:

```text
img1/000001.jpg → frame_id=1
img1/000002.jpg → frame_id=2
```

Quy tắc:

- đọc frame theo thứ tự số, không sort chuỗi đơn giản;
- giữ đúng quan hệ giữa `frame_id` và annotation;
- không trộn frame giữa các sequence;
- ảnh dùng cho train detector và inference tracker.

## 3. `seqinfo.ini` — metadata

Ví dụ:

```ini
[Sequence]
name=SNMOT-060
imDir=img1
frameRate=25
seqLength=750
imWidth=1920
imHeight=1080
imExt=.jpg
```

| Field | Ý nghĩa |
|---|---|
| `name` | Tên sequence |
| `imDir` | Thư mục frame |
| `frameRate` | Số frame/giây |
| `seqLength` | Tổng số frame |
| `imWidth` | Chiều rộng ảnh |
| `imHeight` | Chiều cao ảnh |
| `imExt` | Phần mở rộng ảnh |

`imWidth` và `imHeight` cần thiết để chuyển bbox sang tọa độ YOLO chuẩn hóa.

## 4. `gt/gt.txt` — ground truth

`gt.txt` là annotation chuẩn dùng để train và evaluate.

Format MOTChallenge:

```text
frame, track_id, bb_left, bb_top, bb_width, bb_height, mark, x_3d, y_3d, z_3d
```

Ví dụ:

```text
1,1,914,855,55,172,1,-1,-1,-1
```

| Cột | Ý nghĩa |
|---:|---|
| 1 | `frame_id` |
| 2 | `track_id` ground truth |
| 3 | x góc trái bbox |
| 4 | y góc trái bbox |
| 5 | chiều rộng bbox |
| 6 | chiều cao bbox |
| 7 | `mark`/confidence theo format MOT |
| 8–10 | tọa độ 3D, thường là `-1` |

Lưu ý: cột thứ 7 không phải class label. Loại object được bổ sung từ `gameinfo.ini`.

MOT dùng bbox dạng:

```text
x, y, width, height
```

Có thể chuyển sang `xyxy` bằng:

```text
x1 = x
y1 = y
x2 = x + width
y2 = y + height
```

## 5. `gameinfo.ini` — role của object

Ví dụ:

```ini
trackletID_1= player team left;10
trackletID_14= referee;main
trackletID_18= ball;1
trackletID_25= goalkeeper team right;X
```

Nó ánh xạ:

```text
sequence + track_id → role
```

Ví dụ:

```text
SNMOT-060 + ID 14 → referee
SNMOT-060 + ID 18 → ball
SNMOT-060 + ID 25 → goalkeeper

## 6. Multi-class cho Phase 1

Phase 1 sẽ train detector với bốn class:

```text
0 = ball
1 = player
2 = goalkeeper
3 = referee
```

Mỗi dòng trong `gt.txt` được gán class bằng cách tra `track_id` trong `gameinfo.ini`. Không gộp player, goalkeeper và referee vào một class `person`.

Lợi ích:

- detector biết role của từng object;
- tracker có thể association theo class;
- đánh giá được từng loại object;
- có thể dùng logic motion khác nhau cho người và bóng.

Chi phí:

- cần ánh xạ role chính xác cho mọi `track_id`;
- ball nhỏ và khó detect hơn;
- cần theo dõi metric riêng cho từng class.

## 6.1. Team classifier

Team được dự đoán bằng classifier riêng trên crop của object người:

```text
frame
  ↓
role detector
  ↓
person/player crop
  ↓
team classifier
  ↓
team_id
```

Team label:

```text
0 = team_left
1 = team_right
2 = neutral
```

Quy tắc:

```text
player team left       → role_id=1, team_id=0
player team right      → role_id=1, team_id=1
goalkeeper team left   → role_id=2, team_id=0
goalkeeper team right  → role_id=2, team_id=1
referee                → role_id=3, team_id=2
ball                   → role_id=0, team_id=2
```

`team_left` và `team_right` là nhãn cục bộ theo sequence/game. Không được tự động xem chúng là tên câu lạc bộ cố định trên toàn dataset nếu chưa có mapping theo `gameID`.

Detection sau Phase 1 nên có cả hai thuộc tính:

```python
{
    "bbox": [...],
    "role_id": 1,
    "team_id": 0,
    "role_confidence": 0.91,
    "team_confidence": 0.87,
}
```

## 7. Input cho Phase 1 — train detector

Phase 1 sử dụng:

```text
img1/*.jpg
gt/gt.txt
gameinfo.ini
seqinfo.ini
```

Data flow:

```text
gt.txt + gameinfo.ini
        ↓
tra role của từng track_id
        ↓
gán role_id: ball/player/goalkeeper/referee
        ↓
train role detector
        ↓
crop person/player và train team classifier
        ↓
đánh giá role + team
```

`det.txt` không dùng làm ground truth và không dùng làm label chuẩn để train YOLO.

## 8. Chuyển MOT bbox sang YOLO label

YOLO cần một file label cùng tên ảnh:

```text
images/train/000001.jpg
labels/train/000001.txt
```

Mỗi dòng YOLO có dạng:

```text
class_id center_x center_y width height
```

Với MOT bbox `x, y, w, h` và ảnh có kích thước `W × H`:

```text
center_x = (x + w / 2) / W
center_y = (y + h / 2) / H
width    = w / W
height   = h / H
```

Ví dụ:

```text
MOT:
1,1,914,855,55,172,1,-1,-1,-1
```

Với ảnh 1920×1080, class `player=1`, label YOLO xấp xỉ:

```text
1 0.4904 0.8713 0.0286 0.1593
```

`track_id` không ghi vào YOLO label. Detector chỉ học class và bounding box; tracking mới sử dụng identity.

Nếu frame không có object hợp lệ, vẫn giữ ảnh và tạo file label rỗng.

## 9. Chia train/validation/test

Không chia ngẫu nhiên từng frame vì các frame liên tiếp rất giống nhau.

Ưu tiên chia theo `gameID`; nếu không đủ metadata thì chia theo sequence:

```text
train: game/sequence dùng để train
val:   game/sequence chưa xuất hiện trong train
test:  chỉ dùng đánh giá cuối
```

Không để các frame của cùng video xuất hiện đồng thời trong train và validation.

## 10. Output của Phase 1

YOLO trả về detection cho từng frame:

```python
{
    "frame_id": 1,
    "bbox": [x1, y1, x2, y2],
    "class_id": 1,
    "confidence": 0.91,
}
```

Khi lưu để đưa vào Phase 2:

```text
frame, -1, x, y, width, height, confidence, -1, -1, -1
```

Ví dụ:

```text
1,-1,914,855,55,172,0.91,-1,-1,-1
```

Format MOT chuẩn không chứa class ID trong dòng output. Vì vậy cần giữ `class_id` trong detection/track nội bộ hoặc lưu thêm file sidecar nếu muốn đánh giá theo class.

Đặt tên detection của model là `yolo_det.txt`; không ghi đè lên `det.txt` gốc.

## 11. Input cho Phase 2 — tracking

Phase 2 nhận output của detector:

```text
img1/*.jpg
yolo_det.txt hoặc detection trong memory
seqinfo.ini
```

Pipeline:

```text
frame ảnh
   ↓
YOLO detector
   ↓
detection với class_id và confidence
   ↓
motion prediction
   ↓
class-aware data association
   ↓
track lifecycle
   ↓
track_result.txt
```

Nguyên tắc association:

```text
player       chỉ ghép với player
goalkeeper   chỉ ghép với goalkeeper
referee      chỉ ghép với referee
ball         chỉ ghép với ball
```

`gt.txt` không được đưa vào tracker trong inference thực tế. Nó chỉ dùng để đánh giá:

```text
track_result + gt.txt → HOTA, IDF1, MOTA, ID Switches
```

## 12. Phân biệt các file

| File | Nguồn | Vai trò |
|---|---|---|
| `gt/gt.txt` | annotation thật | train/evaluate |
| `det/det.txt` | detector có sẵn | bỏ qua trong plan mới |
| `yolo_det.txt` | YOLO của project | input cho Phase 2 |
| `track_result.txt` | tracker của project | output MOT cuối |

Data flow chính:

```text
gt.txt        → train/evaluate detector
yolo_det.txt  → input tracker
track_result  → evaluate tracking
```

## 13. Kiểm tra dữ liệu trước khi train

Cần kiểm tra:

- mọi `frame_id` trong `gt.txt` có ảnh tương ứng;
- `frame_id` nằm trong `1..seqLength`;
- `x`, `y`, `width`, `height` không âm;
- bbox không vượt ảnh bất hợp lý;
- mọi `track_id` cần dùng ánh xạ được tới `gameinfo.ini`;
- role được ánh xạ đúng sang class `0..3`;
- không có role không xác định bị gán class ngẫu nhiên;
- YOLO coordinates nằm trong `0..1`;
- frame không có object có label file rỗng;
- train/validation không trộn cùng game.

## 14. Data contract tối thiểu cho code

Loader sau này nên cung cấp:

```text
Sequence:
    name
    frame_rate
    sequence_length
    image_width
    image_height

Frame:
    frame_id
    image_path
    annotations

Annotation:
    track_id
    role
    class_id
    bbox_xywh
    bbox_xyxy
```

Phase 1 cần `image_path`, `class_id` và bbox. Phase 2 cần `frame_id`, bbox detection, `class_id` và confidence. `track_id` không phải target của detector.

## 15. Tóm tắt

```text
img1 + gt.txt + gameinfo.ini
        ↓
tra role và gán 4 class
        ↓
tạo YOLO labels
        ↓
train/evaluate YOLO
        ↓
YOLO sinh yolo_det.txt
        ↓
tracker nhận yolo_det.txt
        ↓
track_result.txt
        ↓
so sánh với gt.txt
```

Quy tắc cốt lõi:

> `gt.txt` là đáp án chuẩn, `yolo_det.txt` là detection của model, còn `track_result.txt` là kết quả tracking.
```
