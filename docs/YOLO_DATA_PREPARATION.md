# YOLO Data Preparation Report

## 1. Mục đích

Role detector của Phase 1 sử dụng YOLO để phát hiện các object trong từng
frame bóng đá. Vì vậy dữ liệu raw của SoccerNet cần được chuyển sang format
YOLO trước khi training.

Báo cáo này mô tả format raw, format YOLO và quá trình chuyển đổi dữ liệu.

## 2. Format dữ liệu raw hiện tại

Dataset đang được lưu tại:

```text
tracking-2023/train/train/
```

Mỗi sequence có cấu trúc:

```text
SNMOT-060/
├── img1/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── gt/
│   └── gt.txt
├── det/
│   └── det.txt
├── seqinfo.ini
└── gameinfo.ini
```

### 2.1. Ảnh frame

Ảnh nằm trong `img1/` và được đánh số theo frame:

```text
img1/000001.jpg → frame_id = 1
img1/000002.jpg → frame_id = 2
```

Thông tin kích thước ảnh và số frame được đọc từ `seqinfo.ini`:

```ini
[Sequence]
name=SNMOT-060
imDir=img1
seqLength=750
imWidth=1920
imHeight=1080
imExt=.jpg
```

### 2.2. Ground truth bbox

`gt/gt.txt` sử dụng format MOTChallenge:

```text
frame, track_id, bb_left, bb_top, bb_width, bb_height, mark, ...
```

Ví dụ:

```text
1,1,914,855,55,172,1,-1,-1,-1
```

Format MOT sử dụng bbox dạng:

```text
x, y, width, height
```

Trong khi YOLO sử dụng bbox dạng tâm và kích thước chuẩn hóa.

### 2.3. Role của object

`gt.txt` không chứa trực tiếp class role. Role được suy ra từ
`gameinfo.ini` thông qua `track_id`:

```text
track_id + gameinfo.ini → role
```

Ví dụ:

```ini
trackletID_1= player team left;10
trackletID_14= referee;main
trackletID_18= ball;1
trackletID_25= goalkeeper team right;X
```

Role mapping của detector:

```text
0 = ball
1 = player
2 = goalkeeper
3 = referee
```

Tracklet có role `other` không được đưa vào 4 class YOLO này.

## 3. Format YOLO yêu cầu

YOLO yêu cầu mỗi ảnh có một file label `.txt` cùng tên:

```text
images/train/SNMOT-060_000001.jpg
labels/train/SNMOT-060_000001.txt
```

Mỗi dòng trong label có format:

```text
class_id center_x center_y width height
```

Tất cả tọa độ phải được chuẩn hóa về khoảng `[0, 1]`.

Ví dụ:

```text
1 0.49036458 0.87129630 0.02864583 0.15925926
```

Trong đó `1` là class `player`, các giá trị còn lại là tọa độ bbox đã chuẩn
hóa theo kích thước ảnh.

Với bbox raw:

```text
x = 914
y = 855
w = 55
h = 172
```

và ảnh kích thước `1920 × 1080`:

```text
center_x = (x + w / 2) / 1920
center_y = (y + h / 2) / 1080
width    = w / 1920
height   = h / 1080
```

## 4. Dataset YOLO sau khi chuyển đổi

Dataset YOLO được tạo tại:

```text
yolo/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── dataset.yaml
└── export_summary.json
```

`dataset.yaml` định nghĩa:

```yaml
path: .
train: images/train
val: images/val
test: images/test
names:
  0: ball
  1: player
  2: goalkeeper
  3: referee
```

## 5. Cách chuyển đổi đã thực hiện

Pipeline chuyển đổi:

```text
tracking-2023/train/train/
        │
        ├── img1/*.jpg
        ├── gt/gt.txt
        ├── gameinfo.ini
        └── seqinfo.ini
        │
        ▼
Đọc bbox và role
        │
        ▼
Chuyển MOT bbox sang YOLO bbox
        │
        ▼
Chia sequence theo gameID
        │
        ▼
yolo/images + yolo/labels
```

Ảnh raw không bị di chuyển hoặc chỉnh sửa. Ảnh trong `yolo/images/` được tạo
bằng hardlink, tức là dùng chung nội dung ảnh với raw thay vì copy thêm toàn
bộ dữ liệu.

## 6. Quy tắc chia train/validation/test

Không chia ngẫu nhiên từng frame vì các frame liên tiếp rất giống nhau.

Toàn bộ sequence thuộc cùng một `gameID` được giữ trong cùng một split:

```text
gameID group
     ↓
train hoặc val hoặc test
```

Kết quả trên `tracking-2023/train/train/`:

| Split | Số frame |
|---|---:|
| train | 14.250 |
| val | 13.500 |
| test | 15.000 |
| Tổng | 42.750 |

## 7. Kết quả thống kê

Tổng số object theo split:

| Class | Train | Val | Test |
|---|---:|---:|---:|
| ball | 13.845 | 12.599 | 14.487 |
| player | 194.920 | 212.984 | 197.067 |
| goalkeeper | 8.501 | 6.881 | 8.725 |
| referee | 19.176 | 20.395 | 21.975 |

Tổng cộng có 57 sequence và 42.750 frame được chuyển đổi.

## 8. Annotation lỗi

Validator phát hiện một annotation có bbox không hợp lệ:

```text
Sequence: SNMOT-065
File: gt/gt.txt
Line: 1285
Lỗi: bbox width = 0
```

Annotation này được bỏ qua khi export YOLO. File raw không bị sửa.

```text
skipped_invalid_annotations = 1
```

## 9. Các file chịu trách nhiệm chuyển đổi

Validator raw:

```text
src/football_tracking/data/validator.py
scripts/data/validate_dataset.py
```

YOLO exporter:

```text
src/football_tracking/data/yolo_exporter.py
scripts/data/export_yolo.py
```

Dataset có thể được tạo lại bằng:

```powershell
python scripts/data/export_yolo.py `
  --source tracking-2023/train/train `
  --output yolo `
  --image-mode hardlink
```

## 10. Trạng thái hiện tại

Đã hoàn thành:

- kiểm tra raw dataset;
- xác định role mapping;
- chuyển MOT annotation sang YOLO annotation;
- tạo cấu trúc `images/` và `labels/`;
- chia train/val/test theo game;
- tạo `dataset.yaml`;
- xử lý một bbox lỗi mà không thay đổi raw.

Chưa thực hiện:

- kiểm tra trực quan bbox bằng ảnh preview;
- train YOLO detector;
- đánh giá mAP, Precision và Recall;
- crop detection để chuẩn bị team classification.

