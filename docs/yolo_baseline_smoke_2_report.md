# Báo cáo baseline YOLO: `yolo_baseline_smoke-2`

## 1. Kết luận nhanh

Pipeline train và validation đã chạy hoàn chỉnh trong 5 epoch trên GPU. Loss train giảm đều, trong khi các metric validation cải thiện nhẹ. Đây là một smoke test để xác nhận pipeline, chưa phải kết quả cuối để đánh giá chất lượng mô hình.

Kết quả tốt nhất trong lần chạy:

- Precision: **0.53678** ở epoch 4.
- Recall: **0.48034** ở epoch 4.
- mAP50: **0.46476** ở epoch 4.
- mAP50-95: **0.23109** ở epoch 5.

## 2. Cấu hình lần chạy

| Tham số | Giá trị |
|---|---|
| Model | `yolo11n.pt` pretrained |
| Task | Object detection |
| Epochs | 5 |
| Image size | 640 |
| Batch size | 4 |
| Device | GPU `0` |
| Workers | 2 |
| AMP | Bật |
| Cache | Tắt (`cache: false`) |
| Seed | 42 |
| Validation | Bật |
| Dataset classes | `ball`, `player`, `goalkeeper`, `referee` |

Dataset YAML dùng ba split tương đối:

```text
train: images/train
val: images/val
test: images/test
```

## 3. Metric theo từng epoch

| Epoch | Train box | Train cls | Train dfl | Precision | Recall | mAP50 | mAP50-95 | Val box | Val cls | Val dfl |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.52726 | 1.03797 | 0.93725 | 0.50347 | 0.44510 | 0.45313 | 0.22298 | 1.62488 | 0.94838 | 0.95397 |
| 2 | 1.40403 | 0.69786 | 0.90667 | 0.50468 | 0.46186 | 0.45270 | 0.22286 | 1.60918 | 0.87304 | 0.94841 |
| 3 | 1.35014 | 0.65237 | 0.89418 | 0.52849 | 0.46500 | 0.45950 | 0.23054 | 1.61548 | 0.84613 | 0.94599 |
| 4 | 1.30155 | 0.62336 | 0.88598 | 0.53678 | 0.48034 | 0.46476 | 0.23033 | 1.60876 | 0.82214 | 0.95626 |
| 5 | 1.25353 | 0.59524 | 0.87660 | 0.53561 | 0.47687 | 0.46399 | 0.23109 | 1.61955 | 0.82736 | 0.95055 |

## 4. Phân tích loss

### Loss train

- `train/box_loss`: giảm từ **1.52726** xuống **1.25353**, giảm khoảng **17.9%**.
- `train/cls_loss`: giảm từ **1.03797** xuống **0.59524**, giảm khoảng **42.6%**.
- `train/dfl_loss`: giảm từ **0.93725** xuống **0.87660**, giảm khoảng **6.5%**.

Ba thành phần loss train đều có xu hướng giảm. Điều này cho thấy mô hình đã bắt đầu học từ dữ liệu và pipeline forward/backward/optimizer hoạt động.

### Loss validation

- `val/box_loss` dao động quanh **1.61**, gần như chưa giảm rõ rệt.
- `val/cls_loss` giảm từ **0.94838** xuống **0.82736**.
- `val/dfl_loss` dao động quanh **0.95**.

Train loss giảm nhanh hơn validation loss. Đây mới là dấu hiệu ban đầu của khoảng cách train-validation, nhưng 5 epoch chưa đủ để kết luận overfitting.

## 5. Phân tích metric

So với epoch 1, đến epoch 5:

- Precision tăng từ **0.50347** lên **0.53561**.
- Recall tăng từ **0.44510** lên **0.47687**.
- mAP50 tăng từ **0.45313** lên **0.46399**.
- mAP50-95 tăng từ **0.22298** lên **0.23109**.

Metric tăng nhẹ và tương đối ổn định. Epoch 4 có mAP50, precision và recall cao nhất; epoch 5 có mAP50-95 cao nhất. Sự khác biệt nhỏ giữa epoch 4 và 5 cho thấy mô hình chưa có bước nhảy lớn trong thời lượng smoke test này.

## 6. Thời gian train

Tổng thời gian ghi nhận ở epoch 5 là **1186.29 giây**, tương đương khoảng **19 phút 46 giây**. Tốc độ trung bình khoảng **237 giây/epoch**, tức gần **4 phút/epoch**.

## 7. Các artifact đã sinh ra

Thư mục kết quả:

```text
outputs/runs/detection/yolo_baseline_smoke-2/
```

Các file chính:

- `results.csv`: metric và loss theo epoch.
- `results.png`: biểu đồ tổng hợp loss/metric.
- `BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png`, `BoxPR_curve.png`: các đường cong đánh giá detection.
- `confusion_matrix.png` và `confusion_matrix_normalized.png`: confusion matrix.
- `labels.jpg`: phân bố nhãn trong dataset được Ultralytics đọc.
- `train_batch0.jpg` đến `train_batch2.jpg`: ảnh batch train sau augmentation.
- `val_batch*_labels.jpg`: ground truth của validation batch.
- `val_batch*_pred.jpg`: prediction của validation batch.

## 8. Đánh giá và bước tiếp theo

Baseline đạt mục tiêu chính của smoke test: kiểm tra model, dataset YAML, GPU, training loop, validation và việc sinh artifact. Kết quả hiện tại chưa nên dùng để kết luận mô hình đã đủ tốt cho tracking hoặc phát hiện bóng trong thực tế.

Các bước nên làm tiếp:

1. Kiểm tra trực quan `val_batch*_labels.jpg` và `val_batch*_pred.jpg`, đặc biệt với các cầu thủ nhỏ và quả bóng.
2. Chạy training dài hơn, ví dụ 50–100 epoch, với cùng seed để có đường học ổn định hơn.
3. Đánh giá metric theo từng class để biết class nào đang kéo mAP xuống.
4. Với quả bóng, thử thêm một cấu hình `imgsz` lớn hơn như 960 hoặc 1280 vì bóng thường chiếm rất ít pixel.
5. Chỉ sau khi detection ổn định mới đưa kết quả bounding box sang bước phân loại team và tracking.

## 9. Nguồn số liệu

Báo cáo này được tổng hợp từ:

- `outputs/runs/detection/yolo_baseline_smoke-2/results.csv`
- `outputs/runs/detection/yolo_baseline_smoke-2/args.yaml`
- `yolo/dataset.yaml`
- `scripts/detection/train_yolo_baseline.py`
