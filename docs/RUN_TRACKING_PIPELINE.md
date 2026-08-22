# Chạy pipeline tracking trên máy GPU

Pipeline code hiện tại gồm:

```text
src/football_tracking/vision/role_tracker.py
  → YOLO role detector + BoT-SORT

src/football_tracking/vision/ball_tracker.py
  → sliced ball detector + temporal tracker

src/football_tracking/vision/team_classifier.py
  → torso color feature + team assignment

src/football_tracking/pipeline.py
  → sequence loop

scripts/run_tracking_pipeline.py
  → CLI entry point
```

## Cài đặt

Trên máy GPU:

```powershell
pip install -e ".[training]"
```

## Chạy role + ball + team trên một sequence

```powershell
& .venv\Scripts\python.exe scripts\run_tracking_pipeline.py `
  --sequence tracking-2023\train\train\SNMOT-060 `
  --role-weights outputs\runs\detection\yolo_baseline_smoke-2\weights\best.pt `
  --ball-weights outputs\runs\detection\yolo_baseline_smoke-2\weights\best.pt `
  --tracker-config configs\botsort.yaml `
  --device 0 `
  --output-dir outputs\tracking
```

Lệnh trên dùng role checkpoint cho cả ball như một smoke test. Khi có ball
checkpoint riêng, thay giá trị của `--ball-weights`.

## Output

```text
outputs/tracking/SNMOT-060/
├── tracks.jsonl       # role, team, visibility, confidence
└── track_result.txt   # MOT-style output
```

`tracks.jsonl` là output chính vì vẫn giữ `role_id`, `team_id`, ball visibility
và trạng thái prediction. `track_result.txt` không chứa class ID theo chuẩn MOT;
ball dùng ID export riêng `1000000`.

## Smoke test trên GPU

Chạy trước một sequence ngắn hoặc copy một số frame vào thư mục test riêng.
Kiểm tra:

- process không lỗi khi đọc frame;
- player/GK/referee có `track_id` ổn định;
- ball không nhảy liên tục giữa các tile;
- `tracks.jsonl` có cả role và team;
- số dòng output tương ứng với số frame đã xử lý.

## Compile không cần GPU

Có thể kiểm tra syntax bằng:

```powershell
& .venv\Scripts\python.exe -m compileall -q src scripts\run_tracking_pipeline.py
```

Compile chỉ kiểm tra syntax/import bytecode; inference thật vẫn cần Ultralytics,
OpenCV và checkpoint trên máy GPU.
