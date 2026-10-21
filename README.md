# Abnormal Behavior Detector

Nhận một video → trả lời **có bất thường không** và **hành vi gì** (Fighting, Theft, …).

- Thiết kế gốc: [`abnormal_behavior_detection_system_design.md`](abnormal_behavior_detection_system_design.md)
- Các quyết định khi hiện thực hoá (và chỗ khác thiết kế gốc): [`docs/decisions.md`](docs/decisions.md)

```
video ─► YOLO (person) ─► ByteTrack ─► person clips ┐
   └──────────────► full-frame clips ───────────────┤
                                                    ▼
                       S3D (họ I3D, frozen) — 1 feature / segment / stream
                                                    ▼
                  attention pooling nhiều người  (+ nối frame stream)
                                                    ▼
                          Temporal Transformer (S segments)
                                 ▼                     ▼
               anomaly score / segment  ──top-k MIL──► Normal/Abnormal
                                 └─ attention ─► Behavior head ─► class
```

## Cài đặt

```bash
brew install uv          # nếu chưa có
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

## Dữ liệu

Đặt dataset tại `data/Ano-AAD` (hoặc đổi `paths.dataset_root`):

```
data/Ano-AAD/
├── normal/*.mp4
└── abnormal/<Behavior>/*.mp4
```

## Chạy pipeline

Mọi output nằm trong `outputs/` (đổi bằng `paths.work_dir`). Ghi đè config bằng
`--config my.yaml` hoặc `--set khoa.con=giatri` (lặp được, đặt **trước** tên lệnh).

```bash
abd index                       # quét dataset -> outputs/manifest.csv + thống kê
abd split                       # chia theo VIDEO, stratified -> outputs/splits.json
abd track --render 10           # YOLO + ByteTrack -> outputs/tracks/, render 10 video để kiểm tra bằng mắt
abd extract                     # feature S3D -> outputs/features/s3d/*.npz
abd --set model.input=frame  train --name baseline
abd --set model.input=person train --name person
abd --set model.input=both   train --name both
abd cv --name cv_frame --folds 5          # 5-fold CV, mean ± std (nên dùng: dataset nhỏ)
abd evaluate --ckpt outputs/runs/both/best.pt --split test
abd pseudo-label --ckpt outputs/runs/both/best.pt      # đoạn thời gian nghi bất thường (pseudo label)
abd infer --ckpt outputs/runs/both/best.pt --video x.mp4 --render x_out.mp4
```

Chỉ muốn baseline full-frame (không cần YOLO): `abd --set 'features.streams=[frame]' extract`.

Output `infer`:

```json
{"anomaly": true, "behavior": "Fighting", "confidence": 0.92, "anomaly_score": 0.95,
 "segments": [{"start_sec": 20.1, "end_sec": 35.4, "mean_score": 0.91}], ...}
```

## Metrics

`outputs/runs/<name>/metrics.json` gồm cho val và test:

- **anomaly**: precision / recall / F1 (ngưỡng chọn trên val), ROC-AUC, PR-AUC
- **behavior** (chỉ video abnormal): accuracy, macro P/R/F1, confusion matrix
- **end_to_end**: cascade như khi triển khai (Normal + C lớp)

Đoạn thời gian từ `pseudo-label` / `infer` là **pseudo label**, không phải ground truth.

## Chi phí tính toán (Apple M1, MPS)

| Bước | Ước lượng |
|---|---|
| S3D, 1 clip 16×224² | ~0.6 s |
| Frame stream, 32 segment | ~20 s / video |
| Person stream, ≤4 người | thêm tới ~80 s / video |
| Train head | vài phút |

Dataset lớn nên chạy `track` + `extract` trên GPU CUDA (Colab/cloud), copy `outputs/features` về
và train head trên máy. `features.backbone=r3d_18` nhanh hơn để debug.

## Chạy trên Google Colab (GPU)

Phần nặng (tracking + trích feature) chạy trên Colab, kết quả mang về máy.

1. Trên máy: đóng gói code (chỉ code đã commit)
   ```bash
   scripts/pack_code.sh
   ```
2. Tạo thư mục `MyDrive/abd/` trên Google Drive, upload `archive.zip` (dataset) và `dist/abd_code.zip`.
3. Mở [`notebooks/colab_pipeline.ipynb`](notebooks/colab_pipeline.ipynb) trên Colab
   (File → Upload notebook), chọn GPU T4, **Run all**. Bị ngắt thì Run all lại — video đã xong được bỏ qua.
4. Xem vài video trong `MyDrive/abd/outputs/vis/` để kiểm tra tracking.
5. Tải `MyDrive/abd/abd_results.zip` về thư mục project, rồi:
   ```bash
   scripts/import_results.sh abd_results.zip
   ```
   `outputs/` cũ được đổi tên thành `outputs.bak.<thời gian>`. Sau đó train / cv / infer trên máy như bình thường.

## Test

```bash
.venv/bin/pytest -q
```
