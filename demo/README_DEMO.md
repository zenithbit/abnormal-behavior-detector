# ABD demo (Windows + NVIDIA GPU)

## Cài đặt (một lần, cần internet, ~5–10 phút)

Double-click **`setup.bat`**. Cuối cùng phải thấy dòng `CUDA: True`.
Nếu thấy `CPU ONLY`: cập nhật driver NVIDIA (cần driver hỗ trợ CUDA 12.4, tức ≥ 551) rồi chạy lại.

## Demo

- **Kéo-thả** một file video lên **`demo.bat`**, hoặc: `demo.bat samples\inputs\<video>.mp4`
- Kết quả: `results\<tên>_result.mp4` (tự mở) và `results\<tên>_result.json`
- Không cần internet khi demo (trọng số S3D + YOLO nằm trong `weights\`).

Video có:
- khung người + ID (YOLO + ByteTrack),
- dòng trên cùng: hành vi dự đoán + điểm bất thường của đoạn hiện tại (đỏ = vượt ngưỡng, xanh = bình thường).

JSON:

```json
{"anomaly": true, "behavior": "Fighting", "confidence": 0.81, "anomaly_score": 0.93,
 "behavior_probs": {...}, "segments": [{"start_sec": 20.1, "end_sec": 35.4, "mean_score": 0.91}]}
```

## Dự phòng

`samples\rendered\` có sẵn video + JSON đã chạy trước (video thuộc **tập test**, model chưa thấy khi train).
Nếu cài đặt trục trặc, trình chiếu thẳng các file này.

## Nói gì khi demo

- Pipeline: YOLO tìm người → tracking → S3D (họ I3D) trích đặc trưng không gian-thời gian →
  Temporal Transformer → (1) bất thường hay không, (2) loại hành vi, (3) đoạn thời gian nghi vấn.
- Đoạn thời gian là **ước lượng từ model** (dataset chỉ có nhãn mức video), không phải ground truth.
- Số liệu đánh giá: xem `models\metrics.json` / `models\cv_summary.json`.
