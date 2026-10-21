# Quyết định kỹ thuật (ADR) — V1

Tài liệu này ghi lại các điểm đã chốt khi chuyển từ bản thiết kế
(`abnormal_behavior_detection_system_design.md`) sang code, và lý do.

## D1. Baseline full-frame trước, person-stream sau

- Pipeline hỗ trợ 3 chế độ đầu vào cho head: `frame`, `person`, `both`.
- Chạy `frame` trước làm baseline. YOLO + tracking chỉ được giữ nếu `person`/`both`
  thắng baseline trên tập validation.
- Lý do: kiểm chứng data pipeline độc lập với detector; có số liệu để biết YOLO
  có thực sự giúp ích hay không.

## D2. Supervised + MIL thay cho "Normal-first" ở V1

- Mỗi video được chia thành `S` segment (mặc định 32). Mỗi segment → 1 feature.
- Temporal Transformer xử lý chuỗi `S` segment → điểm anomaly cho **từng segment**.
- Điểm video = trung bình top-k điểm segment (Multiple Instance Learning, kiểu
  Sultani et al. 2018). Loss BCE ở mức video.
- Lợi ích: chỉ cần nhãn video-level, nhưng vẫn ra được **điểm theo thời gian** —
  chính là thứ Phase 6 (pseudo temporal labeling) cần, mà không phải train
  một mô hình "normal-only" riêng.
- "Normal-first" (unsupervised) để dành cho V2 nếu cần phát hiện loại bất thường
  chưa từng thấy.

## D3. Backbone: S3D (họ I3D) từ torchvision thay cho I3D gốc

- I3D gốc phụ thuộc `pytorchvideo`/`mmaction2` (khó cài, ít bảo trì).
- S3D = I3D với conv tách rời (separable), cùng họ Inception, pretrained
  Kinetics-400, feature 1024-d giống I3D, có sẵn trong torchvision và chạy được MPS.
- Có tùy chọn `r3d_18` (512-d) nhẹ hơn để debug nhanh.

## D4. Trích feature một lần, lưu đĩa; train head trên feature

- Backbone đóng băng ở V1. Feature lưu `.npz` mỗi video.
- Lý do: máy dev là M1 8GB — không thể fine-tune 3D CNN end-to-end. Train head
  trên feature mất vài phút. Fine-tune backbone (Phase 8) chỉ làm khi có GPU.

## D5. Gộp nhiều người: attention pooling có mask, theo từng segment

- Mỗi segment giữ tối đa `K` track (mặc định 4), ưu tiên track xuất hiện lâu
  và to trong segment. Feature `[S, K, D]` + mask.
- Attention pooling (learned query) → `[S, D]`. Ở chế độ `both`, nối thêm
  feature full-frame (ngữ cảnh / tương tác) rồi project.
- Thứ tự: gộp người **trong** segment trước, rồi Transformer **theo thời gian**.

## D6. Behavior head

- Attention pooling theo thời gian (dùng điểm anomaly làm trọng số) → phân loại
  behavior. Loss CE chỉ tính trên video abnormal (mask normal).
- Chưa dùng pseudo-label segment-level ở V1; sinh pseudo-label bằng
  `abd pseudo-label` để kiểm tra bằng mắt trước khi dùng để train.

## D7. Chia dữ liệu

- Chia theo **video**, stratified theo behavior class, seed cố định, ghi ra
  `splits.json` để mọi bước sau dùng chung. Mặc định 70/15/15.

## D8. Chống overfit

- Transformer nhỏ (2 layer, 4 head, d_model 256), dropout 0.3, weight decay,
  early stopping theo val PR-AUC.
