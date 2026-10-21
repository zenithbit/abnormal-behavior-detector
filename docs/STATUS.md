# Trạng thái & bàn giao — 2026-10-22

## Đã xong

- Pipeline V1 đầy đủ: `index → split → track → extract → train / cv → evaluate / pseudo-label / infer`.
- Dataset Ano-AAD: 395 video (41 normal, 354 abnormal, 9 lớp). Rất mất cân bằng.
- Baseline **full-frame** (trích trên Mac), 5-fold CV:

| model | anomaly ROC-AUC | balanced acc | behavior acc | end-to-end macro-F1 |
|---|---|---|---|---|
| frame_notf_mean (tốt nhất) | 0.888 ± 0.052 | 0.796 ± 0.076 | 0.404 ± 0.034 | 0.346 ± 0.052 |
| frame_tf2 (mặc định) | 0.867 ± 0.070 | 0.780 ± 0.094 | 0.398 ± 0.020 | 0.342 ± 0.035 |
| logistic regression (mean+max) | 0.893 | — | 0.441 | 0.422 (10-class acc 0.43) |

  - Ngưỡng chọn theo Youden (balanced); ngưỡng theo F1 làm specificity chỉ ~0.25.
  - Các lớp liên quan người (Arrest, Attack, Fighting, Ill-treatment, violence) nhầm lẫn nhau;
    `violence` F1 0.16. Lớp theo bối cảnh (fire-raising, Traffic, Explosion) tốt hơn.
  - Head nơ-ron chưa thắng logistic regression → nút thắt là feature/dữ liệu, không phải head.
- Quyết định của người dùng: **giữ 9 lớp**, chờ luồng person từ Colab rồi chọn model tốt nhất.

## Đã xong trên Windows (GTX 1070) — 2026-10-22

- Import `abd_results.zip` từ Colab (feature S3D frame + person, 395 video).
- `scripts/compare_models.sh`: 3 luồng × 4 head, 5-fold CV. NN tốt nhất `both_notf` (e2e F1 0.355 ± 0.059).
- Model cuối `final` (both_notf) trên test: e2e F1 0.251 (val 0.445) — test nhỏ, dao động mạnh.
- Thêm `abd linear` (logistic regression trên mean+max pooling). **LR both tốt nhất**:
  CV e2e F1 0.464 ± 0.024, test 0.370.
- Bảng đầy đủ: [`results/README.md`](../results/README.md). Setup máy mới + flow: [`SETUP.md`](SETUP.md).

## Việc tiếp theo

1. Quyết định model cho demo: LR both tốt hơn rõ nhưng chỉ chấm điểm mức video (không định vị đoạn).
   Hướng khả thi: LR cho anomaly + behavior, NN `final` chỉ để lấy điểm từng segment → cần thêm hỗ trợ LR trong `abd infer`.
2. Đóng gói demo: cần video gốc (`archive.zip` → `data/`), `yolo11n.pt`, rồi `scripts/pack_demo.sh` (xem [`SETUP.md`](SETUP.md) bước 6).

## Không có trong git

`data/` (dataset), `outputs/` (feature, model), `archive.zip`, `dist/`. Dataset lấy lại từ `archive.zip`
bằng `scripts/prepare_ano_aad.sh archive.zip data/Ano-AAD` (chỉ cần nếu muốn trích feature lại).
