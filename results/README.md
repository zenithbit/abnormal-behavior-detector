# Kết quả — 2026-10-22

Dataset Ano-AAD, 395 video (41 normal, 354 abnormal, 9 lớp). Feature S3D luồng frame + person trích trên Colab.
Ngưỡng bất thường chọn trên val theo Youden. "e2e" = cascade như khi triển khai (Normal + 9 lớp).

- `abd_results.zip` (282 MB, **không có trong git** — tải từ Google Drive `MyDrive/abd/abd_results.zip`):
  `manifest.csv`, `splits.json`, `tracks/`, `features/s3d/`, `runs/` từ Colab.
  Nạp vào máy: `scripts/import_results.sh abd_results.zip`.
- `runs/*/metrics.json`: model train trên split cố định (val + test).
- `runs/*/cv_summary.json`: 5-fold CV (mean, std, từng fold).

## Tập test cố định (59 video: 53 abnormal, 6 normal)

| model | run | AUC | specificity | behavior acc | e2e acc | e2e macro-F1 |
|---|---|---|---|---|---|---|
| **LR both** | `lr_both` | **0.962** | **0.833** | **0.453** | **0.424** | **0.370** |
| LR frame | `lr_frame` | 0.874 | 1.000 | 0.415 | 0.390 | 0.329 |
| LR person | `lr_person` | 0.950 | 0.833 | 0.358 | 0.356 | 0.284 |
| NN both_notf | `final` | 0.871 | 0.667 | 0.302 | 0.305 | 0.251 |

## 5-fold CV (ước lượng đáng tin hơn — test chỉ có 2–8 video mỗi lớp)

| model | run | AUC | bal. acc | behavior acc | e2e acc | e2e macro-F1 |
|---|---|---|---|---|---|---|
| **LR both** | `cvlr_both` | **0.949 ± 0.047** | **0.894 ± 0.060** | **0.477 ± 0.052** | **0.486 ± 0.017** | **0.464 ± 0.024** |
| LR frame | `cvlr_frame` | 0.920 ± 0.045 | 0.800 ± 0.052 | 0.415 ± 0.031 | 0.397 ± 0.032 | 0.367 ± 0.041 |
| NN both_notf | `cmp_both_notf` | 0.898 ± 0.074 | 0.839 ± 0.066 | 0.429 ± 0.049 | 0.405 ± 0.055 | 0.355 ± 0.059 |
| NN frame_notf_mean | `cmp_frame_notf_mean` | 0.896 ± 0.052 | 0.793 ± 0.106 | 0.404 ± 0.049 | 0.377 ± 0.056 | 0.334 ± 0.067 |
| NN both_tf1 | `cmp_both_tf1` | 0.863 ± 0.085 | 0.771 ± 0.090 | 0.384 ± 0.070 | 0.375 ± 0.055 | 0.332 ± 0.065 |
| LR person | `cvlr_person` | 0.936 ± 0.018 | 0.860 ± 0.040 | 0.361 ± 0.043 | 0.382 ± 0.029 | 0.327 ± 0.029 |
| NN both_notf_mean | `cmp_both_notf_mean` | 0.908 ± 0.060 | 0.820 ± 0.058 | 0.387 ± 0.043 | 0.372 ± 0.025 | 0.318 ± 0.052 |
| NN frame_tf2 | `cmp_frame_tf2` | 0.879 ± 0.086 | 0.812 ± 0.049 | 0.367 ± 0.020 | 0.354 ± 0.042 | 0.315 ± 0.055 |
| NN frame_tf1 | `cmp_frame_tf1` | 0.899 ± 0.056 | 0.811 ± 0.110 | 0.396 ± 0.035 | 0.370 ± 0.066 | 0.311 ± 0.080 |
| NN frame_notf | `cmp_frame_notf` | 0.887 ± 0.070 | 0.802 ± 0.078 | 0.384 ± 0.038 | 0.352 ± 0.053 | 0.306 ± 0.073 |
| NN both_tf2 | `cmp_both_tf2` | 0.912 ± 0.046 | 0.833 ± 0.062 | 0.384 ± 0.032 | 0.359 ± 0.032 | 0.304 ± 0.056 |
| NN person_notf_mean | `cmp_person_notf_mean` | 0.831 ± 0.022 | 0.735 ± 0.042 | 0.353 ± 0.035 | 0.337 ± 0.052 | 0.283 ± 0.055 |
| NN person_notf | `cmp_person_notf` | 0.849 ± 0.021 | 0.750 ± 0.058 | 0.311 ± 0.019 | 0.294 ± 0.025 | 0.250 ± 0.036 |
| NN person_tf2 | `cmp_person_tf2` | 0.888 ± 0.044 | 0.768 ± 0.084 | 0.333 ± 0.032 | 0.301 ± 0.044 | 0.248 ± 0.043 |
| NN person_tf1 | `cmp_person_tf1` | 0.851 ± 0.029 | 0.762 ± 0.046 | 0.299 ± 0.027 | 0.268 ± 0.027 | 0.228 ± 0.034 |

`cv_frame` / `cv_person` / `cv_both` là CV chạy trên Colab bằng code cũ (head mặc định, chưa có balanced acc).

## Kết luận

- **LR trên mean+max pooling, luồng both, tốt nhất**: hơn NN tốt nhất ~+0.11 e2e F1 ở CV, độ lệch giữa fold nhỏ hơn nhiều.
- Gộp frame + person giúp rõ với LR; luồng person một mình yếu nhất về phân loại hành vi.
- Head transformer không giúp: biến thể `notf` (không transformer) ≥ `tf1` / `tf2`.
- Lỗi lớn nhất của NN `final` trên test: 6/7 video Burglary bị coi là Normal; `violence` / `Ill-treatment` nhầm với `Arrest`.
- LR chỉ cho điểm mức video — không có điểm từng segment nên không định vị được đoạn thời gian.
  `abd infer` / `pack_demo.sh` hiện chỉ dùng checkpoint NN (`best.pt`).

Thời gian trên GTX 1070: `scripts/compare_models.sh` (12 cấu hình × 5 fold) ~12 phút; `train` ~30 giây; `linear` vài giây.
