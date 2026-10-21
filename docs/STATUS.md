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

## Đang chạy

- Colab (`notebooks/colab_pipeline.ipynb`): track + extract luồng frame + person → `MyDrive/abd/abd_results.zip`.

## Việc tiếp theo (trên máy Windows)

1. Cài đặt (PowerShell, trong thư mục repo):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   uv venv --python 3.11 .venv
   uv pip install --python .venv\Scripts\python.exe torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
   uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
   .venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
   ```
   Các script `scripts/*.sh` chạy trong **Git Bash** (tự nhận `.venv\Scripts`).
2. Tải `abd_results.zip` từ Drive về thư mục repo → (Git Bash) `scripts/import_results.sh abd_results.zip`.
3. `scripts/compare_models.sh` → chọn cấu hình thắng (bảng sắp theo end-to-end macro-F1).
4. Train model cuối với cấu hình đó: `abd --set model.input=<mode> ... train --name final`.
5. Chọn 4–5 video **trong tập test** (`outputs/splits.json`), rồi
   `scripts/pack_demo.sh outputs/runs/final <videos...>` → `dist/abd_demo.zip`.
   Trên chính máy Windows có GPU thì có thể demo trực tiếp bằng
   `abd infer --ckpt outputs/runs/final/best.pt --video x.mp4 --render x_out.mp4`.
6. Nếu kịp: thêm baseline logistic regression vào CLI (đang tốt nhất ở end-to-end).

## Không có trong git

`data/` (dataset), `outputs/` (feature, model), `archive.zip`, `dist/`. Dataset lấy lại từ `archive.zip`
bằng `scripts/prepare_ano_aad.sh archive.zip data/Ano-AAD` (chỉ cần nếu muốn trích feature lại).
