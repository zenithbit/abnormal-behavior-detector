# Thiết kế hệ thống nhận diện hành vi bất thường trong video

## 1. Mục tiêu hệ thống

### 1.1. Mục tiêu tổng quát

Xây dựng một hệ thống Computer Vision có khả năng nhận một video đầu vào và trả lời hai câu hỏi:

1. Video có chứa hành vi bất thường hay không?
2. Nếu có, hành vi bất thường thuộc loại nào?

Ví dụ:

```text
Input:
    video_01.mp4

Output:
    Anomaly: YES
    Behavior: Fighting
    Confidence: 92%
```

Hoặc:

```text
Input:
    video_02.mp4

Output:
    Anomaly: NO
    Behavior: Normal
    Confidence: 97%
```

### 1.2. Phạm vi đầu ra

Phiên bản đầu tiên **không đặt mục tiêu phân loại hành vi trực tiếp trên từng bounding box/person**.

Output chính là:

```text
Video/Clip
    -> Normal / Abnormal
    -> Nếu Abnormal: Behavior class
```

Bounding box của người chỉ được sử dụng trong quá trình xử lý đầu vào.

> **Lý do:** Dataset Ano-AAD được tổ chức theo nhãn video/lớp hành vi, không cung cấp ground truth rõ ràng cho từng người trong từng frame. Nếu cố huấn luyện trực tiếp `Person A -> Fighting`, `Person B -> Normal` khi không có nhãn person-level thì mô hình sẽ học theo pseudo-label hoặc giả định chưa được chứng minh. Vì vậy nên bắt đầu với video/clip-level classification, sau đó mới mở rộng sang person-level localization nếu cần.

---

# 2. Ý tưởng tổng thể

Pipeline đề xuất:

```text
                         INPUT VIDEO
                              |
                              v
                    +-------------------+
                    | YOLO Person       |
                    | Detection         |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    | Person Tracking   |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    | Person Clips /    |
                    | Temporal Tracks   |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    | I3D               |
                    | Spatio-temporal   |
                    | Feature Extractor |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    | Feature           |
                    | Aggregation       |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    | Temporal          |
                    | Transformer       |
                    +---------+---------+
                              |
                    +---------+---------+
                    |                   |
                    v                   v
            +---------------+   +---------------+
            | Anomaly Head  |   | Behavior Head |
            | Normal/Abnormal|   | Behavior Class|
            +-------+-------+   +-------+-------+
                    |                   |
                    +---------+---------+
                              |
                              v
                         FINAL OUTPUT
```

Hệ thống được chia thành hai nhiệm vụ:

```text
Stage 1:
Anomaly Detection
    Normal vs Abnormal

Stage 2:
Behavior Classification
    Fighting / Theft / Assault / ...
```

---

# 3. Dataset

## 3.1. Dataset chính: Ano-AAD

Theo cấu trúc dataset đã thống nhất:

```text
Ano-AAD/
├── normal/
│   ├── video_01.mp4
│   ├── video_02.mp4
│   └── ...
│
└── abnormal/
    ├── Fighting/
    │   ├── video_01.mp4
    │   ├── video_02.mp4
    │   └── ...
    │
    ├── Theft/
    │   ├── video_01.mp4
    │   └── ...
    │
    └── ...
```

Ta chuyển cấu trúc này thành label:

```text
normal/video_01.mp4
    anomaly_label = 0
    behavior = Normal

abnormal/Fighting/video_01.mp4
    anomaly_label = 1
    behavior = Fighting
```

## 3.2. Vì sao chọn Ano-AAD?

Ano-AAD phù hợp với hướng thiết kế hiện tại vì cùng một dataset có thể hỗ trợ hai nhiệm vụ:

```text
normal/
    -> học đặc trưng Normal

abnormal/<class>/
    -> phát hiện Abnormal
    -> biết class hành vi
```

Điều này đơn giản hóa pipeline dữ liệu và tránh phải ghép nhiều dataset có label space khác nhau.

> Không nên đơn giản concatenate nhiều dataset nếu cách định nghĩa `Normal`, `Abnormal` và các behavior class khác nhau. Trước tiên nên xây dựng một pipeline ổn định trên một dataset có cấu trúc nhãn nhất quán.

---

# 4. Chia train/validation/test

## 4.1. Nguyên tắc quan trọng

Phải chia **theo video trước khi chia clip**.

Ví dụ:

```text
Train:
    video_01
    video_02
    video_03

Validation:
    video_04

Test:
    video_05
```

Không được:

```text
video_01
    clip_01 -> train
    clip_02 -> test
```

## 4.2. Vì sao?

Các clip từ cùng một video có thể rất giống nhau về:

- background
- camera
- người
- ánh sáng
- chuyển động
- bối cảnh

Nếu một video xuất hiện ở cả train và test, metric có thể cao giả tạo.

> Đây là một trong những lỗi data leakage quan trọng nhất đối với video classification.

---

# 5. Bước 1 — YOLO Person Detection

## 5.1. Vai trò của YOLO

YOLO chỉ có nhiệm vụ:

```text
Frame
    |
    v
YOLO
    |
    v
Person bounding boxes
```

Ví dụ:

```text
Frame t
    Person 1 -> bbox
    Person 2 -> bbox
    Person 3 -> bbox
```

YOLO **không chịu trách nhiệm** dự đoán:

```text
Fighting
Theft
Assault
...
```

## 5.2. Vì sao dùng YOLO?

Video surveillance thường chứa rất nhiều thông tin không liên quan:

```text
Người
Xe
Đường
Tòa nhà
Cây
Biển quảng cáo
Background
...
```

Nếu đưa toàn bộ frame trực tiếp vào mô hình hành vi, feature extractor phải học cả:

```text
background + object + person + motion
```

Trong khi mục tiêu chính là hành vi của con người.

YOLO giúp tạo:

```text
Full frame
    |
    v
Person regions
```

từ đó giảm bớt background noise và tập trung mô hình vào human-centric information.

## 5.3. Vì sao không để YOLO phân loại hành vi?

YOLO rất phù hợp cho object detection, nhưng behavior recognition là bài toán **spatio-temporal**.

Ví dụ một frame riêng lẻ có thể không cho biết hành vi:

```text
Frame 1: hai người đứng gần nhau
Frame 2: chuyển động
Frame 3: chuyển động tiếp diễn
...
```

Hành vi nằm trong **chuỗi thời gian**, không chỉ trong một bounding box.

Vì vậy:

```text
YOLO
    = Where are the people?

I3D + Temporal Transformer
    = What are they doing over time?
```

Đây là sự phân chia nhiệm vụ hợp lý hơn.

---

# 6. Bước 2 — Person Tracking

Sau khi YOLO detect người ở từng frame, cần tracking:

```text
Frame 1:
    Person A -> ID 1

Frame 2:
    Person A -> ID 1

Frame 3:
    Person A -> ID 1
```

Có thể sử dụng một tracker như ByteTrack/BoT-SORT tùy implementation.

## Vì sao cần tracking?

Nếu chỉ detect độc lập từng frame:

```text
Frame 1 -> bbox A
Frame 2 -> bbox ?
Frame 3 -> bbox ?
```

mô hình không biết các bounding box đó thuộc cùng một người.

Tracking tạo:

```text
Person ID 1
    |
    +-- frame 1
    +-- frame 2
    +-- frame 3
    +-- ...
    +-- frame T
```

Đây chính là temporal sequence cần thiết cho I3D.

## Kết quả

Ta có:

```text
Video
    |
    v
YOLO
    |
    v
Tracking
    |
    +--> Track Person 1
    |
    +--> Track Person 2
    |
    +--> Track Person 3
```

---

# 7. Bước 3 — Tạo Person Temporal Clips

Không nên đưa từng frame riêng lẻ vào model.

Thay vào đó, từ mỗi track tạo sequence:

```text
Person 1:

Frame t-15
Frame t-14
...
Frame t
...
Frame t+14
```

Ví dụ một clip có `T` frames:

```text
X_person ∈ R^(T × H × W × 3)
```

Trong đó:

- `T`: số frame
- `H`: chiều cao crop
- `W`: chiều rộng crop
- `3`: RGB

## Vì sao?

Hành vi là temporal.

Ví dụ:

```text
Một frame:
    "người đang đứng"

Nhiều frame:
    "người liên tục thực hiện một chuỗi chuyển động"
```

Do đó video model cần nhìn **sequence**, không chỉ image.

---

# 8. Bước 4 — I3D làm Spatio-Temporal Feature Extractor

## 8.1. Chọn I3D

I3D (Inflated 3D ConvNet) là lựa chọn phù hợp cho phiên bản đầu vì nó được thiết kế để xử lý video bằng 3D convolution.

Thay vì chỉ học:

```text
Spatial:
    H × W
```

I3D học:

```text
Temporal + Spatial:
    T × H × W
```

Nó có khả năng học:

```text
Appearance
+
Motion
```

## 8.2. Vì sao không chỉ dùng CNN 2D?

CNN 2D xử lý tốt từng frame:

```text
Frame -> CNN -> Feature
```

Nhưng nếu muốn hiểu:

```text
Frame 1
Frame 2
Frame 3
...
Frame T
```

thì phải có thêm temporal modeling.

I3D tích hợp temporal dimension ngay trong feature extraction.

## 8.3. Vì sao chưa chọn Video Transformer ngay từ đầu?

VideoMAE/Video Swin/TimeSformer cũng là những hướng rất mạnh.

Tuy nhiên project hiện tại cần một baseline rõ ràng, dễ phân tích và dễ debug.

I3D cung cấp:

```text
Video
 -> spatio-temporal features
```

sau đó Transformer đảm nhận:

```text
long-range temporal reasoning
```

Sự phân chia này giúp kiến trúc dễ hiểu:

```text
I3D
= Extract local spatio-temporal features

Transformer
= Model temporal relationships
```

---

# 9. Bước 5 — Feature Aggregation

Sau I3D, mỗi person track có feature:

```text
Person 1 -> Feature sequence
Person 2 -> Feature sequence
Person 3 -> Feature sequence
```

Nhưng video có thể có số người khác nhau:

```text
Video A -> 2 people
Video B -> 5 people
Video C -> 10 people
```

Do đó cần aggregation.

Có thể biểu diễn:

```text
Person features
      |
      v
Interaction / Attention / Pooling
      |
      v
Video-level representation
```

## Vì sao không đơn giản chọn một người?

Vì hành vi bất thường có thể liên quan đến:

```text
một người
```

hoặc:

```text
nhiều người tương tác
```

Ví dụ các hành vi tương tác không thể được hiểu đầy đủ chỉ từ một crop.

Do đó model cần có khả năng tổng hợp thông tin từ nhiều person tracks.

---

# 10. Bước 6 — Temporal Transformer

Sau I3D:

```text
Video
 ↓
Person tracks
 ↓
I3D
 ↓
Feature sequence
```

Tiếp tục:

```text
Feature sequence
       |
       v
Temporal Transformer
       |
       v
Contextual temporal representation
```

## Vì sao chọn Transformer?

Transformer có khả năng sử dụng attention để xem xét quan hệ giữa các thời điểm khác nhau.

Ví dụ:

```text
t1 --------\
t2 ----------\
t3 ----------- > Temporal relationship
t4 ----------/
t5 ---------/
```

Điều này hữu ích khi hành vi không chỉ phụ thuộc vào chuyển động ngay lập tức mà phụ thuộc vào diễn biến trước đó.

## I3D và Transformer không trùng nhiệm vụ

Có thể hiểu:

```text
I3D:
    "Trong đoạn video ngắn này đang xảy ra chuyển động/đặc trưng gì?"

Transformer:
    "Các đặc trưng ở các thời điểm khác nhau liên hệ với nhau như thế nào?"
```

Đây là lý do kết hợp hai thành phần.

---

# 11. Bước 7 — Anomaly Detection Head

Temporal representation được đưa vào anomaly head:

```text
Transformer feature
        |
        v
   Anomaly Head
        |
        v
Normal / Abnormal
```

Output:

```text
P(Normal)
P(Abnormal)
```

Ví dụ:

```text
P(Normal)   = 0.08
P(Abnormal) = 0.92
```

=> Abnormal.

---

# 12. Bước 8 — Behavior Classification Head

Nếu clip/video được xác định là abnormal, behavior head dự đoán class:

```text
Transformer feature
        |
        v
 Behavior Head
        |
        v
Fighting
Theft
Assault
...
```

Ví dụ:

```text
Fighting = 0.92
Theft    = 0.04
Other    = 0.04
```

=> Fighting.

---

# 13. Multi-task Architecture

Thay vì xây dựng hai model hoàn toàn độc lập, có thể dùng shared backbone:

```text
                    Video
                      |
                     YOLO
                      |
                   Tracking
                      |
                    I3D
                      |
             Temporal Transformer
                      |
              Shared Representation
                 /            \
                /              \
               v                v
       Anomaly Head       Behavior Head
             |                  |
       Normal/Abnormal     Behavior class
```

## Vì sao?

Hai nhiệm vụ có quan hệ trực tiếp:

```text
Anomaly detection:
    "Có gì bất thường?"

Behavior classification:
    "Bất thường đó là gì?"
```

Cả hai đều cần thông tin từ cùng một temporal representation.

Shared backbone giúp:

- tránh xây dựng hai pipeline hoàn toàn độc lập
- tái sử dụng feature
- có thể học representation phù hợp cho cả hai nhiệm vụ

---

# 14. Loss Function

Có thể sử dụng multi-task loss:

```text
L_total =
    λ_anomaly * L_anomaly
    +
    λ_behavior * L_behavior
```

Trong đó:

```text
L_anomaly
    = Binary Cross Entropy

L_behavior
    = Cross Entropy
```

Behavior loss chỉ được áp dụng cho abnormal samples:

```text
Normal:
    L_anomaly = calculated
    L_behavior = ignored

Abnormal:
    L_anomaly = calculated
    L_behavior = calculated
```

Ví dụ:

```text
Normal video
    anomaly target = 0

Fighting video
    anomaly target = 1
    behavior target = Fighting
```

## Vì sao multi-task loss?

Vì mô hình phải đồng thời học:

```text
Normal vs Abnormal
```

và:

```text
Nếu abnormal -> class nào?
```

Multi-task learning cho phép backbone học representation phục vụ cả hai mục tiêu.

---

# 15. Chiến lược đặc biệt: Normal-first Anomaly Detection

Đây là phần quan trọng trong ý tưởng ban đầu của project.

Thay vì ngay lập tức yêu cầu model học:

```text
Normal
vs
Fighting
vs
Theft
vs
...
```

ta có thể bắt đầu bằng:

```text
NORMAL DATA
     |
     v
Learn normal behavior representation
```

Sau đó chạy model trên abnormal videos:

```text
Abnormal video
      |
      v
Anomaly scoring theo temporal clips
      |
      v
High anomaly score
      |
      v
Candidate abnormal segments
```

---

# 16. Pseudo Temporal Labeling

Ano-AAD có video-level label.

Ví dụ:

```text
abnormal/Fighting/video01.mp4
```

Ta biết:

```text
Video = Fighting
```

nhưng chưa biết:

```text
00:00 - 00:10 -> Normal?
00:10 - 00:20 -> Fighting?
00:20 - 00:30 -> Normal?
```

Anomaly detector có thể giúp tạo pseudo-label:

```text
Video:
0s -------- 60s

Anomaly score:
0.1 0.1 0.2 0.8 0.9 0.95 0.2 0.1

                     ↑
              candidate anomaly
```

Sau thresholding:

```text
0s - 20s     -> Normal candidate
20s - 35s    -> Abnormal candidate
35s - 60s    -> Normal candidate
```

Gán class của video:

```text
20s - 35s -> Fighting
```

Ta thu được pseudo temporal annotation.

## Vì sao hướng này hấp dẫn?

Không cần manually annotate từng frame ngay từ đầu.

Ta tận dụng:

```text
Video-level label
+
Anomaly score
```

để tạo:

```text
Pseudo temporal label
```

Đây là một dạng weak supervision / pseudo-labeling.

---

# 17. Không tin tuyệt đối vào pseudo-label

Pseudo-label có thể sai.

Ví dụ model có thể đánh dấu:

```text
camera movement
lighting change
scene transition
crowd movement
```

là anomaly.

Do đó không nên:

```text
anomaly score > threshold
    -> chắc chắn là ground truth
```

Thay vào đó cần dùng:

```text
High confidence
+
Temporal consistency
+
Contiguous segment
```

Ví dụ:

```text
score > threshold
for several consecutive clips
```

mới coi là candidate anomaly segment.

Threshold và số clip liên tiếp cần được xác định qua validation, không nên cố định một con số tùy ý ngay từ đầu.

---

# 18. Behavior Classification Training

Sau khi có candidate abnormal segments:

```text
Ano-AAD
    |
    v
Anomaly detector
    |
    v
High-confidence abnormal segments
    |
    v
Parent video label
    |
    v
Pseudo labeled behavior clips
```

Ví dụ:

```text
abnormal/Fighting/video01.mp4

        ↓ anomaly localization

[00:20 - 00:35]

        ↓

label = Fighting
```

Sau đó dùng các clip này để train behavior classifier.

---

# 19. Một vấn đề quan trọng: Interaction Behavior

Một số hành vi có thể phụ thuộc vào tương tác giữa nhiều người.

Ví dụ:

```text
Person A
      ↕
Person B
```

Nếu chỉ crop riêng:

```text
Person A crop
Person B crop
```

một phần context tương tác có thể bị mất.

Do đó cần giữ khả năng aggregation giữa các person tracks:

```text
Person 1 feature ──┐
Person 2 feature ──┼──> Interaction / Attention
Person 3 feature ──┘
                         |
                         v
                  Video representation
```

## Vì sao?

Mục tiêu không phải:

```text
"một người đang làm gì?"
```

mà là:

```text
"video đang thể hiện hành vi bất thường gì?"
```

Vì vậy context giữa nhiều người vẫn quan trọng.

---

# 20. Không ép mô hình phải chọn đúng một người

Đây là lý do quan trọng để không đặt bbox-level output làm mục tiêu chính.

Ví dụ một abnormal event có:

```text
Person A
Person B
Person C
```

Không nhất thiết chỉ có:

```text
Person B = abnormal
```

Có thể cả A và B đều tham gia vào hành vi.

Vì vậy ở phiên bản đầu:

```text
Video -> Fighting
```

là mục tiêu rõ ràng hơn:

```text
Person A -> Fighting
Person B -> Fighting
Person C -> Normal
```

khi dataset chưa có person-level annotation.

---

# 21. Output cuối cùng

## Input

```text
video.mp4
```

## Processing

```text
Video
  |
  v
YOLO
  |
  v
Person Detection
  |
  v
Tracking
  |
  v
Person Temporal Clips
  |
  v
I3D
  |
  v
Feature Aggregation
  |
  v
Temporal Transformer
  |
  +--------------------+
  |                    |
  v                    v
Anomaly Head       Behavior Head
  |                    |
  v                    v
Normal/Abnormal    Behavior class
```

## Output

Normal:

```text
{
    "anomaly": false,
    "behavior": "Normal",
    "confidence": 0.97
}
```

Abnormal:

```text
{
    "anomaly": true,
    "behavior": "Fighting",
    "confidence": 0.92
}
```

---

# 22. Visualization

Mặc dù output chính không phải bbox-level, hệ thống vẫn có thể hiển thị YOLO bounding boxes để người dùng nhìn thấy vùng người được model sử dụng.

Ví dụ:

```text
+----------------------------------------+
|                                        |
|       [Person]       [Person]          |
|                                        |
|             VIDEO                      |
|                                        |
+----------------------------------------+

Result:
Anomaly: YES
Behavior: Fighting
Confidence: 92%
```

Có thể highlight các temporal segments có anomaly score cao:

```text
Timeline:

0s          20s          40s          60s
|------------|============|-------------|
             ^^^^^^^^^^^^^
             anomaly segment

Behavior: Fighting
```

Điều này hữu ích hơn việc cố gắn nhãn Fighting lên một bbox khi chưa có ground truth person-level.

---

# 23. Metrics

Cần đánh giá riêng từng nhiệm vụ.

## 23.1. Anomaly Detection

Các metric:

```text
Precision
Recall
F1-score
ROC-AUC
PR-AUC
```

Đặc biệt:

```text
Recall
```

quan trọng nếu mục tiêu là hạn chế bỏ sót abnormal events.

## 23.2. Behavior Classification

```text
Accuracy
Macro Precision
Macro Recall
Macro F1
Confusion Matrix
```

Macro F1 đặc biệt hữu ích khi số lượng sample giữa các behavior class không cân bằng.

## 23.3. Temporal anomaly localization

Nếu có pseudo/ground-truth temporal labels phù hợp, có thể đánh giá:

```text
Frame-level AUC
Temporal IoU
Segment-level metrics
```

Nếu dataset không có ground truth temporal chính thức, cần ghi rõ rằng temporal segments được tạo bằng pseudo-labeling và không nên báo cáo chúng như ground truth.

---

# 24. Những gì không nên làm

## Không nên 1: YOLO phân loại hành vi

```text
YOLO -> Fighting
```

Không phải nhiệm vụ chính của YOLO.

Đúng:

```text
YOLO -> Person
I3D + Transformer -> Behavior
```

## Không nên 2: Chia clip ngẫu nhiên trước khi chia video

Sai:

```text
Video A clip 1 -> train
Video A clip 2 -> test
```

Đúng:

```text
Video A -> train
Video B -> test
```

## Không nên 3: Xem pseudo-label là ground truth

Pseudo-label có thể sai.

Cần:

```text
confidence filtering
+
temporal smoothing
+
validation
```

## Không nên 4: Ép person-level classification ngay từ đầu

Dataset hiện tại không cung cấp đủ supervision để chứng minh:

```text
Person X = Fighting
```

Do đó nên bắt đầu:

```text
Video = Fighting
```

---

# 25. Lộ trình triển khai

## Phase 1 — Data Pipeline

Xây dựng:

```text
Dataset Loader
Video Reader
Train/Val/Test Split
Clip Sampling
```

Output:

```text
video -> tensor clips + labels
```

---

## Phase 2 — YOLO + Tracking

Xây dựng:

```text
Video
 -> YOLO
 -> Tracker
 -> Person Tracks
```

Kiểm tra visualization trước khi đưa vào deep learning.

Mục tiêu:

```text
Track ID ổn định
Bounding boxes hợp lý
Không crop sai người quá nhiều
```

---

## Phase 3 — I3D Feature Extraction

Xây dựng:

```text
Person Clip
    ↓
I3D
    ↓
Feature vector
```

Kiểm tra:

- tensor shape
- temporal length
- feature dimension
- inference speed
- GPU memory

---

## Phase 4 — Temporal Transformer

Xây dựng:

```text
I3D features
    ↓
Temporal Transformer
    ↓
Video representation
```

Trước tiên có thể train một baseline đơn giản để kiểm chứng pipeline.

---

## Phase 5 — Anomaly Detection

Train:

```text
Normal
vs
Abnormal
```

Đánh giá:

```text
Precision
Recall
F1
ROC-AUC
PR-AUC
```

---

## Phase 6 — Temporal Pseudo-labeling

Chạy anomaly detector trên abnormal videos:

```text
video
 ↓
clip anomaly scores
 ↓
threshold + temporal smoothing
 ↓
candidate abnormal segments
```

Sau đó:

```text
candidate segment
+
parent video class
=
pseudo behavior clip
```

---

## Phase 7 — Behavior Classification

Train classifier:

```text
Abnormal clip
    ↓
Transformer representation
    ↓
Behavior classifier
    ↓
Fighting / Theft / ...
```

---

## Phase 8 — Multi-task Fine-tuning

Sau khi từng component hoạt động:

```text
I3D
+
Temporal Transformer
+
Anomaly Head
+
Behavior Head
```

fine-tune jointly:

```text
L_total =
    λ1 L_anomaly
    +
    λ2 L_behavior
```

---

## Phase 9 — End-to-End Inference

Cuối cùng:

```text
Input video
      ↓
YOLO
      ↓
Tracking
      ↓
Person clips
      ↓
I3D
      ↓
Transformer
      ↓
Anomaly?
   /       \
 No         Yes
 |           |
Normal    Behavior
              |
              v
         Fighting / ...
```

---

# 26. Phiên bản kiến trúc nên chốt

Kiến trúc đề xuất cho project:

```text
                     VIDEO
                       |
                       v
              YOLO (Person only)
                       |
                       v
                   Tracking
                       |
                       v
              Person Temporal Clips
                       |
                       v
                      I3D
                       |
                       v
             Multi-person Aggregation
                       |
                       v
             Temporal Transformer
                       |
             +---------+---------+
             |                   |
             v                   v
       Anomaly Head        Behavior Head
             |                   |
             v                   v
     Normal / Abnormal     Behavior Class
             |                   |
             +---------+---------+
                       |
                       v
                  FINAL RESULT
```

## Vai trò của từng model

| Component | Nhiệm vụ | Lý do |
|---|---|---|
| YOLO | Detect người | Tập trung vào human regions |
| Tracker | Theo dõi người | Tạo temporal identity |
| I3D | Spatio-temporal feature extraction | Học appearance + motion |
| Aggregation | Kết hợp nhiều người | Hỗ trợ interaction |
| Transformer | Temporal reasoning | Hiểu diễn biến theo thời gian |
| Anomaly Head | Normal/Abnormal | Nhiệm vụ anomaly detection |
| Behavior Head | Behavior class | Xác định loại bất thường |

---

# 27. Kết luận

Hệ thống không nên được hiểu là:

```text
YOLO
    ↓
"Person này đang Fighting"
```

mà là:

```text
YOLO
    ↓
"Đây là những người xuất hiện trong video"

I3D
    ↓
"Đây là các đặc trưng chuyển động/không gian"

Temporal Transformer
    ↓
"Đây là diễn biến của video theo thời gian"

Anomaly Head
    ↓
"Video có bất thường không?"

Behavior Head
    ↓
"Nếu có, đó là hành vi gì?"
```

Mục tiêu cuối cùng:

```text
                    INPUT VIDEO
                         |
                         v
                 ┌───────────────┐
                 │   PROCESSING  │
                 │ YOLO + I3D +  │
                 │  Transformer  │
                 └───────┬───────┘
                         |
                         v
                ┌─────────────────┐
                │   PREDICTION    │
                ├─────────────────┤
                │ Anomaly: YES    │
                │ Behavior:Fighting│
                │ Confidence: 92% │
                └─────────────────┘
```

**Đây là phiên bản V1 nên triển khai trước.**

Sau khi V1 hoạt động tốt, có thể mở rộng sang V2:

```text
Video
 ↓
YOLO + Tracking
 ↓
Person-level features
 ↓
Person Attention / MIL
 ↓
Person anomaly scores
 ↓
BBox-level attribution
```

Khi đó hệ thống mới có thể tiến tới việc trả lời:

```text
Video → Fighting
        +
Person A → high anomaly score
Person B → high anomaly score
Person C → low anomaly score
```

nhưng đây là **bài toán nâng cấp**, không nên đưa vào mục tiêu cốt lõi của V1 khi dataset hiện tại chưa có person-level ground truth.
