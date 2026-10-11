# Model checkpoints

The PyTorch checkpoints the packaged TFLite models (5.0.0) were exported from.

| File | Role | Exported as |
|---|---|---|
| `stage1_hand_yolov8n.pt` | hand detector, Hands v4 | `app/model/model-a8/a9.tflite`, 960x544 |
| `stage2_gesture_yolov8s_cls.pt` | gesture classifier, gesture15 | `app/model/gesture-a8/a9.tflite`, 128x128 |

They are here for provenance and reuse. **Training is not part of this repository**; the
datasets and scripts live in `~/datasets/hands` (stage 1) and `~/datasets/handgesture2`
(stage 2, linked as `../dataset`).

---

## Stage 1 — hand detector (Hands v4)

| | |
|---|---|
| Architecture | YOLOv8n, anchor-free, 1 class `hand` |
| Trained at | 960 (fine-tune of Hands v3, which was trained at 1920) |
| Data | 1,900 verified frames from five Axis cameras (indoor and outdoor, day and night), x8, + 10,000 HaGRID images with every hand labelled + 351 person-free frames |
| Deployed | 960x544, int8, split outputs `[1,4,10710]` + `[1,1,10710]` |

Measured on the test split (1,136 real hands; strict test sessions never trained on), int8
ARTPEC-9, confidence 0.3, hand found (IoU >= 0.05):

| hand width at the 960x544 input | 6-10 px | 10-16 px | 16-24 px | 24-40 px | 40+ px | all |
|---|---|---|---|---|---|---|
| Hands v3 (trained at 1920) | 0.63 | 0.80 | 0.95 | 0.97 | 0.96 | 0.83 |
| **Hands v4** | **0.82** | **0.86** | 0.96 | 0.97 | 1.00 | **0.89** |

Training at the deployment size is what made the difference: at 960 every hand is half
the size v3 learned. ARTPEC-8 int8 is within 0.02 of ARTPEC-9.

**Known limits:** int8 box coordinates step by 3.76 px at the input (about 15 px on a 4K
frame), so small boxes can sit beside the hand and duplicate boxes appear (NMS 0.2 merges
most). Some false hands on elbows, a red garden-tool handle, a steering wheel; stage 2
turns most of these into `no_gesture`.

## Stage 2 — gesture classifier (gesture15)

| | |
|---|---|
| Architecture | YOLOv8s-cls, 128x128 input |
| Classes | 15: call, dislike, fist, four, like, middle_finger, mute, no_gesture, ok, one, palm, peace, rock, stop, three |
| Data | HaGRID crops, 6,000 per class (75 % degraded to 16-96 px first) + 1,903 verified real hands from an indoor (2688x1512) and an outdoor (3840x2160) camera, cut like the ACAP crops (square, 1.25x box) plus 7 copies each with the box shifted up to 30 % and sized 1.05-1.7x, + 70 false hand detections as `no_gesture` |
| Training | 30 epochs from ImageNet weights, vertical flip off (it would turn like into dislike) |

Held-out real hands (445, split by 10 s blocks):

| | exact boxes | camera-like box errors |
|---|---|---|
| 4.x classifier (gesture20, best of the 15 kept classes) | 89.7 %* | 74.4 %* |
| YOLOv8n-cls, same data | 93.7 % | 88.2 % |
| **YOLOv8s-cls, float** | **96.4 %** | **94.0 %** |
| YOLOv8s-cls, ARTPEC-9 int8 | 96.2 % | 93.7 % |
| YOLOv8s-cls, ARTPEC-8 int8 | 94.6 % | 93.5 % |

\* office camera only. The s model was chosen over n for the 6-point gain; it runs once
per hand on a 128x128 crop, small next to the detector.

With the 60 % threshold -> `no_gesture` rule: 92.6-92.8 % with camera-like boxes.
Weakest: `fist` (-> `one`, `no_gesture`), `three` (-> `peace`).

**Known limits:** one person, two cameras. Int8 calibration: 150 real + 150 HaGRID crops.
