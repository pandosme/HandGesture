# Model Quantization

HandGesture packages two full-INT8 models per chip: the stage 1 hand detector and the
stage 2 gesture classifier. Both have uint8 input and output tensors.

## Files and chip requirements

| Target | Weight quantization | Detector | Classifier |
|---|---|---|---|
| ARTPEC-8 | per-tensor only | `app/model/model-a8.tflite` | `app/model/gesture-a8.tflite` |
| ARTPEC-9 | per-channel | `app/model/model-a9.tflite` | `app/model/gesture-a9.tflite` |

Labels are chip-independent: `app/model/labels.txt` (`hand`) and
`app/model/gesture-labels.txt` (15 gestures, in classifier output order).

At build time the Dockerfile copies the selected chip's files to the names the runtime
loads: `model/model.tflite`, `model/labels.txt`, `model/classifier.tflite` and
`model/classifier-labels.txt`.

## Tensor layouts (5.0.0)

| Model | Input | Output |
|---|---|---|
| Detector (Hands v4) | `[1,544,960,3]` uint8 | coordinates `[1,4,10710]` uint8 + scores `[1,1,10710]` uint8 |
| Classifier (gesture15) | `[1,128,128,3]` uint8, scale 1/255, zero 0 | `[1,15]` uint8, scale 1/256, zero 0 |

The detector keeps coordinates and scores in **separate** tensors so each gets its own
quantization scale. A fused `[1,5,N]` tensor would share one scale between pixel
coordinates (hundreds) and scores (0-1) and lose the scores.

Even split, the coordinate tensor is coarse: its scale is 3.76 px at the 960x544 input
(about 15 px on a 4K frame). Small boxes therefore snap to that grid, which can place the
box beside a far hand and produce near-duplicate boxes. NMS 0.2 merges most duplicates,
and stage 2 was trained on deliberately shifted boxes to tolerate the rest.

## Runtime and build-time checks

The runtime reads the quantization parameters (scale, zero point) of both models from the
loaded TFLite files (`model_quant.c`), and the classifier's input size and class count
from its tensors. A model with a different input size or class count therefore works
without code changes, as long as the label files match.

`app/extract_model_params.py` runs in the SDK image during the build and rejects a
package that would not work:

- uint8 NHWC input for both models
- split rank-3 detector outputs with matching box counts, and the anchor count implied
  by strides 8, 16 and 32
- detector and classifier label counts equal to the models' class counts
- per-tensor weights for A8, per-channel for A9

It also writes `model_params.h` (generated, not in git) with build-time fallbacks for the
quantization constants.

```bash
./build.sh --target a8
./build.sh --target a9
```

## Producing the models

The 5.0.0 models were exported with `acap-model-pipeline` (`export-detector` /
`export-classifier`, int8 with chip-specific quantization), calibrated on real camera data:
300 frames at 960x544 for the detector, and 150 real hand crops plus 150 HaGRID crops for
the classifier. The stand-alone `export_yolov8.py` and `export_gesture.py` here produce the
same format:

```bash
./export_yolov8.py  --target a8 --weights models/stage1_hand_yolov8n.pt --calibration-dir <960x544 frames>
./export_gesture.py --target a8 --weights models/stage2_gesture_yolov8s_cls.pt --calibration-dir <hand crops>
```

## Troubleshooting

- `per-axis quantized tensors; ARTPEC-8 requires per-tensor`: an A9 export was given to the
  A8 build. Re-export with `--target a8`.
- `no per-axis quantized tensors`: an A8 export was given to the A9 build.
- Class-count errors: the labels file does not match the model; list the labels in the
  model's class order.
- Fused or float outputs: use the exporters above, not Ultralytics' built-in TFLite export.
