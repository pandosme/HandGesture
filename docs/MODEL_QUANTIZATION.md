# Model Quantization Parameters

HandGesture uses two full-INT8 YOLOv8 models in each package: a detector and a
gesture classifier. Both use uint8 input and output tensors.

## Chip Requirements

| Target | Weight quantization | Model files |
|---|---|---|
| ARTPEC-8 | Per-tensor only | `model-a8.tflite`, `gesture-a8.tflite` |
| ARTPEC-9 | Per-channel | `model-a9.tflite`, `gesture-a9.tflite` |

The output layout is otherwise identical. The detector must expose separate
coordinate `[1,4,N]` and score `[1,classes,N]` tensors. Float models and fused
YOLO outputs are rejected.

## Build-Time Validation

`app/extract_model_params.py` runs inside the SDK image after the selected files
are copied to the canonical runtime names. It validates:

- uint8 NHWC input
- split rank-3 detector outputs with matching box counts
- the anchor count implied by strides 8, 16 and 32
- detector and gesture label counts
- per-tensor A8 or per-channel A9 quantization

It then writes `model_params.h` with the coordinate, score and gesture output
scales and zero points. Larod API v3 exposes tensor shapes at runtime but not
these quantization constants.

Run the same validation through the target build:

```bash
./build.sh --target a8
./build.sh --target a9
```

The canonical `model.tflite` and `gesture.tflite` names exist only inside each
target build. Source models use chip-specific filenames.

## Troubleshooting

- `per-axis quantized tensors; ARTPEC-8 requires per-tensor`: re-export with
   `./export_yolov8.py --target a8 --weights <detector.pt> --calibration-dir <images>`
   or `./export_gesture.py --target a8 --weights <classifier.pt> --calibration-dir <crops>`.
- `no per-axis quantized tensors`: an A8 export was supplied to the A9 build.
- Class-count errors: update the corresponding labels file in model class order.
- Fused or float outputs: use the supplied exporters instead of Ultralytics'
   built-in TFLite export.
