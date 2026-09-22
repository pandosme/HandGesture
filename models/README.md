# Model checkpoints

The PyTorch checkpoints the packaged TFLite models were exported from.

| File | Role | Size |
|---|---|---|
| `stage1_hand_yolov8n.pt` | hand detector | 5.9 MB |
| `stage2_gesture_yolov8n_cls.pt` | gesture classifier | 2.9 MB |

They are here for provenance and reuse. **Training is not part of this repository** —
HandGesture is the application. For training and export tooling see
[DetectX](https://github.com/pandosme/DetectX).

---

## Stage 1 — hand detector

`stage1_hand_yolov8n.pt`

| | |
|---|---|
| Architecture | YOLOv8n, anchor-free, 3.0 M parameters |
| Classes | 1 — `hand` |
| Trained at | 640x640 (deployed at 1280x736) |
| Dataset | [HaGRIDv2](https://github.com/hukenovs/hagrid), all 34 gesture classes collapsed to one |
| Training data | 250 001 images, 1.3 M boxes |
| Result | precision 1.000, recall 0.999, mAP50 **0.995**, mAP50-95 **0.886** |

### Why it works at distance

HaGRID is selfie data: the median hand fills about **12 % of the image**, while a hand
at 10 m from a mounted camera fills roughly **1 %**. A detector trained on that data as
it comes never sees the object sizes it will actually meet.

The checkpoint was trained with aggressive downscaling augmentation — random scaling
across 0.1-1.9x combined with full mosaic — which drags the training distribution down
into the 16-60 px band a camera produces. That single choice matters more to range than
model size does.

Measured recall against hand size at the network input:

| hand size | 40 px | 32 px | 24 px | 20 px | 16 px |
|---|---|---|---|---|---|
| recall | 0.997 | 0.994 | 0.992 | 0.964 | 0.927 |

### Resolution independence

Trained at 640 square, deployed at 1280x736. This works because the backbone is fully
convolutional and the head is anchor-free: the network keys off object size in
**pixels**, not on input dimensions. The same checkpoint exports to any input size that
is a multiple of 32, and higher resolution buys proportionally more range.

---

## Stage 2 — gesture classifier

`stage2_gesture_yolov8n_cls.pt`

| | |
|---|---|
| Architecture | YOLOv8n-cls, 1.5 M parameters, 3.3 GFLOPs at 128 px |
| Classes | 19 — HaGRID v1's 18 gestures plus `no_gesture` |
| Input | 128x128 crop |
| Dataset | HaGRID, 114 000 training crops (6 000 per class) |
| Result | top-1 **0.996**, top-5 1.000 |

```
call      dislike   fist            four       like
mute      ok        one             palm       peace
rock      stop      three           three2     two_up
peace_inverted      stop_inverted   two_up_inverted     no_gesture
```

### Why HaGRID v1's classes

HaGRIDv2 adds 15 more gestures, but most are near-duplicates of ones already present —
`three3`, `three_gun`, `thumb_index2`, `hand_heart2`, `little_finger`. Pairs like that
confuse *systematically*: the model is consistently confident and consistently wrong,
which temporal voting cannot repair, because voting only fixes errors that are random.
The v1 set keeps every visually distinct shape and drops the confusable tail.

At 32 px and 48 px crops there are **no systematic class-pair confusions** in the
validation set — remaining errors are scattered, which is exactly the kind that voting
across frames does fix.

### `no_gesture` is the rejection class

Without it, every hand the detector finds would be forced into some gesture and the
application would fire continuously on idle hands.

Training it properly required reading the source annotations rather than the converted
YOLO labels: the conversion had discarded 200 393 `no_gesture` boxes — the idle second
hand present in most gesture images — leaving only 2 807. A rejection class trained on
2 807 samples against 18 classes with 30 000 each does not hold.

### Robustness to small crops

75 % of training crops were degraded into the 16-96 px band before being resized to
128. A hand 30 px wide in the scene, cropped and upscaled, is still 30 px of real
information; a classifier trained only on sharp crops collapses on exactly the input
stage 1 hands over.

Measured top-1 against the true size of the hand before upscaling:

| source hand | 48 px | 32 px | 24 px | 20 px | 16 px |
|---|---|---|---|---|---|
| top-1 | 0.996 | 0.994 | 0.990 | 0.986 | 0.973 |

### Orientation is meaningful

`peace`/`peace_inverted`, `stop`/`stop_inverted` and `two_up`/`two_up_inverted` are
three distinct pairs separated only by vertical orientation. The checkpoint was trained
with vertical flipping disabled and rotation limited to 10°; horizontal flipping is
used freely, since a mirrored right hand is simply a left hand.

Anything that re-orients the crop before classification will invert those six classes.

---

## Deployment

Both are exported to full-INT8 TFLite with uint8 input and output, one build per chip
(ARTPEC-8 per-tensor, ARTPEC-9 per-channel). Quantization costs the classifier 0.27 %
top-1 — 0.9974 to 0.9947, with 99.74 % agreement against the float model.

The INT8 calibration set was 363 real hand crops cut by stage 1 from five cameras in
the target installation, not dataset images. Calibration ranges come from whatever
images are fed in, and the error lands hardest on small low-contrast objects — the ones
that decide range.

See [../docs/MODEL_QUANTIZATION.md](../docs/MODEL_QUANTIZATION.md) for the tensor
layouts and scales.

---

## Known limits

* Accuracy figures come from downscaled dataset images: cleaner than a real camera
  frame, with no motion blur, no compression artefacts, and the subject facing the
  camera. Treat them as an optimistic ceiling.
* HaGRID is upright, front-facing, single-hand imagery. Hands seen from above, at a
  steep angle, or overlapping are outside the training distribution.
* The detector is trained on hands making gestures. An open relaxed hand at the side is
  under-represented — that is what `no_gesture` covers, and it is the least reliable
  class at small sizes.
