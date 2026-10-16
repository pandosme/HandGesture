# Bundled models

| File | Model | Input |
|---|---|---|
| model-<chip>-960x544.tflite | Hands v4 hand detector, YOLOv8n, 1 class (`hand`) -- Detection Range *Speed* | 960x544 |
| model-<chip>-1120x640.tflite | Hands v5 hand detector -- *Balanced* | 1120x640 |
| model-<chip>-1280x736.tflite | Hands v5 hand detector -- *Distance* | 1280x736 |
| gesture-<chip>.tflite | gesture15 classifier, YOLOv8s-cls, 15 classes | 128x128 |

`<chip>` is `a8` (per-tensor int8) or `a9` (per-channel int8). The detectors were
trained on verified frames from five Axis cameras plus the HaGRID dataset; the classifier
on HaGRID gesture crops plus verified camera hands. Exported with acap-model-pipeline.

| File | SHA-256 |
|---|---|
| model-a8-960x544.tflite | 9605317c47f8c909... |
| model-a8-1120x640.tflite | d5fd5cd0204b6b5d... |
| model-a8-1280x736.tflite | 5604b1ac9862f57a... |
| model-a9-960x544.tflite | 08c6dcce9f604f95... |
| model-a9-1120x640.tflite | e813f8abda72833f... |
| model-a9-1280x736.tflite | a5c1e1f51ba0cc3d... |
| gesture-a8.tflite | 80ea1f9ae48ea0c1... |
| gesture-a9.tflite | 7424cc4c9302652d... |
| labels.txt | fad6926e5d29328d... |
| gesture-labels.txt | 60319b001611fc39... |
