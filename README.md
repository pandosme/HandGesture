# HandGesture

On-camera hand gesture recognition for Axis cameras with **ARTPEC-8** and **ARTPEC-9**.
HandGesture finds hands in the video, reads which gesture each hand shows, and publishes
the result over **MQTT** and as **ONVIF events**, so a camera action rule, Node-RED flow
or home-automation system can react to a gesture.

HandGesture is a ready-to-run, single-purpose variant of
[DetectX](https://github.com/pandosme/DetectX): the models are built in, nothing needs
training. To build your own detector or two-tier models, use DetectX.

---

## Recognised gestures

```
call      dislike   fist      four      like
mute      ok        one       palm      peace
rock      stop      three     middle_finger
no_gesture
```

`no_gesture` is the label for hands that do not show a clear gesture: a relaxed hand, a
hand holding something, a hand the classifier is unsure about (see
[Gesture threshold](#gesture-threshold)), and objects the hand detector mistakes for a hand.
**It is switched off by default** (see [Recommended setup](#recommended-setup)), so only
real gestures are reported.

## How it works

```
   960x544 video                largest 16:9 video (e.g. 3840x2160)
        │                              │
        ▼                              │
   1. hand detector                    │
      "where are the hands?"           │
        │ 0-N hands                    ▼
        └──────►  each hand is cropped from the full-resolution video
                              │
                              ▼
                  2. gesture classifier: "which gesture?"
                              │
                              ▼
                  gesture  ->  MQTT / ONVIF event
```

The hand detector works on a small, fast frame. Each hand it finds is then cut from the
camera's full-resolution video, so the gesture classifier sees every finger even when
the person is several metres away.

---

## Requirements

* Axis camera with **ARTPEC-8** or **ARTPEC-9** and a deep-learning processor (DLPU)
* **ARTPEC-9: Axis OS 13 or later.** Earlier firmware computes these models wrongly on
  the ARTPEC-9 DLPU.

There is one package per chip, and they are **not interchangeable**:

| Package | Chip |
|---|---|
| `HandGesture_<version>_artpec8.eap` | ARTPEC-8 |
| `HandGesture_<version>_artpec9.eap` | ARTPEC-9 |

The chip is listed in the camera's datasheet. Installing the wrong package gives wrong
detections.

## Install

In the camera's web interface: **Apps → Add app**, select the package for your chip, then
start the app and open it.

The first start takes 30-60 seconds while the models load onto the DLPU. The page shows
"Loading model" and refreshes by itself when it is ready.

---

## Settings

### Detections

Live view with the detected hands and their gestures. **FPS** shows how many images are
processed per second (hover over it for the inference time per image).

| Setting | Default | What it does |
|---|---|---|
| Hand Confidence | 40 % | How sure the detector must be that something is a hand. Lower finds more distant hands, but also more false ones. |
| Gesture Threshold | 60 % | How sure the classifier must be of the gesture. Below it, the hand gets the *Below Threshold Label*. |
| Below Threshold Label | `no_gesture` | Label for hands whose gesture is uncertain. Empty reports them as `hand`. |
| Area of interest / Exclude zones | whole image | Only hands inside the area, and outside the exclude zones, are reported. |

#### Gesture threshold

Most false gestures come from hands that are moving, half hidden or far away. The
threshold turns those into `no_gesture`, so they never fire a gesture event. Raise it if
you get gestures nobody made; lower it if real gestures are missed.

### Recommended setup

Leave **`no_gesture` unticked** under Events/Labels (the default). Hands without a clear
gesture, and the occasional object the detector mistakes for a hand (an elbow, a garden
tool), are then not reported at all: no box, no detection message, no event. Only real
gestures come through.

`no_gesture` detections carry a high confidence (often 98-99 %). That is the classifier
being sure there is *no gesture*, not a sign that it is a hand; the detector's own score is
in `hand_c`. Tick `no_gesture` only if you want to know where hands are regardless of
gesture. If a fixed object keeps being detected, draw an exclude zone over it.

### Events

Which gestures fire events, and how strict the trigger is (**Prioritize**):

| Prioritize | A gesture event fires when the gesture is ... | Delay |
|---|---|---|
| Speed | seen once | first frame |
| Balanced | in about 3 frames in a row (75 % of the frames within 300 ms) | ~0.25 s |
| Accuracy (default) | in at least 3 frames and 60 % of the frames within 500 ms | ~0.5 s |

Balanced and Accuracy also require that no other gesture was seen more often in the same
window. A hand being raised often passes through `no_gesture`, `one` or `fist` for a frame
or two before the real gesture settles; those passing shapes do not fire. Two hands showing
different gestures can both fire.

**Min Event State Duration** keeps an event active until the gesture has been gone for
that long.

### MQTT

Broker address, credentials, TLS and the topic prefix (pre-topic, default `handgesture`).

### About

Camera, MQTT connection and application details, and the MQTT topics with this camera's
serial filled in. Check here first if something is wrong.

---

## Integration

All payloads include the camera's `serial` and the `name` and `location` set on the MQTT page.

### Detections

Topic `handgesture/detection/<serial>`, every processed frame with at least one hand:

```json
{
  "detections": [
    {
      "label": "peace", "c": 92, "hand_c": 88,
      "x": 274, "y": 224, "w": 180, "h": 104,
      "timestamp": 1756453942980, "refId": 260
    }
  ],
  "name": "Front door", "location": "Main entrance", "serial": "<serial>"
}
```

* `label`, `c`: the gesture and its confidence (percent).
* `hand_c`: how sure the detector is that this is a hand.
* `x`, `y`, `w`, `h`: the hand's box, in a 0-1000 space over the image.
* Below the gesture threshold, `label` is the Below Threshold Label (`no_gesture`), `c` is
  the hand confidence, and three fields are added: `gesture` and `gesture_c` (the
  classifier's best guess) and `"fallback": true`. These only appear when `no_gesture` is
  ticked under Events/Labels.

### Events

Topic `handgesture/event/<serial>/<gesture>/<true|false>`, only when a gesture event
changes state. `true` carries the detection that fired it plus `"state": true`; `false`
carries the gesture, `"state": false` and a timestamp.

```json
{
  "label": "stop", "c": 99, "hand_c": 89,
  "x": 254, "y": 483, "w": 54, "h": 110,
  "timestamp": 1791744527089, "refId": 7139,
  "state": true,
  "name": "Front door", "location": "Main entrance", "serial": "<serial>"
}
```

The same state is available as an **ONVIF event**,
`tnsaxis:CameraApplicationPlatform/handgesture/<gesture>`, with a boolean `state`. There is
one event per gesture, so a camera action rule can trigger on a single gesture.

### Rock-Paper-Scissors

The About page links to a small game played against the camera over MQTT: `fist` is rock,
`palm` is paper, `peace` is scissors.

---

## Range and accuracy

Range depends on how many pixels a hand covers, so on resolution, lens and distance.

* **Finding the hand:** reliable when the hand is at least ~64 px wide in a 4K image
  (~45 px at 2688x1512). Smaller hands are often still found.
* **Reading the gesture:** about 93-96 % of real hands on the author's indoor and outdoor
  cameras. Weakest are `fist` (sometimes read as `one`) and `three` (sometimes `peace`).

Gestures are recognised best facing the camera, in reasonable light. The models were
trained on one person; expect somewhat lower accuracy for other people and viewpoints.

## Performance

The live view shows **FPS**, the number of images processed per second. Measured with the
packaged models (hand detector at 960x544):

| Camera | No hands in view | One hand |
|---|---|---|
| ARTPEC-8 | 10.2 FPS (~98 ms per image) | 8.4-9 FPS (~110-120 ms) |
| ARTPEC-9 | 16 FPS (~62 ms per image) | slightly lower |

**Every hand in view lowers the frame rate.** The hand detector runs once per image, but
the gesture classifier runs once *per hand*: about 13-20 ms per hand on an ARTPEC-8, less
on an ARTPEC-9. Two or three people gesturing at the same time therefore give noticeably
fewer images per second than one, roughly 7.5 FPS with two hands on an ARTPEC-8.

The camera's own object analytics (AXIS Object Analytics / video object detection) share
the deep-learning processor but cost little: about 1 % on ARTPEC-8 and 3 % on ARTPEC-9.

Gestures need about 5 images per second to be recognised reliably (Accuracy votes over
500 ms), so both chips have room to spare.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| Nonsense detections on ARTPEC-9 | Axis OS older than 13 |
| Wrong or unstable detections | Wrong package for the chip (artpec8 vs artpec9) |
| Hands found but always `no_gesture` | Gesture threshold too high, or the hand is too far away |
| Gestures nobody made | Raise the Gesture Threshold or use Prioritize = Accuracy |
| Boxes on things that are not hands, all `no_gesture` 99 % | Untick `no_gesture` under Events/Labels (the default on new installs), or add an exclude zone |
| Nothing found at distance | The hand is too small in the image; use a narrower lens or move the camera closer |
| Page stays on "Model failed to load" | Restart the app; if it persists, check the log |

```sh
journalctl -f -u handgesture          # on the camera
```

A healthy start logs `Stage 2 ready: 15 classes at 128x128` and
`Full-resolution crop video <W>x<H> ... started`.

---

## Models and building

The packages contain two models, both full-INT8 and built separately for each chip:

| Model | What it does | Input |
|---|---|---|
| Hand detector (YOLOv8n) | finds hands; trained on real camera footage from five cameras plus the HaGRID dataset | 960x544 |
| Gesture classifier (YOLOv8s-cls) | reads the gesture of each hand, 15 classes; trained on HaGRID plus real camera hands | 128x128 |

To train and export your own detector or two-tier models, use
[DetectX](https://github.com/pandosme/DetectX).

Building the packages from source needs Docker:

```sh
./build.sh                 # both chips
./build.sh --target a8     # one chip
```

---

## Version history

### 5.0.0

* **Far better gesture recognition.** Each hand is cut from the camera's full-resolution
  video, and both models were retrained on real camera footage: a new hand detector that
  finds smaller, more distant hands, and a new gesture classifier (93-96 % on real hands,
  up from ~74 %).
* **15 gestures.** `peace_inverted`, `stop_inverted`, `three2`, `two_up` and
  `two_up_inverted` were dropped as rarely used and easily confused.
* **Gesture threshold** with a configurable label for uncertain hands (default 60 % ->
  `no_gesture`).
* **Balanced** added to Prioritize; Balanced and Accuracy now vote over time, so passing
  shapes while a hand is raised no longer fire events.
* No hidden minimum-size filter (4.x silently ignored small, distant hands).
* The page refreshes itself when the model has loaded.
* The About page shows the MQTT topics with the camera's serial.
* More robust: a failed inference no longer makes the app restart in a loop.
* Defaults: hand confidence 40 %, gesture threshold 60 %, `no_gesture` unticked (only real
  gestures are reported). Existing installs keep their label selection; untick
  `no_gesture` by hand after upgrading.

### 4.0.0

First two-stage release: a YOLOv8 hand detector feeding a gesture classifier, separate
ARTPEC-8 and ARTPEC-9 packages, 20 gestures. Requires Axis OS 13 on ARTPEC-9.

### 3.5.4 and earlier

Single-stage YOLOv5 releases. See the repository history.

---

## Support this project

HandGesture is free and open source. If it saves you time, consider supporting
its development:

<a href="https://buymeacoffee.com/fredjuhlinl" target="_blank">
  <img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy me a coffee">
</a>

**https://buymeacoffee.com/fredjuhlinl**

---

## License

MIT — see [LICENSE](LICENSE).
