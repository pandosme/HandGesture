# DetectX Hand-Gesture

DetectX Hand Gesture is an ACAP to be installed in Axis Communications Cameras.  It provides real-time detection designed for users to communicate with a system using hand gestures.
**Intended for system integrators**, all outputs are designed for machine-to-machine (M2M) workflows, with flexible configuration from a built-in web UI and standards-based output via MQTT, ONVIF, or HTTP.

Examples of use cases are:
* Healthcare:Patient/nurse communication. Greater accessibility and autonomy.
* Retail: Customer satisfaction monitoring, customer asisstance needed, silent staff alert notifications.
* Industry: Touchless machine control. Enhanced safety and efficiency.
* Consumer/home: Smart home device management. Hands-free convenience.
* Accessibility: Adaptive tech for disabilities. Inclusive interface design.

### [Download ACAP](https://www.dropbox.com/scl/fi/92rsm7zkqxlaqvxlfq038/HandGestures.zip?rlkey=ukyzhemobwgncuw2zi47099f6&st=y1r6ictd&dl=1)

***

# DetectX Hand-Gesture User & Integration Guide

***

![gestures](https://raw.githubusercontent.com/hukenovs/hagrid/Hagrid_v1/images/gestures.jpg)

## Hand Gesture Model Overview
**Dataset:** [Hagird V2](https://github.com/hukenovs/hagrid)
**Labels:**  
- like
- dislike
- ok
- call
- fist
- one
- two (peace)
- three
- four
- five (palm)
- stop
- mute
- rock
- middle_finger
- peace_inverted
- stop_inverted
- thumb_index
- two_up
- two_up_inverted
- no_gesture

**Platform details:**  
- **ARTPEC-8:**  
  - Model input: **960**  
  - Model size: **Small**
- **ARTPEC-9:**  
  - Model input: **1440**  
  - Model size: **Medium**

*Note: ARTPEC-8 and ARTPEC-9 are Axis camera chipset platforms. ARTPEC-9 offers enhanced performance and supports larger model input sizes for improved detection quality.*

***

## Application Overview

DetectX provides real-time detection and state data from network cameras directly to your systems. **Intended for system integrators**, all outputs are designed for machine-to-machine (M2M) workflows, with flexible configuration from a built-in web UI and standards-based output via MQTT, ONVIF, or HTTP.

Typical use cases include:
- Counting and presence of faces in scene
- Provide cropped face detections to system that want to further process the image

***

## Menu & Feature Walkthrough

Each menu item below describes both *user options* and *integration outputs*, matched to the associated interface screenshot for easy visual orientation.

***

### 1. Detections

Allows you to see object detections overlayed on the video, and to adjust detection parameters.

<img src="https://pandosme.github.io/assets/DetectX_Detections.jpeg" alt="Detections Page" width="500"/>

- **Adjust Confidence Threshold:**  
  Set the minimum confidence (0–100) for labeling a detection as valid.
- **Set Area of Interest (AOI):**  
  Drag and resize a region to receive detections from only a selected area of the scene.
- **Configure Minimum Object Size:**  
  Exclude detections smaller than the specified pixel area.

**Visualization Notes:**
- The overlay updates approximately two times per second (“best effort”). The bounding boxes may lag or not exactly match all detections due to UI and network constraints.
- Use this page for *quick confirmation* that detection is working and properly tuned.

***

### 2. MQTT

Here you configure the gateway between the camera and your backend system.

<img src="https://pandosme.github.io/assets/DetectX_MQTT.jpg" alt="MQTT Page" width="500"/>

- **Broker Address and Port:**  
  Specify the IP or hostname for your MQTT broker and port (default: 1883).
- **Authentication:**  
  Optional username and password if security is enforced.
- **Pre-topic:**  
  The prefix added to all MQTT topics (e.g., `detectx/detection/...`). Change if routing multiple cameras.
- **Additional Metadata:**  
  *Name* and *Location* properties help you distinguish events in multi-camera setups.

**Connection Status** is displayed, along with currently active parameters for fast troubleshooting.

***

### 3. Events/Labels

This section allows you to tailor detection and event signaling to your application:

<br><img src="https://pandosme.github.io/assets/DetectX_Evenst_Labels.jpg" alt="Detection Export Page" width="500"/><br>
*Note: Screenshot is based on DetectX COCO version.*  <br>

- **Selectable Labels:**  
  Check or uncheck which object types (labels) are actively processed, reducing false positives or narrowing the scope (e.g., only cars and persons).
- **Event State Settings:**  
  - *Prioritize*: Opt for accuracy (suppresses false triggers) or responsiveness.
  - *Minimum Event State Duration*: Avoid chattering by forcing a minimum active/inactive state period for each label.

**Note:**  
Each label produces an independent event state. Tuning event parameters is crucial for noisy or high-traffic scenes.

***

### 4. Detection Export

When downstream systems require not only detection data but *cropped images* for each detection:

<br><img src="https://pandosme.github.io/assets/DetectX_Detection-Export.jpg" alt="Detection Export Page" width="500"/><br>
*Note: Screenshot is based on DetectX COCO version.*  <br>

- **Enable/Disable Detection Cropping**
- **Set Border Adjustment:**  
  Expand or shrink the crop region around detected objects (e.g., add 25px margin).
- **Output Methods:**  
  - **MQTT:** Sends cropped images as base64 payloads.
  - **HTTP POST:** Posts the payload to a configurable endpoint.
- **Throttle Output:**  
  Limit image frequency to reduce load or network traffic.

#### View the Latest Crops
<br><img src="https://pandosme.github.io/assets/DetectX_crops.jpg" alt="Crops Gallery" width="600"/><br>
*Note: Screenshot is based on DetectX COCO version.*  <br>

- Opens a gallery of up to 10 most recent image crops, labeled by type and confidence.
- Essential for quality assurance—check that crops are readable, in correct locations, and correspond to real detections.

***

### 5. About

<br><img src="https://pandosme.github.io/assets/DetectX_About.jpg" alt="About Page" width="480"/><br>
*Note: Screenshot is based on DetectX COCO version.*  <br>

A dashboard combining:
- **Model Status:** Input size, inference time, DLPU backend, and status.
- **Device Details:** Camera type, firmware, serial, CPU & network usage.
- **MQTT Status:** Broker and topic configuration, connection health.
- **Application Info:** Name, version, vendor, support/documentation link.

Use this page as your *first check* when troubleshooting or confirming installation.

***

## Integration & Payload Examples

DetectX delivers three primary payload types, all enrichable with the configured device name, location, and serial for easy association in your backend systems.
<br>*Note: Payloads are based on DetectX COCO version.*  <br>

### 1. Detection (Bounding Box) on MQTT

**Topic:**  
`detectx/detection/<serial>`

**Example Payload:**
```json
{
  "detections": [
    {
      "label": "car",
      "c": 77,
      "x": 274,
      "y": 224,
      "w": 180,
      "h": 104,
      "timestamp": 1756453942980,
      "refId": 260
    }
  ],
  "name": "Front",
  "location": "",
  "serial": "B8A44F3024BB"
}
```

***

### 2. Event State on MQTT or ONVIF

**Topic:**  
`detectx/event/<serial>/<label>/<state>`

**Example Payload:**
```json
{
  "label": "car",
  "state": false,
  "timestamp": 1756453946184,
  "name": "Front",
  "location": "",
  "serial": "B8A44F3024BB"
}
```

***

### 3. Detection Crop Image

**MQTT/HTTP Topic or POST:**  
`detectx/crop/<serial>`

**Example Payload:**
```json
{
  "label": "truck",
  "timestamp": 1756454378759,
  "confidence": 47,
  "x": 25,
  "y": 25,
  "w": 218,
  "h": 106,
  "image": "/9j/4AAQSk...",  // JPEG in Base64
  "name": "Front",
  "location": "",
  "serial": "B8A44F3024BB"
}
```

***

## System Integrator Tips

- **Start with the About page** to confirm firmware, model, and MQTT status before field adjustments.
- Use Detection and Crops pages for rapid troubleshooting—verify detections visually before integrating triggers or actions.
- Use unique device names/locations in MQTT setup for scalable multi-camera deployments.
- Adjust event suppression and AOI settings based on site/scene context for best accuracy.

***

## Troubleshooting & Support

- If bounding boxes do not appear but the model status is OK, check confidence, AOI, and MQTT broker configuration.
- If crop images are misaligned or cut-off, adjust crop borders and AOI, validating via the “View the latest crops” gallery.
- Monitor CPU and network on the About page to avoid overload (especially on ARTPEC-8 devices).

***

### 3.5.2	Sep 10, 2025
- Fixed faulty input resolution for ARTPEC-9 version, resulting in very poor detection accuracy

### 3.5.1	Aug 28, 2025
- Bumbed to DetectX version 3.5.1 with adjustments and bug fixes

### 3.3.10
- Bumbed to DetectX version 3.3.10 with adjustments and bug fixes

### 3.3.8
- Bumbed to DetectX version 3.3.8 with adjustments and bug fixes

### 3.2.6
- Fixed a bug that could crash the application if the user interface is active for a longer period of time.  
  
### 3.2.5
- Fixed a serious memory leak

### 3.2.1
- Fixed a flaw that impact events
- Bumbed up to ACAP Wrapper 3.2.0

### 3.1.0	November 28, 2024
- Switched to latest SDK
  * Refactoring 
- Modified events to give all labels its own event
- Updated visualization in user interface
- Remove event labale counter

### 2.2.0	October 19, 2024
- Added event "Label Counter" for use cases needing to know how many objects are detected
- Fixed flaw for Detection transition

### 2.1.3	October 17, 2024
- Fixed model tflite export that resulted in very high (2s) inference time

### 2.1.1	October 13, 2024
- Fixed flawed event states
- Fixed potential memoryleak

### 2.1.0	October 11, 2024
- Added support for Detection transition

### 2.0.0	October 10, 2024
- Updated model with training with selected labes from Hagrid V2


