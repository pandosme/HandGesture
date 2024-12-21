# Hand Gesture

This ACAP is based on [DetectX](https://github.com/pandosme/DetectX), an open-source package.
The model is trained on selected labels in the [Hagird V2](https://github.com/hukenovs/hagrid) dataset.  

![gestures](https://raw.githubusercontent.com/hukenovs/hagrid/Hagrid_v1/images/gestures.jpg)


# Pre-requsite
- Axis Camera based on ARTPEC-8.  A special firmware for ARTPEC-7 having a TPU can be requested.

# User interface
The user interface is designed to validate detection and apply various filters.

## Detections
The 10 latest detections is shown in video as bounding box and table.  The events are shown in a separate table.

### Confidence
Initial filter to reduce the number of false detection. 

### Set Area of Intrest
Additional filter to reduce the number of false detection. Click button and use mouse to define an area that the center of the detection must be within.

### Set Minimum Size
Additional filter to reduce the number of false detection. Click button and use mouse to define a minimum width and height that the detection must have.

## Advanced
Additional filters to apply on the detection and output.

### Detection transition
A minumum time that the detection must be stable before an event is fired.  It define how trigger-happy the evant shall be.

### Min event state duration
The minumum event duration a detection may have.  

### Labels Processed
Enable or disable selected gestures.

## Integration
The service fires two different events targeting different use cases.  Service may monitor these event using camera event syste, ONVIF event stream and MQTT.
## Label state
A stateful event (high/low) for each detected label.  The event includes property state (true/false).  

# History

### 2.0.0	October 10, 2024
- Updated model with training with selected labes from Hagrid V2

### 2.1.0	October 11, 2024
- Added support for Detection transition

### 2.1.1	October 13, 2024
- Fixed flawed event states
- Fixed potential memoryleak

### 2.1.3	October 17, 2024
- Fixed model tflite export that resulted in very high (2s) inference time

### 2.2.0	October 19, 2024
- Added event "Label Counter" for use cases needing to know how many objects are detected
- Fixed flaw for Detection transition

### 3.1.0	November 28, 2024
- Switched to latest SDK
  * Refactoring 
- Modified events to give all labels its own event
- Updated visualization in user interface
- Remove event labale counter

### 3.2.1
- Fixed a flaw that impact events
- Bumbed up to ACAP Wrapper 3.2.0


