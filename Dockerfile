ARG ARCH=aarch64
ARG VERSION=12.11.0
ARG UBUNTU_VERSION=24.04
ARG REPO=docker.io/axisecp
ARG SDK=acap-native-sdk

#-------------------------------------------------------------------------------
# Stage 1: TensorFlow environment (cached layer)
#-------------------------------------------------------------------------------
FROM ${REPO}/${SDK}:${VERSION}-${ARCH}-ubuntu${UBUNTU_VERSION} AS tensorflow-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment for installations using pip
RUN python3 -m venv /opt/venv

# Install the TensorFlow version validated by this repository (CACHED)
RUN . /opt/venv/bin/activate && pip install --no-cache-dir tensorflow==2.21.0

#-------------------------------------------------------------------------------
# Stage 2: Build ACAP application
#-------------------------------------------------------------------------------
FROM tensorflow-base

WORKDIR /opt/app

# Copy application source, headers, and prebuilt libraries
COPY ./app .

# Ensure local lib and include directories exist (if not already present)
RUN mkdir -p lib include

# Select chip-specific models while preserving the runtime's canonical paths.
# The hand detector comes in three input sizes (settings "detectorSize": Speed
# 960x544, Balanced 1120x640, Distance 1280x736). model/model.tflite is the chip's
# default: Distance on ARTPEC-9, Speed on ARTPEC-8 (with the camera's own object
# analytics running, the larger sizes drop an ARTPEC-8 below 5 FPS).
ARG TARGET_CHIP
RUN case "$TARGET_CHIP" in a8) DEFAULT=960x544 ;; a9) DEFAULT=1280x736 ;; *) echo 'TARGET_CHIP must be a8 or a9' >&2; exit 2 ;; esac \
    && for size in 960x544 1120x640 1280x736; do cp "model/model-${TARGET_CHIP}-${size}.tflite" "model/model-${size}.tflite"; done \
    && cp "model/model-${DEFAULT}.tflite" model/model.tflite \
    && python3 -c "import json,sys; p='settings/settings.json'; d=json.load(open(p)); d['detectorSize']=sys.argv[1]; open(p,'w').write(json.dumps(d, indent=4) + '\\n')" "$DEFAULT" \
    && cp "model/gesture-${TARGET_CHIP}.tflite" model/classifier.tflite \
    && cp model/gesture-labels.txt model/classifier-labels.txt \
    && if [ "$TARGET_CHIP" = a9 ]; then \
        python3 -c "import json; p='manifest.json'; d=json.load(open(p)); d['acapPackageConf']['setup']['compatibleOsVersions'][0]['min']='13.0'; open(p, 'w').write(json.dumps(d, indent=4) + '\\n')"; \
       fi

# Validate every detector size (and the classifier) and generate model_params.h for
# the default detector (validated last, so its header is the one kept).
RUN . /opt/venv/bin/activate \
    && for size in 960x544 1120x640 1280x736; do python extract_model_params.py "model/model-${size}.tflite" --target "$TARGET_CHIP" || exit 1; done \
    && python extract_model_params.py 'model/model.tflite' --target "$TARGET_CHIP"

# Build and package ACAP application with assets required by your app
RUN . /opt/axis/acapsdk/environment-setup* && acap-build . \
    -a 'settings/settings.json' \
    -a 'settings/events.json' \
    -a 'settings/mqtt.json' \
    -a 'model/model.tflite' \
    -a 'model/model-960x544.tflite' \
    -a 'model/model-1120x640.tflite' \
    -a 'model/model-1280x736.tflite' \
    -a 'model/labels.txt' \
    -a 'model/classifier.tflite' \
    -a 'model/classifier-labels.txt'
