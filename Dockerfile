ARG ARCH=aarch64
ARG VERSION=12.5.0
ARG UBUNTU_VERSION=24.04
ARG REPO=axisecp
ARG SDK=acap-native-sdk

FROM ${REPO}/${SDK}@sha256:af15866817a654d464ca3fc1e4dc3db0fbcb8803c3b63139f2d2ef2f462665b5

#-------------------------------------------------------------------------------
# Build ACAP application
#-------------------------------------------------------------------------------

WORKDIR /opt/app

# Copy application files
COPY ./app .
ARG CHIP=

RUN . /opt/axis/acapsdk/environment-setup* && acap-build . \
    -a 'model/model.tflite' \
    -a 'model/model.json' \
    -a 'settings/settings.json' \
    -a 'settings/events.json' \
    -a 'settings/mqtt.json'
