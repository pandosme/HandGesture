ARG ARCH=aarch64
ARG VERSION=12.2.0
ARG UBUNTU_VERSION=24.04
ARG REPO=axisecp
ARG SDK=acap-native-sdk

FROM ${REPO}/${SDK}:${VERSION}-${ARCH}-ubuntu${UBUNTU_VERSION}

#-------------------------------------------------------------------------------
# Build ACAP application
#-------------------------------------------------------------------------------

WORKDIR /opt/app
COPY ./app .
ARG CHIP=

RUN . /opt/axis/acapsdk/environment-setup* && acap-build . \
    -a 'model/model.tflite' \
	-a 'model/model.json' \	
	-a 'settings/settings.json' \
	-a 'settings/events.json' \
	-a 'settings/mqtt.json' \
