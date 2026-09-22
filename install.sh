#!/bin/sh -eu
#
# Install a HandGesture package on an Axis camera.
#
#   ./install.sh <camera-host> [a8|a9]
#
# The chip defaults to a8. Credentials come from the environment so they are
# never committed:
#
#   export AXIS_USER=youruser
#   read -rs AXIS_PASS && export AXIS_PASS     # typed, not echoed
#
# ARTPEC-8 and ARTPEC-9 packages are NOT interchangeable: the A8 build is
# quantized per-tensor, which the ARTPEC-8 DLPU requires, while A9 uses
# per-channel weights. Installing the wrong one gives wrong detections.

usage() { echo "Usage: $0 <camera-host> [a8|a9]" >&2; exit 2; }

[ "$#" -ge 1 ] || usage
CAMERA_HOST=$1
CHIP=${2:-a8}

case "$CHIP" in
	a8) SUFFIX=artpec8 ;;
	a9) SUFFIX=artpec9 ;;
	*)  echo "Invalid chip: $CHIP (expected a8 or a9)" >&2; usage ;;
esac

EAP_FILE=$(find . -maxdepth 1 -name "*_${SUFFIX}.eap" 2>/dev/null | sort | tail -n 1)
if [ -z "$EAP_FILE" ]; then
	echo "ERROR: no *_${SUFFIX}.eap in the current directory -- run ./build.sh --target $CHIP" >&2
	exit 1
fi

USERNAME=${AXIS_USER:-}
PASSWORD=${AXIS_PASS:-}
if [ -z "$USERNAME" ]; then printf 'Axis username: '; read -r USERNAME; fi
if [ -z "$PASSWORD" ]; then
	printf 'Axis password: '
	stty -echo 2>/dev/null || true
	read -r PASSWORD
	stty echo 2>/dev/null || true
	echo
fi

echo "Installing $EAP_FILE on $CAMERA_HOST ($CHIP)..."
curl --fail --digest -u "$USERNAME:$PASSWORD" \
	-F "packfil=@$EAP_FILE;type=application/octet-stream" \
	"http://$CAMERA_HOST/axis-cgi/applications/upload.cgi"

echo
echo 'Done. Check the log with:'
echo "  curl --digest -u \"\$AXIS_USER:\$AXIS_PASS\" 'http://$CAMERA_HOST/axis-cgi/admin/systemlog.cgi?appname=handgesture'"
