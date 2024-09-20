#!/usr/bin/env bash

set -exu

CACHE_PATH="/home/${USER}/.cache/pkm"
# DATA_PATH="/path/to/data/"
DATA_PATH="/home/jh/workspace/data/pkm"

# replace /tmp/docker
# SHARE_PATH="/path/to/share/"
SHARE_PATH="/home/jh/workspace/isaaclab/gate"

# x11 authority
__ISAACLAB_TMP_XAUTH=$(mktemp --suffix=".xauth")
xauth_cookie= xauth nlist ${DISPLAY} | sed -e s/^..../ffff/ | xauth -f $__ISAACLAB_TMP_XAUTH nmerge -


SCRIPT_DIR="$( cd "$( dirname $(realpath "${BASH_SOURCE[0]}") )" && pwd )"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

mkdir -p "${CACHE_PATH}"


docker run -it -e DISPLAY -e TERM -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=${__ISAACLAB_TMP_XAUTH} -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ${__ISAACLAB_TMP_XAUTH}:${__ISAACLAB_TMP_XAUTH} -v /etc/localtime:/etc/localtime:ro \
    --mount type=bind,source="${REPO_ROOT}",target="/home/user/$(basename ${REPO_ROOT})" \
    --mount type=bind,source="${DATA_PATH}",target="/input" \
    --mount type=bind,source="${CACHE_PATH}",target="/home/user/.cache/pkm" \
    --mount type=bind,source="${SHARE_PATH}",target="/gate" \
    --shm-size=32g --network host --privileged --gpus all isaaclab
