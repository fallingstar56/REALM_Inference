#!/usr/bin/env bash
set -e -o pipefail

BYellow='\033[1;33m'
Color_Off='\033[0m'

# Parse the command line arguments.
GUI=true
DOCKER_DISPLAY=""
OMNIGIBSON_HEADLESS=1

# Parse command line arguments
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--headless)
        GUI=false
        shift
        ;;
        *)
        REALM_DATA_PATH="$1"
        shift
        ;;
    esac
done

echo "The NVIDIA Omniverse License Agreement (EULA) must be accepted before"
echo "Omniverse Kit can start. The license terms for this product can be viewed at"
echo "https://docs.omniverse.nvidia.com/app_isaacsim/common/NVIDIA_Omniverse_License_Agreement.html"

while true; do
    read -p "Do you accept the Omniverse EULA? [y/n] " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

if [ "$GUI" = true ] ; then
    xhost +local:root
    DOCKER_DISPLAY=$DISPLAY
    OMNIGIBSON_HEADLESS=0
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REALM_ROOT=$( cd -- "$( dirname -- "${SCRIPT_DIR}" )" &> /dev/null && pwd )

GR00T_DOCKER_ARGS=()
if [[ -n "${GR00T_ROOT:-}" ]]; then
    if [[ ! -d "$GR00T_ROOT" ]]; then
        echo "GR00T_ROOT is set but does not exist: $GR00T_ROOT" >&2
        exit 1
    fi
    GR00T_DOCKER_ARGS+=(
        -e "GR00T_ROOT=/app/Isaac-GR00T"
        -e "ISAAC_GR00T_ROOT=/app/Isaac-GR00T"
        -v "$GR00T_ROOT:/app/Isaac-GR00T:rw"
    )
fi

cd $REALM_ROOT
mkdir -p $REALM_DATA_PATH/isaac-sim/cache/kit
mkdir -p $REALM_DATA_PATH/isaac-sim/cache/ov
mkdir -p $REALM_DATA_PATH/isaac-sim/cache/pip
mkdir -p $REALM_DATA_PATH/isaac-sim/cache/glcache
mkdir -p $REALM_DATA_PATH/isaac-sim/cache/computecache
mkdir -p $REALM_DATA_PATH/isaac-sim/logs
mkdir -p $REALM_DATA_PATH/isaac-sim/config
mkdir -p $REALM_DATA_PATH/isaac-sim/data
mkdir -p $REALM_DATA_PATH/isaac-sim/documents

docker run \
    --gpus all \
    --privileged \
    -e DISPLAY=${DOCKER_DISPLAY} \
    -e OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS} \
    -e XDG_RUNTIME_DIR=/tmp/xdg-runtime \
    -e OMNI_KIT_ALLOW_ROOT=1 \
    -e TORCH_CUDA_ARCH_LIST="12.0" \
    -e CUDA_FORCE_PTX_JIT=1 \
    --tmpfs /tmp/xdg-runtime:rw,exec,nosuid,nodev,mode=700 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd):/app:rw \
    -v $REALM_DATA_PATH/datasets:/data \
    -v $REALM_DATA_PATH/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit:rw \
    -v $REALM_DATA_PATH/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v $REALM_DATA_PATH/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v $REALM_DATA_PATH/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v $REALM_DATA_PATH/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v $REALM_DATA_PATH/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v $REALM_DATA_PATH/isaac-sim/config:/root/.nvidia-omniverse/config:rw \
    -v $REALM_DATA_PATH/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v $REALM_DATA_PATH/isaac-sim/documents:/root/Documents:rw \
    -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
    "${GR00T_DOCKER_ARGS[@]}" \
    --network=host --rm -it realm
