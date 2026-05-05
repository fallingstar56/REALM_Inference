#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export REALM_ROOT="${REALM_ROOT:-$SCRIPT_DIR}"
export GR00T_N16_ROOT="${GR00T_N16_ROOT:-/home/xuhanyuan/Isaac-GR00T-n1.6-release}"
export REALM_DATA_PATH="${REALM_DATA_PATH:-/home/xuhanyuan/REALM_DATA}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export GR00T_N16_MODEL_PATH="${GR00T_N16_MODEL_PATH:-nvidia/GR00T-N1.6-DROID}"
export GR00T_N16_EMBODIMENT_TAG="${GR00T_N16_EMBODIMENT_TAG:-OXE_DROID}"
export GR00T_N16_DEVICE="${GR00T_N16_DEVICE:-cuda:0}"

HOST="${GR00T_HOST:-127.0.0.1}"
SERVER_HOST="${GR00T_SERVER_HOST:-0.0.0.0}"
PORT="${GR00T_PORT:-5555}"
MODEL_NAME="gr00t_n16"
MODEL_TYPE="GR00T_N16"
REPEATS="${REPEATS:-15}"
MAX_STEPS="${MAX_STEPS:-350}"
HORIZON="${HORIZON:-8}"
DOCKER_IMAGE="${REALM_DOCKER_IMAGE:-realm}"
MODEL_SERVER_TIMEOUT="${MODEL_SERVER_TIMEOUT:-900}"

TASK_IDS=(0 1 2 3 4 5 6 8 9)
PERTURBATION_IDS=(1 3 4 5 15)

SERVER_PID=""
SERVER_LOG="${SERVER_LOG:-$REALM_ROOT/logs/gr00t_n16_server_$(date +%Y%m%d_%H%M%S).log}"

usage() {
	cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run REALM with GR00T N1.6 for task IDs: ${TASK_IDS[*]}
and perturbation IDs: ${PERTURBATION_IDS[*]}.

Options:
  --no-uv-sync          Skip 'uv sync --python 3.10' before starting GR00T N1.6.
  --reuse-server        Use an existing N1.6 server on ${HOST}:${PORT}; do not start one.
  --help                Show this help.

Useful environment overrides:
  REALM_ROOT                Default: $REALM_ROOT
  GR00T_N16_ROOT            Default: $GR00T_N16_ROOT
  REALM_DATA_PATH           Default: $REALM_DATA_PATH
  GR00T_N16_MODEL_PATH      Default: $GR00T_N16_MODEL_PATH
  GR00T_N16_DEVICE          Default: $GR00T_N16_DEVICE
  GR00T_PORT                Default: $PORT
  REALM_DOCKER_IMAGE        Default: $DOCKER_IMAGE
  OMNIVERSE_EULA_ACCEPTED=1 skips the Omniverse EULA prompt.
EOF
}

SKIP_UV_SYNC="${SKIP_UV_SYNC:-0}"
REUSE_SERVER="${REUSE_GR00T_N16_SERVER:-0}"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--no-uv-sync)
			SKIP_UV_SYNC=1
			shift
			;;
		--reuse-server)
			REUSE_SERVER=1
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "Unknown option: $1" >&2
			usage >&2
			exit 1
			;;
	esac
done

require_command() {
	local cmd="$1"
	if ! command -v "$cmd" >/dev/null 2>&1; then
		echo "Missing required command: $cmd" >&2
		exit 1
	fi
}

port_is_listening() {
	nc -z "$HOST" "$PORT" >/dev/null 2>&1
}

describe_port_listener() {
	if command -v ss >/dev/null 2>&1; then
		ss -ltnp "sport = :$PORT" 2>/dev/null || true
	elif command -v lsof >/dev/null 2>&1; then
		lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true
	fi
}

wait_for_server() {
	local elapsed=0
	local interval=5

	echo "Waiting for GR00T N1.6 server on ${HOST}:${PORT} (timeout: ${MODEL_SERVER_TIMEOUT}s)"
	until port_is_listening; do
		if (( elapsed >= MODEL_SERVER_TIMEOUT )); then
			echo "GR00T N1.6 server did not become ready within ${MODEL_SERVER_TIMEOUT}s." >&2
			echo "Server log: $SERVER_LOG" >&2
			exit 1
		fi
		sleep "$interval"
		elapsed=$((elapsed + interval))
	done
	echo "GR00T N1.6 server is ready on ${HOST}:${PORT}."
}

cleanup() {
	if [[ -n "$SERVER_PID" ]]; then
		echo "Stopping GR00T N1.6 server (pid: $SERVER_PID)."
		kill -TERM "-$SERVER_PID" >/dev/null 2>&1 || kill "$SERVER_PID" >/dev/null 2>&1 || true
		sleep 2
		kill -KILL "-$SERVER_PID" >/dev/null 2>&1 || true
		wait "$SERVER_PID" >/dev/null 2>&1 || true
	fi
}
trap cleanup EXIT

require_command nc
require_command docker

if [[ ! -d "$REALM_ROOT" ]]; then
	echo "REALM_ROOT does not exist: $REALM_ROOT" >&2
	exit 1
fi

if [[ ! -d "$GR00T_N16_ROOT" ]]; then
	echo "GR00T_N16_ROOT does not exist: $GR00T_N16_ROOT" >&2
	exit 1
fi

if [[ ! -d "$REALM_DATA_PATH" ]]; then
	echo "REALM_DATA_PATH does not exist: $REALM_DATA_PATH" >&2
	exit 1
fi

if [[ ! -f "$GR00T_N16_ROOT/gr00t/eval/run_gr00t_server.py" ]]; then
	echo "GR00T N1.6 server entrypoint not found under GR00T_N16_ROOT: $GR00T_N16_ROOT" >&2
	exit 1
fi

if [[ "${OMNIVERSE_EULA_ACCEPTED:-0}" != "1" ]]; then
	if [[ ! -t 0 ]]; then
		echo "Non-interactive shell detected. Set OMNIVERSE_EULA_ACCEPTED=1 after accepting the NVIDIA Omniverse EULA." >&2
		exit 1
	fi

	echo "The NVIDIA Omniverse License Agreement (EULA) must be accepted before Omniverse Kit can start."
	echo "License terms: https://docs.omniverse.nvidia.com/app_isaacsim/common/NVIDIA_Omniverse_License_Agreement.html"
	while true; do
		read -r -p "Do you accept the Omniverse EULA? [y/n] " answer
		case "$answer" in
			[Yy]*) export OMNIVERSE_EULA_ACCEPTED=1; break ;;
			[Nn]*) exit 1 ;;
			*) echo "Please answer yes or no." ;;
		esac
	done
fi

mkdir -p "$REALM_ROOT/logs"

if [[ "$REUSE_SERVER" == "1" ]]; then
	if ! port_is_listening; then
		echo "--reuse-server was set, but no server is listening on ${HOST}:${PORT}." >&2
		exit 1
	fi
	echo "Reusing existing GR00T N1.6 server on ${HOST}:${PORT}."
	echo "Make sure it was started with --embodiment-tag OXE_DROID --use_sim_policy_wrapper."
elif port_is_listening; then
	echo "A process is already listening on ${HOST}:${PORT}." >&2
	describe_port_listener >&2
	echo "run16.sh will not reuse an existing server automatically, because an N1.7 server returns eef_9d modality keys and breaks the N1.6 adapter." >&2
	echo "Stop the existing process, choose another GR00T_PORT, or pass --reuse-server only if it is already an N1.6 server started with:" >&2
	echo "  --embodiment-tag OXE_DROID --use_sim_policy_wrapper" >&2
	exit 1
else
	require_command uv

	if [[ "$SKIP_UV_SYNC" != "1" ]]; then
		echo "Running uv sync for GR00T N1.6 environment."
		(cd "$GR00T_N16_ROOT" && uv sync --python 3.10)
	fi

	echo "Starting GR00T N1.6 server. Log: $SERVER_LOG"
	pushd "$GR00T_N16_ROOT" >/dev/null
	setsid uv run python gr00t/eval/run_gr00t_server.py \
		--model-path "$GR00T_N16_MODEL_PATH" \
		--embodiment-tag "$GR00T_N16_EMBODIMENT_TAG" \
		--use_sim_policy_wrapper \
		--host "$SERVER_HOST" \
		--port "$PORT" \
		--device "$GR00T_N16_DEVICE" \
		>"$SERVER_LOG" 2>&1 &
	SERVER_PID=$!
	popd >/dev/null
	wait_for_server
fi

mkdir -p "$REALM_DATA_PATH/isaac-sim/cache/kit"
mkdir -p "$REALM_DATA_PATH/isaac-sim/cache/ov"
mkdir -p "$REALM_DATA_PATH/isaac-sim/cache/pip"
mkdir -p "$REALM_DATA_PATH/isaac-sim/cache/glcache"
mkdir -p "$REALM_DATA_PATH/isaac-sim/cache/computecache"
mkdir -p "$REALM_DATA_PATH/isaac-sim/logs"
mkdir -p "$REALM_DATA_PATH/isaac-sim/config"
mkdir -p "$REALM_DATA_PATH/isaac-sim/data"
mkdir -p "$REALM_DATA_PATH/isaac-sim/documents"

DOCKER_ARGS=(
	--gpus all
	--privileged
	-e OMNIGIBSON_HEADLESS=1
	-e XDG_RUNTIME_DIR=/tmp/xdg-runtime
	-e OMNI_KIT_ALLOW_ROOT=1
	-e TORCH_CUDA_ARCH_LIST="12.0"
	-e CUDA_FORCE_PTX_JIT=1
	-e GR00T_N16_ROOT=/app/Isaac-GR00T-n1.6-release
	-e ISAAC_GR00T_N16_ROOT=/app/Isaac-GR00T-n1.6-release
	--tmpfs /tmp/xdg-runtime:rw,exec,nosuid,nodev,mode=700
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw
	-v "$REALM_ROOT:/app:rw"
	-v "$REALM_DATA_PATH/datasets:/data"
	-v "$REALM_DATA_PATH/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit:rw"
	-v "$REALM_DATA_PATH/isaac-sim/cache/ov:/root/.cache/ov:rw"
	-v "$REALM_DATA_PATH/isaac-sim/cache/pip:/root/.cache/pip:rw"
	-v "$REALM_DATA_PATH/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw"
	-v "$REALM_DATA_PATH/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw"
	-v "$REALM_DATA_PATH/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw"
	-v "$REALM_DATA_PATH/isaac-sim/config:/root/.nvidia-omniverse/config:rw"
	-v "$REALM_DATA_PATH/isaac-sim/data:/root/.local/share/ov/data:rw"
	-v "$REALM_DATA_PATH/isaac-sim/documents:/root/Documents:rw"
	-v "$GR00T_N16_ROOT:/app/Isaac-GR00T-n1.6-release:rw"
	--network=host
	--rm
)

if [[ -e /usr/share/nvidia/nvoptix.bin ]]; then
	DOCKER_ARGS+=( -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro )
fi

echo "Starting one REALM Docker container and running all evaluations."
docker run "${DOCKER_ARGS[@]}" "$DOCKER_IMAGE" bash -lc "
set -euo pipefail
cd /app
export GR00T_N16_ROOT=/app/Isaac-GR00T-n1.6-release
export ISAAC_GR00T_N16_ROOT=/app/Isaac-GR00T-n1.6-release
export OMNIGIBSON_HEADLESS=1

pip install pyzmq==27.0.1

for task_id in ${TASK_IDS[*]}; do
	for perturbation_id in ${PERTURBATION_IDS[*]}; do
		experiment_name=Task\${task_id}P\${perturbation_id}
		echo \"===== Running \${experiment_name} =====\"
		python examples/02_evaluate.py \
			--perturbation_id \"\${perturbation_id}\" \
			--task_id \"\${task_id}\" \
			--repeats \"$REPEATS\" \
			--max_steps \"$MAX_STEPS\" \
			--horizon \"$HORIZON\" \
			--model_name \"$MODEL_NAME\" \
			--model_type \"$MODEL_TYPE\" \
			--host \"$HOST\" \
			--port \"$PORT\" \
			--experiment_name \"\${experiment_name}\" \
			--save_mp4
	done
done
"

echo "All GR00T N1.6 REALM evaluations finished."
