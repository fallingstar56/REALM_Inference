#!/bin/bash
#SBATCH --job-name omnigibson-test
#SBATCH --partition l40s
#SBATCH --gpus 1
#SBATCH --mem 120G
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-gpu 64
#SBATCH --time 00-04:30:00

#---------------------------------------------------------------------------------

REALM_ROOT=$(pwd)
RUN_ID=$(date +%Y%m%d_%H%M%S)
DEBUG=false
RENDERING_MODE="rt"
MULTI_VIEW_FLAG=""
RESUME_FLAG=""
TASK_CFG_PATH=""
NO_RENDER_FLAG=""
ROBOT_FLAG=""
BASE_PORT=8000
GR00T_SERVER_ENTRYPOINT="gr00t/eval/run_gr00t_server.py"
GR00T_EMBODIMENT_TAG="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
GR00T_SERVER_HOST="0.0.0.0"
GR00T_SERVER_DEVICE="cuda:0"
MODEL_SERVER_TIMEOUT="${MODEL_SERVER_TIMEOUT:-180}"

PORT_CHECK_CMD=""
if command -v nc >/dev/null 2>&1; then
  PORT_CHECK_CMD="nc"
elif command -v ss >/dev/null 2>&1; then
  PORT_CHECK_CMD="ss"
else
  echo "Need either 'nc' or 'ss' available to check the port."
  exit 1
fi

wait_for_port() {
  local port="$1"
  local timeout="${2:-180}"
  local interval=2
  local elapsed=0

  while (( elapsed < timeout )); do
    case "$PORT_CHECK_CMD" in
      nc)
        if nc -z localhost "$port" 2>/dev/null; then
          return 0
        fi
        ;;
      ss)
        if ss -ltn "sport = :$port" 2>/dev/null | grep -q "$port"; then
          return 0
        fi
        ;;
    esac
    sleep "$interval"
    elapsed=$(( elapsed + interval ))
  done

  return 1
}

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --policy_config) POLICY_CONFIG="$2"; shift 2 ;;
    --checkpoint_path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --policy_run_dir) POLICY_RUN_DIR="$2"; shift 2 ;;
    --base_port|--base-port) BASE_PORT="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --experiment_name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --task_id) TASK_ID="$2"; shift 2 ;;
    --task_cfg_path) TASK_CFG_PATH="$2"; shift 2 ;;
    --perturbation_id) PERTURBATION_ID="$2"; shift 2 ;;
    --run_id) RUN_ID="$2"; shift 2 ;;
    --model_type) MODEL_TYPE="$2"; shift 2 ;;
    --debug) DEBUG=true; shift 1;;
    --rendering_mode) RENDERING_MODE="$2"; shift 2 ;;
    --multi-view) MULTI_VIEW_FLAG="--multi-view"; shift 1;;
    --resume) RESUME_FLAG="--resume"; shift 1;;
    --no_render) NO_RENDER_FLAG="--no_render"; shift 1;;
    --robot) ROBOT_FLAG="--robot $2"; shift 2 ;;
    *) shift ;;
  esac
done



#---------------------------------------------------------------------------------

export HF_HOME=$REALM_ROOT/hf_cache
export HUGGINGFACE_HUB_CACHE=$REALM_ROOT/hf_cache
[[ -d "$HF_HOME" ]] || mkdir -p "$HF_HOME"

export XDG_CACHE_HOME=$REALM_ROOT/python_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25

port=$((BASE_PORT + PERTURBATION_ID + 100 * TASK_ID))

if [ "$DEBUG" = "false" ]; then
  if [ "$MODEL_TYPE" = "openpi" ]; then
    POLICY_SIF="/scratch/project/open-34-32/sedlam/projects/REALM_openpi/uv_cuda128.sif"
    cd "$POLICY_RUN_DIR" || exit
    apptainer exec \
      --writable-tmpfs \
      --nv \
      --bind /scratch \
      --bind "$(pwd)":/app \
      --bind $CHECKPOINT_PATH:/checkpoint \
      --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
      --env XDG_CACHE_HOME=$XDG_CACHE_HOME \
      --env GIT_LFS_SKIP_SMUDGE=1 \
      $POLICY_SIF uv run /app/scripts/serve_policy.py \
        --port=$port \
        policy:checkpoint \
        --policy.config=$POLICY_CONFIG \
        --policy.dir=/checkpoint & SERVER_PID=$!
    sleep 120
  elif [ "$MODEL_TYPE" = "molmoact" ]; then
    POLICY_SIF="/scratch/project/open-34-32/sedlam/projects/molmoact/apptainer/molmoact.sif"
    cd "$POLICY_RUN_DIR" || exit
    apptainer exec \
      --writable-tmpfs \
      --nv \
      --bind /scratch \
      --bind "$(pwd)":/app \
      --bind $CHECKPOINT_PATH:/checkpoint \
      $POLICY_SIF /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate && pip install tyro && pip install /app/packages/openpi-client && python /app/inference/run_molmoact_server.py --port=${port}"
    sleep 120
  elif [ "$MODEL_TYPE" == "GR00T" ]; then
    if [[ -z "${POLICY_RUN_DIR:-}" ]]; then
      echo "POLICY_RUN_DIR is not set for GR00T. Pass the Isaac-GR00T root with --policy_run_dir."
      exit 1
    fi
    if [[ ! -f "$POLICY_RUN_DIR/$GR00T_SERVER_ENTRYPOINT" ]]; then
      echo "POLICY_RUN_DIR does not contain $GR00T_SERVER_ENTRYPOINT: $POLICY_RUN_DIR"
      exit 1
    fi
    cd "$POLICY_RUN_DIR" || exit
    uv run python "$GR00T_SERVER_ENTRYPOINT" \
      --model-path "$CHECKPOINT_PATH" \
      --embodiment-tag "$GR00T_EMBODIMENT_TAG" \
      --host "$GR00T_SERVER_HOST" \
      --port "$port" \
      --device "$GR00T_SERVER_DEVICE" & SERVER_PID=$!

    echo "Waiting for the GR00T model server to start (maximum: ${MODEL_SERVER_TIMEOUT}s)"
    if wait_for_port "$port" "$MODEL_SERVER_TIMEOUT"; then
      echo "GR00T server is listening on port ${port}"
    else
      echo "GR00T server did not start listening on ${port} within ${MODEL_SERVER_TIMEOUT}s"
      kill "$SERVER_PID" 2>/dev/null || true
      exit 1
    fi
  fi
fi

#---------------------------------------------------------------------------------

cd $REALM_ROOT || exit
mkdir -p "$REALM_ROOT/tmp/$SLURM_JOB_ID"
mkdir -p "$REALM_ROOT/mamba_cache/$SLURM_JOB_ID"
mkdir -p "$REALM_ROOT/pip_cache/$SLURM_JOB_ID"

if [ "$DEBUG" = "true" ]; then
  MODEL_NAME="debug"
elif [ "$MODEL_TYPE" = "molmoact" ]; then
  MODEL_NAME="molmoact"
else
  CLEAN_PATH="${CHECKPOINT_PATH%/}"
  MODEL_NAME=$(basename "$(dirname "${CLEAN_PATH%/}")")_$(basename "${CLEAN_PATH%/}")
fi

if [ -n "$TASK_CFG_PATH" ]; then
  TASK_CFG_ARG="--task_cfg_path $TASK_CFG_PATH"
else
  TASK_CFG_ARG=""
fi

apptainer exec \
  --userns \
  --nv \
  --writable-tmpfs \
  --bind "$(pwd)":/app \
  --bind "$REALM_DATA_PATH"/datasets:/data \
  --bind "$REALM_DATA_PATH"/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit \
  --bind "$REALM_DATA_PATH"/isaac-sim/cache/ov:/root/.cache/ov \
  --bind "$REALM_DATA_PATH"/isaac-sim/cache/pip:/root/.cache/pip \
  --bind "$REALM_DATA_PATH"/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache \
  --bind "$REALM_DATA_PATH"/isaac-sim/cache/computecache:/root/.nv/ComputeCache \
  --bind "$REALM_DATA_PATH"/isaac-sim/logs:/root/.nvidia-omniverse/logs \
  --bind "$REALM_DATA_PATH"/isaac-sim/config:/root/.nvidia-omniverse/config \
  --bind "$REALM_DATA_PATH"/isaac-sim/data:/root/.local/share/ov/data \
  --bind "$REALM_DATA_PATH"/isaac-sim/documents:/root/Documents \
  --bind "$REALM_ROOT"/tmp/"$SLURM_JOB_ID":/tmp \
  --env TMPDIR=/tmp \
  --env OMNIGIBSON_HEADLESS=1 \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env MAMBA_CACHE_DIR="$REALM_ROOT"/mamba_cache/"$SLURM_JOB_ID" \
  --env PIP_CACHE_DIR="$REALM_ROOT"/pip_cache/"$SLURM_JOB_ID" \
  $REALM_SIF \
  micromamba run -n omnigibson python examples/02_evaluate.py \
  --perturbation_id $PERTURBATION_ID \
  --task_id $TASK_ID \
  $TASK_CFG_ARG \
  --repeats $REPEATS \
  --max_steps $MAX_STEPS \
  --model_name $MODEL_NAME \
  --model_type $MODEL_TYPE \
  --port $port \
  --run_id $RUN_ID \
  --experiment_name $EXPERIMENT_NAME \
  --rendering_mode $RENDERING_MODE \
  $MULTI_VIEW_FLAG \
  $RESUME_FLAG \
  $NO_RENDER_FLAG \
  $ROBOT_FLAG

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "Job finished successfully. Cleaning up..."
  rm -rf "$REALM_ROOT/tmp/$SLURM_JOB_ID"
  rm -rf "$REALM_ROOT/mamba_cache/$SLURM_JOB_ID"
  rm -rf "$REALM_ROOT/pip_cache/$SLURM_JOB_ID"
else
  echo "Job failed (exit code $EXIT_CODE). Preserving temporary directories for debugging."
fi

exit $EXIT_CODE
