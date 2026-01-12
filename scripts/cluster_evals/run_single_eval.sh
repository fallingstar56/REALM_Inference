#!/bin/bash
#SBATCH --job-name omnigibson-test
#SBATCH --partition=l40s
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --ntasks-per-node=1
#SBATCH --time 00-04:30:00

#---------------------------------------------------------------------------------

if [ -z "$1" ]; then
  echo "Usage: $0 <task_id> <repeats> <max_steps> <model>"
  exit 1
fi

echo $SLURM_ARRAY_TASK_ID
echo "debug.."

TASK_ID="$1"
PERTURBATION_ID="$2"
#PERTURBATION_ID="$SLURM_ARRAY_TASK_ID"
REPEATS=$3
MAX_STEPS=$4
POLICY_CONFIG="$5"
CHECKPOINT_PATH="$6"
BASE_PORT=$7
EXPERIMENT_NAME="$8"
RUN_ID="$9"
POLICY_RUN_DIR="${10}"

REALM_ROOT=$(pwd)

#---------------------------------------------------------------------------------

export HF_HOME=$REALM_ROOT/hf_cache
export HUGGINGFACE_HUB_CACHE=$REALM_ROOT/hf_cache
[[ -d "$HF_HOME" ]] || mkdir -p "$HF_HOME"

export XDG_CACHE_HOME=$REALM_ROOT/python_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25

POLICY_SCRIPT="${POLICY_RUN_DIR}/scripts/serve_policy.py" # TODO: assuming default openppi repo
port=$((BASE_PORT + SLURM_ARRAY_TASK_ID + 100 * TASK_ID))

cd "$POLICY_RUN_DIR"
uv run "$POLICY_SCRIPT" \
  --port=$port policy:checkpoint \
  --policy.config="$POLICY_CONFIG" \
  --policy.dir="$CHECKPOINT_PATH" & SERVER_PID=$!
sleep 60

#---------------------------------------------------------------------------------

cd $REALM_ROOT
mkdir -p "$REALM_ROOT/tmp"
mkdir -p "$REALM_ROOT/mamba_cache/$SLURM_JOB_ID"
mkdir -p "$REALM_ROOT/pip_cache/$SLURM_JOB_ID"
MODEL_NAME=$(basename "$(dirname "${CHECKPOINT_PATH%/}")")_$(basename "${CHECKPOINT_PATH%/}")

apptainer exec \
  --userns \
  --nv \
  --writable-tmpfs \
  --bind $(pwd):/app \
  --bind $REALM_DATA_PATH/datasets:/data \
  --bind $REALM_DATA_PATH/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit \
  --bind $REALM_DATA_PATH/isaac-sim/cache/ov:/root/.cache/ov \
  --bind $REALM_DATA_PATH/isaac-sim/cache/pip:/root/.cache/pip \
  --bind $REALM_DATA_PATH/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache \
  --bind $REALM_DATA_PATH/isaac-sim/cache/computecache:/root/.nv/ComputeCache \
  --bind $REALM_DATA_PATH/isaac-sim/logs:/root/.nvidia-omniverse/logs \
  --bind $REALM_DATA_PATH/isaac-sim/config:/root/.nvidia-omniverse/config \
  --bind $REALM_DATA_PATH/isaac-sim/data:/root/.local/share/ov/data \
  --bind $REALM_DATA_PATH/isaac-sim/documents:/root/Documents \
  --bind $REALM_ROOT/tmp:/tmp \
  --env TMPDIR=/tmp \
  --env OMNIGIBSON_HEADLESS=1 \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env MAMBA_CACHE_DIR=$REALM_ROOT/mamba_cache/$SLURM_JOB_ID \
  --env PIP_CACHE_DIR=$REALM_ROOT/pip_cache/$SLURM_JOB_ID \
  $REALM_SIF \
  micromamba run -n omnigibson python examples/02_eval_dynamic_scenes.py \
  --perturbation_id $PERTURBATION_ID \
  --task_id $TASK_ID \
  --repeats $REPEATS \
  --max_steps $MAX_STEPS \
  --model $MODEL_NAME \
  --port $port \
  --run_id $RUN_ID \
  --experiment_name $EXPERIMENT_NAME
