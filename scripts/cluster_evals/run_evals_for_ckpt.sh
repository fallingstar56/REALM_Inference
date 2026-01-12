#!/bin/bash
BASE_PORT=8000
MAX_STEPS=800
REPEATS=25
RUN_ID=$(date +%Y%m%d_%H%M%S)
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --policy_config) POLICY_CONFIG="$2"; shift 2 ;;
    --checkpoint_path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --policy_run_dir) POLICY_RUN_DIR="$2"; shift 2 ;;
    --base_port) BASE_PORT="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --experiment_name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --task_ids) T_RAW="$2"; IFS=',' read -ra TASK_IDS <<< "$2"; shift 2 ;;
    --perturbation_ids) P_IDS="$2"; shift 2 ;;
    *) shift ;;
  esac
done
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"t${T_RAW//,/_}_p${P_IDS//,/_}_s${MAX_STEPS}_r${REPEATS}"}
for i in "${TASK_IDS[@]}"; do
  for j in "${P_IDS[@]}"; do
    sbatch scripts/cluster_evals/run_single_eval.sh \
      "$i" \
      "$j" \
      "$REPEATS" \
      "$MAX_STEPS" \
      "$POLICY_CONFIG" \
      "$CHECKPOINT_PATH"\
      "$BASE_PORT" \
      "$EXPERIMENT_NAME" \
      "$RUN_ID" \
      "$POLICY_RUN_DIR"
  done
done