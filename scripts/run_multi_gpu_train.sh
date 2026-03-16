#!/bin/bash
set -e

export NCCL_CUMEM_HOST_ENABLE=0  # workaround for NCCL errors, especially in containers
export CUDA_VISIBLE_DEVICES=0,1

MODEL_CFG=${MODEL_CFG:-"configs/models/llama32.yaml"}
DATA_CFG=${DATA_CFG:-"configs/data/data.yaml"}
TRAIN_CFG=${TRAIN_CFG:-"configs/train/train.yaml"}

ACCELERATE_CFG=${ACCELERATE_CFG:-"configs/accelerate/accelerate_multi_gpu.yaml"}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29502}

LOG_DIR=${LOG_DIR:-"logs"}
RUN_NAME=${RUN_NAME:-"train_$(date +%Y%m%d_%H%M%S)"}
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

# Tensorboard
TB_DIR=${TB_DIR:-"runs"}
mkdir -p "${TB_DIR}"
export TENSORBOARD_LOGGING_DIR="${TB_DIR}/${RUN_NAME}"

# W&B
WANDB_API_KEY_FROM_CFG=$(grep "^wandb_api_key:" "${TRAIN_CFG}" | awk '{print $2}' || true)
if [ -n "${WANDB_API_KEY_FROM_CFG}" ]; then
  export WANDB_API_KEY="${WANDB_API_KEY_FROM_CFG}"
fi

WANDB_PROJECT_FROM_CFG=$(grep "^wandb_project:" "${TRAIN_CFG}" | awk '{print $2}' || true)
if [ -n "${WANDB_PROJECT_FROM_CFG}" ]; then
  export WANDB_PROJECT="${WANDB_PROJECT_FROM_CFG}"
fi

export WANDB_NAME="${RUN_NAME}"

echo "===================== Training Configuration ====================="
echo "MODEL_CFG              : ${MODEL_CFG}"
echo "DATA_CFG               : ${DATA_CFG}"
echo "TRAIN_CFG              : ${TRAIN_CFG}"
echo "ACCELERATE_CFG         : ${ACCELERATE_CFG}"
echo "MAIN_PROCESS_PORT      : ${MAIN_PROCESS_PORT}"
echo "RUN_NAME               : ${RUN_NAME}"
echo "LOG_FILE               : ${LOG_FILE}"
echo "WANDB_PROJECT          : ${WANDB_PROJECT:-not set}"
echo "WANDB_NAME             : ${WANDB_NAME}"
echo "TENSORBOARD_LOGGING_DIR: ${TENSORBOARD_LOGGING_DIR}"
echo "CUDA_VISIBLE_DEVICES   : ${CUDA_VISIBLE_DEVICES:-not set}"
echo "=================================================================="

accelerate launch \
  --config_file "${ACCELERATE_CFG}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  main.py \
  --model_config "${MODEL_CFG}" \
  --data_config "${DATA_CFG}" \
  --train_config "${TRAIN_CFG}"