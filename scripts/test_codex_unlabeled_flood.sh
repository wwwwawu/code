#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:1}"
IMAGE_DIR="${IMAGE_DIR:-/media/data/jjh/datasets/城市内涝识别数据}"
SAVE_PATH="${SAVE_PATH:-experiments/residual-adapters-refined-mask/codex-test}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-experiments/residual-adapters-refined-mask/uwbench-epoch5/dinov3-vit-l/checkpoints/final_model.pth}"

echo "Running unlabeled urban flooding test"
echo "Device: ${DEVICE}"
echo "Image dir: ${IMAGE_DIR}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Save path: ${SAVE_PATH}"

python test_unlabeled.py \
  --image_dir "${IMAGE_DIR}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --device "${DEVICE}" \
  --save_path "${SAVE_PATH}" \
  --test_dataset codex-test

echo "Unlabeled urban flooding test completed."
