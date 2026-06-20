#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:1}"
EPOCHS="${EPOCHS:-5}"
SAVE_FREQ="${SAVE_FREQ:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"

DATA_ROOT="${DATA_ROOT:-/media/data/jjh/mydatasets}"
META_PATH="${META_PATH:-/media/data/jjh/mydatasets/mydatasets_meta.json}"
BACKBONE_NAME="${BACKBONE_NAME:-facebook/dinov3-vith16plus-pretrain-lvd1689m}"
OUT_DIR="${OUT_DIR:-experiments/residual-adapters-refined-mask/mydatasets/dinov3-vith16plus}"
RESULT_DIR="${RESULT_DIR:-${OUT_DIR}/results}"

echo "Running ProtoWD on mydatasets with DINOv3 ViT-H+"
echo "Device: ${DEVICE}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Backbone: ${BACKBONE_NAME}"
echo "Data root: ${DATA_ROOT}"
echo "Meta path: ${META_PATH}"
echo "Output dir: ${OUT_DIR}"

python train.py \
  --train_data_path "${DATA_ROOT}" \
  --train_meta_path "${META_PATH}" \
  --train_dataset mydatasets \
  --backbone_type dinov3 \
  --backbone_name "${BACKBONE_NAME}" \
  --epoch "${EPOCHS}" \
  --save_freq "${SAVE_FREQ}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --use_residual_adapters \
  --use_refined_mask \
  --no_iou_loss \
  --save_path "${OUT_DIR}"

python test.py \
  --test_data_path "${DATA_ROOT}" \
  --test_meta_path "${META_PATH}" \
  --test_dataset mydatasets \
  --checkpoint_path "${OUT_DIR}/checkpoints/final_model.pth" \
  --device "${DEVICE}" \
  --fast_metrics_stride 8 \
  --enable_analysis \
  --save_path "${RESULT_DIR}"
