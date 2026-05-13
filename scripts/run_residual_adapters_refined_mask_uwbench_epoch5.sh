#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:1}"
EPOCHS="${EPOCHS:-5}"
SAVE_FREQ="${SAVE_FREQ:-5}"
ROOT="experiments/residual-adapters-refined-mask/uwbench-epoch${EPOCHS}"

echo "Running VisualAD + residual-adapter + refined-mask UW-Bench experiments"
echo "Device: ${DEVICE}"
echo "Epochs: ${EPOCHS}"
echo "Output root: ${ROOT}"

python train.py \
  --backbone_type dinov3 \
  --backbone_name facebook/dinov3-vitl16-pretrain-lvd1689m \
  --epoch "${EPOCHS}" \
  --save_freq "${SAVE_FREQ}" \
  --batch_size 1 \
  --device "${DEVICE}" \
  --use_residual_adapters \
  --use_refined_mask \
  --no_iou_loss \
  --save_path "${ROOT}/dinov3-vit-l"

python test.py \
  --checkpoint_path "${ROOT}/dinov3-vit-l/checkpoints/final_model.pth" \
  --device "${DEVICE}" \
  --enable_analysis

python train.py \
  --backbone_type sam \
  --backbone_name vit_l \
  --epoch "${EPOCHS}" \
  --save_freq "${SAVE_FREQ}" \
  --batch_size 1 \
  --device "${DEVICE}" \
  --use_residual_adapters \
  --use_refined_mask \
  --no_iou_loss \
  --save_path "${ROOT}/sam-vit-l"

python test.py \
  --checkpoint_path "${ROOT}/sam-vit-l/checkpoints/final_model.pth" \
  --device "${DEVICE}" \
  --enable_analysis

python train.py \
  --backbone_type clip \
  --backbone_name "ViT-L/14@336px" \
  --epoch "${EPOCHS}" \
  --save_freq "${SAVE_FREQ}" \
  --batch_size 1 \
  --device "${DEVICE}" \
  --use_residual_adapters \
  --use_refined_mask \
  --no_iou_loss \
  --save_path "${ROOT}/clip-vit-l"

python test.py \
  --checkpoint_path "${ROOT}/clip-vit-l/checkpoints/final_model.pth" \
  --device "${DEVICE}" \
  --enable_analysis

echo "All refined-mask runs completed."
