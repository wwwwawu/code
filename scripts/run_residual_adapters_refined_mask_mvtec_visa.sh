#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SAVE_FREQ="${SAVE_FREQ:-1}"
MVTEC_PATH="${MVTEC_PATH:-/media/data/jjh/datasets/mvtec}"
VISA_PATH="${VISA_PATH:-/media/data/jjh/datasets/VisA}"
ROOT="${ROOT:-results}"
META_DIR="${META_DIR:-data_meta}"
BACKBONE_TYPE="${BACKBONE_TYPE:-clip}"
BACKBONE_NAME="${BACKBONE_NAME:-ViT-L/14@336px}"

echo "Running VisualAD + residual-adapter + refined-mask on MVTec/VisA"
echo "Device: ${DEVICE}"
echo "Batch size: ${BATCH_SIZE}"
echo "MVTec path: ${MVTEC_PATH}"
echo "VisA path: ${VISA_PATH}"
echo "Backbone: ${BACKBONE_TYPE}:${BACKBONE_NAME}"
echo "Output root: ${ROOT}"
echo "Meta dir: ${META_DIR}"

mkdir -p "${META_DIR}"
python generate_dataset_json/mvtec.py --root "${MVTEC_PATH}" --meta_path "${META_DIR}/mvtec_meta.json"
python generate_dataset_json/visa.py --root "${VISA_PATH}" --meta_path "${META_DIR}/visa_meta.json"

python train.py \
  --train_data_path "${MVTEC_PATH}" \
  --train_meta_path "${META_DIR}/mvtec_meta.json" \
  --train_dataset mvtec \
  --backbone_type "${BACKBONE_TYPE}" \
  --backbone_name "${BACKBONE_NAME}" \
  --epoch 2 \
  --save_freq "${SAVE_FREQ}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --use_residual_adapters \
  --adapter_type mlp \
  --use_refined_mask \
  --no_iou_loss \
  --save_path "${ROOT}/mvtec"

python test.py \
  --test_data_path "${VISA_PATH}" \
  --test_meta_path "${META_DIR}/visa_meta.json" \
  --test_dataset visa \
  --checkpoint_path "${ROOT}/mvtec/checkpoints/final_model.pth" \
  --device "${DEVICE}" \
  --fast_metrics_stride 8 \
  --save_path "${ROOT}/mvtec/results0"

python train.py \
  --train_data_path "${VISA_PATH}" \
  --train_meta_path "${META_DIR}/visa_meta.json" \
  --train_dataset visa \
  --backbone_type "${BACKBONE_TYPE}" \
  --backbone_name "${BACKBONE_NAME}" \
  --epoch 1 \
  --save_freq "${SAVE_FREQ}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --use_residual_adapters \
  --adapter_type mlp \
  --use_refined_mask \
  --no_iou_loss \
  --save_path "${ROOT}/visa"

python test.py \
  --test_data_path "${MVTEC_PATH}" \
  --test_meta_path "${META_DIR}/mvtec_meta.json" \
  --test_dataset mvtec \
  --checkpoint_path "${ROOT}/visa/checkpoints/final_model.pth" \
  --device "${DEVICE}" \
  --fast_metrics_stride 8 \
  --save_path "${ROOT}/visa/results0"

echo "MVTec/VisA residual-adapter refined-mask runs completed."
