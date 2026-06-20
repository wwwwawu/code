#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:1}"
EPOCHS="${EPOCHS:-5}"
SAVE_FREQ="${SAVE_FREQ:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
IMAGE_SIZE="${IMAGE_SIZE:-518}"
SIGMA="${SIGMA:-4}"
EVAL_THRESHOLD="${EVAL_THRESHOLD:-0.5}"
FAST_METRICS_STRIDE="${FAST_METRICS_STRIDE:-8}"

DATA_ROOT="${DATA_ROOT:-/media/data/jjh/mydatasets}"
META_PATH="${META_PATH:-/media/data/jjh/mydatasets/mydatasets_meta.json}"
DATASET_NAME="${DATASET_NAME:-mydatasets}"
BACKBONE_NAME="${BACKBONE_NAME:-facebook/dinov3-vith16plus-pretrain-lvd1689m}"
FEATURE_LAYERS="${FEATURE_LAYERS:-18 24 30 32}"
ADAPTER_RATIO="${ADAPTER_RATIO:-0.1}"
ADAPTER_DROPOUT="${ADAPTER_DROPOUT:-0.0}"
OUT_ROOT="${OUT_ROOT:-ablation/aaclip-adapters-spatial-attention-5epoch}"

GROUP_NAMES=(
  "adapter_1_9"
  "adapter_3_6_9_12"
)

ADAPTER_LAYERS=(
  "1,2,3,4,5,6,7,8,9"
  "3,6,9,12"
)

echo "Running AA-CLIP adapter ablation part 2 on ${DEVICE}"
echo "Output root: ${OUT_ROOT}"

mkdir -p "${OUT_ROOT}"

for IDX in "${!GROUP_NAMES[@]}"; do
  GROUP_NAME="${GROUP_NAMES[$IDX]}"
  ADAPTER_IDS="${ADAPTER_LAYERS[$IDX]}"
  OUT_DIR="${OUT_ROOT}/${GROUP_NAME}"
  RESULT_DIR="${OUT_DIR}/results"

  echo
  echo "============================================================"
  echo "${GROUP_NAME}: training ${EPOCHS} epoch(s) on ${DEVICE}"
  echo "Feature layers: ${FEATURE_LAYERS}"
  echo "Adapter layers: ${ADAPTER_IDS}"
  echo "Output: ${OUT_DIR}"
  echo "============================================================"

  python train.py \
    --train_data_path "${DATA_ROOT}" \
    --train_meta_path "${META_PATH}" \
    --train_dataset "${DATASET_NAME}" \
    --backbone_type dinov3 \
    --backbone_name "${BACKBONE_NAME}" \
    --features_list ${FEATURE_LAYERS} \
    --epoch "${EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --save_freq "${SAVE_FREQ}" \
    --batch_size "${BATCH_SIZE}" \
    --image_size "${IMAGE_SIZE}" \
    --device "${DEVICE}" \
    --use_residual_adapters \
    --adapter_type aaclip \
    --adapter_layers custom \
    --adapter_layer_ids "${ADAPTER_IDS}" \
    --adapter_ratio "${ADAPTER_RATIO}" \
    --adapter_dropout "${ADAPTER_DROPOUT}" \
    --no_iou_loss \
    --save_path "${OUT_DIR}"

  python test.py \
    --test_data_path "${DATA_ROOT}" \
    --test_meta_path "${META_PATH}" \
    --test_dataset "${DATASET_NAME}" \
    --checkpoint_path "${OUT_DIR}/checkpoints/final_model.pth" \
    --device "${DEVICE}" \
    --sigma "${SIGMA}" \
    --eval_threshold "${EVAL_THRESHOLD}" \
    --fast_metrics_stride "${FAST_METRICS_STRIDE}" \
    --save_path "${RESULT_DIR}"
done

echo "AA-CLIP adapter ablation part 2 completed."
