#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-5}"
SAVE_FREQ="${SAVE_FREQ:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
IMAGE_SIZE="${IMAGE_SIZE:-756}"
SIGMA="${SIGMA:-4}"
EVAL_THRESHOLD="${EVAL_THRESHOLD:-0.5}"
FAST_METRICS_STRIDE="${FAST_METRICS_STRIDE:-8}"
METRICS_DECIMALS="${METRICS_DECIMALS:-4}"
NUM_ANCHORS="${NUM_ANCHORS:-4}"

DATA_ROOT="${DATA_ROOT:-../UW-Bench/UW-Bench/training_set}"
META_PATH="${META_PATH:-data_meta/uwbench_meta.json}"
DATASET_NAME="${DATASET_NAME:-uwbench}"
BACKBONE_NAME="${BACKBONE_NAME:-facebook/dinov2-large}"
TORCH_HOME="${TORCH_HOME:-$PWD/model_cache/dinov2}"
SPATIAL_ROOT="${SPATIAL_ROOT:-ablation-vitl/spatial-attention-refined-mask-5epoch/image756}"
ADAPTER_ROOT="${ADAPTER_ROOT:-ablation-vitl/resiual-adapters-spatial-attention-refined-mask/image756}"
FEATURES="${FEATURES:-12 16 20 24}"
FEATURES_TAG="${FEATURES// /_}"
SPATIAL_SUMMARY_NAME="${SPATIAL_SUMMARY_NAME:-spatial_refined_mask_metrics_summary.csv}"
ADAPTER_SUMMARY_NAME="${ADAPTER_SUMMARY_NAME:-refined_mask_adapter_group_metrics_summary.csv}"

GROUP_NAMES=(
  "spatial_${FEATURES_TAG}"
  "adapter_3_6_9_12_spatial_${FEATURES_TAG}"
  "adapter_6_9_12_15_spatial_${FEATURES_TAG}"
  "adapter_9_12_15_18_spatial_${FEATURES_TAG}"
  "adapter_12_16_20_24_spatial_${FEATURES_TAG}"
)

GROUP_LAYERS=(
  ""
  "3,6,9,12"
  "6,9,12,15"
  "9,12,15,18"
  "12,16,20,24"
)

GROUP_ROOTS=(
  "${SPATIAL_ROOT}"
  "${ADAPTER_ROOT}"
  "${ADAPTER_ROOT}"
  "${ADAPTER_ROOT}"
  "${ADAPTER_ROOT}"
)

ADAPTER_GROUP_NAMES=(
  "adapter_3_6_9_12_spatial_${FEATURES_TAG}"
  "adapter_6_9_12_15_spatial_${FEATURES_TAG}"
  "adapter_9_12_15_18_spatial_${FEATURES_TAG}"
  "adapter_12_16_20_24_spatial_${FEATURES_TAG}"
)

echo "Running DINOv2 ViT-L refined-mask adapter ablation"
echo "Prototype token mode: backbone-token ([t_w, t_b, cls, patch tokens])"
echo "Device: ${DEVICE}"
echo "Spatial-attention features: ${FEATURES}"
echo "Spatial-attention anchors: ${NUM_ANCHORS}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Image size: ${IMAGE_SIZE}"
echo "Spatial-only output root: ${SPATIAL_ROOT}"
echo "Adapter output root: ${ADAPTER_ROOT}"
echo "Refined mask head: enabled"
echo "Analysis visualization: disabled"
echo "TORCH_HOME: ${TORCH_HOME}"

mkdir -p "${SPATIAL_ROOT}" "${ADAPTER_ROOT}"

for IDX in "${!GROUP_NAMES[@]}"; do
  GROUP_NAME="${GROUP_NAMES[$IDX]}"
  ADAPTER_IDS="${GROUP_LAYERS[$IDX]}"
  ROOT_DIR="${GROUP_ROOTS[$IDX]}"
  OUT_DIR="${ROOT_DIR}/${GROUP_NAME}"
  RESULT_DIR="${OUT_DIR}/results"

  echo
  echo "============================================================"
  if [[ -z "${ADAPTER_IDS}" ]]; then
    echo "${GROUP_NAME}: spatial-attention + refined-mask, residual adapters disabled"
  else
    echo "${GROUP_NAME}: adapter_layer_ids=${ADAPTER_IDS}"
  fi
  echo "Features: ${FEATURES}"
  echo "Output: ${OUT_DIR}"
  echo "============================================================"

  TRAIN_CMD=(
    python train.py
    --train_data_path "${DATA_ROOT}"
    --train_meta_path "${META_PATH}"
    --train_dataset "${DATASET_NAME}"
    --backbone_type dinov2
    --backbone_name "${BACKBONE_NAME}"
    --features_list ${FEATURES}
    --num_anchors "${NUM_ANCHORS}"
    --epoch "${EPOCHS}"
    --learning_rate "${LEARNING_RATE}"
    --save_freq "${SAVE_FREQ}"
    --batch_size "${BATCH_SIZE}"
    --image_size "${IMAGE_SIZE}"
    --device "${DEVICE}"
    --use_refined_mask
    --no_iou_loss
    --save_path "${OUT_DIR}"
  )

  if [[ -n "${ADAPTER_IDS}" ]]; then
    TRAIN_CMD+=(
      --use_residual_adapters
      --adapter_type mlp
      --adapter_layers custom
      --adapter_layer_ids "${ADAPTER_IDS}"
    )
  fi

  TORCH_HOME="${TORCH_HOME}" "${TRAIN_CMD[@]}"

  echo
  echo "${GROUP_NAME}: testing"

  TORCH_HOME="${TORCH_HOME}" python test.py \
    --test_data_path "${DATA_ROOT}" \
    --test_meta_path "${META_PATH}" \
    --test_dataset "${DATASET_NAME}" \
    --checkpoint_path "${OUT_DIR}/checkpoints/final_model.pth" \
    --device "${DEVICE}" \
    --sigma "${SIGMA}" \
    --eval_threshold "${EVAL_THRESHOLD}" \
    --metrics_mode all \
    --fast_metrics_stride "${FAST_METRICS_STRIDE}" \
    --metrics_decimals "${METRICS_DECIMALS}" \
    --save_path "${RESULT_DIR}"
done

python - "${SPATIAL_ROOT}" "${SPATIAL_SUMMARY_NAME}" "spatial_${FEATURES_TAG}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
summary_path = out_root / sys.argv[2]
groups = sys.argv[3:]
fields = [
    "group",
    "precision",
    "recall",
    "f1",
    "iou",
    "miou",
    "pixel_ap",
    "accuracy",
    "pixel_auroc",
    "pixel_f1",
    "image_auroc",
    "image_ap",
    "image_f1",
]

rows = []
for group in groups:
    metrics_path = out_root / group / "results" / "metrics_summary.json"
    row = {"group": group}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        mean = payload.get("metrics", {}).get("mean", {})
        for key in fields[1:]:
            value = mean.get(key, "")
            row[key] = f"{value * 100:.4f}" if isinstance(value, (int, float)) else value
    rows.append(row)

with summary_path.open("w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote aggregate metrics to {summary_path}")
PY

python - "${ADAPTER_ROOT}" "${ADAPTER_SUMMARY_NAME}" "${ADAPTER_GROUP_NAMES[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
summary_path = out_root / sys.argv[2]
groups = sys.argv[3:]
fields = [
    "group",
    "precision",
    "recall",
    "f1",
    "iou",
    "miou",
    "pixel_ap",
    "accuracy",
    "pixel_auroc",
    "pixel_f1",
    "image_auroc",
    "image_ap",
    "image_f1",
]

rows = []
for group in groups:
    metrics_path = out_root / group / "results" / "metrics_summary.json"
    row = {"group": group}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        mean = payload.get("metrics", {}).get("mean", {})
        for key in fields[1:]:
            value = mean.get(key, "")
            row[key] = f"{value * 100:.4f}" if isinstance(value, (int, float)) else value
    rows.append(row)

with summary_path.open("w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote aggregate metrics to {summary_path}")
PY

echo
echo "Completed DINOv2 ViT-L refined-mask adapter ablation."
echo "Spatial-only summary: ${SPATIAL_ROOT}/${SPATIAL_SUMMARY_NAME}"
echo "Adapter summary: ${ADAPTER_ROOT}/${ADAPTER_SUMMARY_NAME}"
