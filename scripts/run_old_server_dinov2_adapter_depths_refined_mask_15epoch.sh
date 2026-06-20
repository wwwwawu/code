#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:1}"
EPOCHS="${EPOCHS:-15}"
SAVE_FREQ="${SAVE_FREQ:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
IMAGE_SIZE="${IMAGE_SIZE:-756}"
SIGMA="${SIGMA:-4}"
EVAL_THRESHOLD="${EVAL_THRESHOLD:-0.5}"
FAST_METRICS_STRIDE="${FAST_METRICS_STRIDE:-8}"
METRICS_DECIMALS="${METRICS_DECIMALS:-4}"
NUM_ANCHORS="${NUM_ANCHORS:-4}"
SEED="${SEED:-111}"
RUN_TAG="${RUN_TAG:-}"

DATA_ROOT="${DATA_ROOT:-../UW-Bench/training_set}"
META_PATH="${META_PATH:-data_meta/uwbench_meta.json}"
DATASET_NAME="${DATASET_NAME:-uwbench}"
BACKBONE_NAME="${BACKBONE_NAME:-facebook/dinov2-large}"
TORCH_HOME="${TORCH_HOME:-$PWD/model_cache/dinov2}"
FEATURES="${FEATURES:-12 16 20 24}"
FEATURES_TAG="${FEATURES// /_}"
OUT_ROOT="${OUT_ROOT:-ablation-vitl/resiual-adapters-spatial-attention-refined-mask-15epoch/image756}"
SUMMARY_NAME="${SUMMARY_NAME:-adapter_depth_epoch_metrics_summary.csv}"
if [[ -n "${RUN_TAG}" ]]; then
  OUT_ROOT="${OUT_ROOT}/${RUN_TAG}"
fi

GROUP_NAMES=(
  "adapter_3_6_9_12_spatial_${FEATURES_TAG}"
  "adapter_3_6_9_12_15_18_spatial_${FEATURES_TAG}"
  "adapter_3_6_9_12_15_18_21_24_spatial_${FEATURES_TAG}"
)

GROUP_LAYERS=(
  "3,6,9,12"
  "3,6,9,12,15,18"
  "3,6,9,12,15,18,21,24"
)

TEST_EPOCHS=(5 10 15)
GROUP_INDICES="${GROUP_INDICES:-0 1 2}"

echo "Running old-server DINOv2 ViT-L adapter-depth experiments"
echo "Prototype token mode: backbone-token ([t_w, t_b, cls, patch tokens])"
echo "Device: ${DEVICE}"
echo "Spatial-attention features: ${FEATURES}"
echo "Spatial-attention anchors: ${NUM_ANCHORS}"
echo "Training epochs: ${EPOCHS}"
echo "Test checkpoints: ${TEST_EPOCHS[*]}"
echo "Selected group indices: ${GROUP_INDICES}"
echo "Run tag: ${RUN_TAG:-none}"
echo "Random seed: ${SEED}"
echo "Batch size: ${BATCH_SIZE}"
echo "Image size: ${IMAGE_SIZE}"
echo "Output root: ${OUT_ROOT}"
echo "Refined mask head: enabled"
echo "Analysis visualization: disabled"
echo "TORCH_HOME: ${TORCH_HOME}"

mkdir -p "${OUT_ROOT}"

for IDX in ${GROUP_INDICES}; do
  if (( IDX < 0 || IDX >= ${#GROUP_NAMES[@]} )); then
    echo "Invalid group index: ${IDX}. Valid indices: 0 1 2" >&2
    exit 1
  fi
  GROUP_NAME="${GROUP_NAMES[$IDX]}"
  ADAPTER_IDS="${GROUP_LAYERS[$IDX]}"
  OUT_DIR="${OUT_ROOT}/${GROUP_NAME}"

  echo
  echo "============================================================"
  echo "${GROUP_NAME}"
  echo "Adapter layers: ${ADAPTER_IDS}"
  echo "Spatial-attention layers: ${FEATURES}"
  echo "Output: ${OUT_DIR}"
  echo "============================================================"

  TORCH_HOME="${TORCH_HOME}" python train.py \
    --train_data_path "${DATA_ROOT}" \
    --train_meta_path "${META_PATH}" \
    --train_dataset "${DATASET_NAME}" \
    --backbone_type dinov2 \
    --backbone_name "${BACKBONE_NAME}" \
    --features_list ${FEATURES} \
    --num_anchors "${NUM_ANCHORS}" \
    --epoch "${EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --save_freq "${SAVE_FREQ}" \
    --batch_size "${BATCH_SIZE}" \
    --image_size "${IMAGE_SIZE}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --use_residual_adapters \
    --adapter_type mlp \
    --adapter_layers custom \
    --adapter_layer_ids "${ADAPTER_IDS}" \
    --use_refined_mask \
    --no_iou_loss \
    --save_path "${OUT_DIR}"

  for TEST_EPOCH in "${TEST_EPOCHS[@]}"; do
    CHECKPOINT_PATH="${OUT_DIR}/checkpoints/epoch_${TEST_EPOCH}.pth"
    RESULT_DIR="${OUT_DIR}/results_epoch${TEST_EPOCH}"

    echo
    echo "${GROUP_NAME}: testing epoch ${TEST_EPOCH}"

    TORCH_HOME="${TORCH_HOME}" python test.py \
      --test_data_path "${DATA_ROOT}" \
      --test_meta_path "${META_PATH}" \
      --test_dataset "${DATASET_NAME}" \
      --checkpoint_path "${CHECKPOINT_PATH}" \
      --device "${DEVICE}" \
      --sigma "${SIGMA}" \
      --eval_threshold "${EVAL_THRESHOLD}" \
      --metrics_mode all \
      --fast_metrics_stride "${FAST_METRICS_STRIDE}" \
      --metrics_decimals "${METRICS_DECIMALS}" \
      --save_path "${RESULT_DIR}"
  done
done

SELECTED_GROUP_NAMES=()
for IDX in ${GROUP_INDICES}; do
  SELECTED_GROUP_NAMES+=("${GROUP_NAMES[$IDX]}")
done

DEVICE_TAG="${DEVICE//:/_}"
SUMMARY_PATH="${SUMMARY_NAME%.csv}_${DEVICE_TAG}.csv"

python - "${OUT_ROOT}" "${SUMMARY_PATH}" "${SELECTED_GROUP_NAMES[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
summary_path = out_root / sys.argv[2]
groups = sys.argv[3:]
test_epochs = (5, 10, 15)
fields = [
    "group",
    "epoch",
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
    for epoch in test_epochs:
        metrics_path = out_root / group / f"results_epoch{epoch}" / "metrics_summary.json"
        row = {"group": group, "epoch": epoch}
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            mean = payload.get("metrics", {}).get("mean", {})
            for key in fields[2:]:
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
echo "Completed old-server DINOv2 ViT-L adapter-depth experiments."
echo "Aggregate summary: ${OUT_ROOT}/${SUMMARY_PATH}"
