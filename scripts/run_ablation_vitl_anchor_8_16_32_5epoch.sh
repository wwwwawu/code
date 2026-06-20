#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-5}"
SAVE_FREQ="${SAVE_FREQ:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
IMAGE_SIZE="${IMAGE_SIZE:-518}"
SIGMA="${SIGMA:-4}"
EVAL_THRESHOLD="${EVAL_THRESHOLD:-0.5}"
FAST_METRICS_STRIDE="${FAST_METRICS_STRIDE:-8}"

DATA_ROOT="${DATA_ROOT:-../UW-Bench/training_set}"
META_PATH="${META_PATH:-data_meta/uwbench_meta.json}"
DATASET_NAME="${DATASET_NAME:-uwbench}"
BACKBONE_NAME="${BACKBONE_NAME:-facebook/dinov3-vitl16-pretrain-lvd1689m}"
FEATURE_LAYERS="${FEATURE_LAYERS:-6 12 18 24}"
ADAPTER_LAYERS="${ADAPTER_LAYERS:-3,6,9,12}"
OUT_ROOT="${OUT_ROOT:-ablation-vitl/all-5epoch}"

ANCHORS=(
  "8"
  "16"
  "32"
)

echo "Running ViT-L anchor ablation"
echo "Device: ${DEVICE}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Image size: ${IMAGE_SIZE}"
echo "Backbone: ${BACKBONE_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Data root: ${DATA_ROOT}"
echo "Meta path: ${META_PATH}"
echo "Feature layers: ${FEATURE_LAYERS}"
echo "Adapter layers: ${ADAPTER_LAYERS}"
echo "Output root: ${OUT_ROOT}"
echo "Analysis visualization: disabled"

mkdir -p "${OUT_ROOT}"

for NUM_ANCHORS in "${ANCHORS[@]}"; do
  OUT_DIR="${OUT_ROOT}/${NUM_ANCHORS}anchor"
  RESULT_DIR="${OUT_DIR}/results"

  echo
  echo "============================================================"
  echo "${NUM_ANCHORS} anchors: training ${EPOCHS} epoch(s) on ${DEVICE}"
  echo "Output: ${OUT_DIR}"
  echo "============================================================"

  python train.py \
    --train_data_path "${DATA_ROOT}" \
    --train_meta_path "${META_PATH}" \
    --train_dataset "${DATASET_NAME}" \
    --backbone_type dinov3 \
    --backbone_name "${BACKBONE_NAME}" \
    --features_list ${FEATURE_LAYERS} \
    --num_anchors "${NUM_ANCHORS}" \
    --epoch "${EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --save_freq "${SAVE_FREQ}" \
    --batch_size "${BATCH_SIZE}" \
    --image_size "${IMAGE_SIZE}" \
    --device "${DEVICE}" \
    --use_residual_adapters \
    --adapter_type mlp \
    --adapter_layers custom \
    --adapter_layer_ids "${ADAPTER_LAYERS}" \
    --use_refined_mask \
    --no_iou_loss \
    --save_path "${OUT_DIR}"

  echo
  echo "${NUM_ANCHORS} anchors: testing"

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

python - "${OUT_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
anchors = ["8", "16", "32"]
summary_path = out_root / "anchor_8_16_32_metrics_summary.csv"
fields = [
    "num_anchors",
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
for num_anchors in anchors:
    metrics_path = out_root / f"{num_anchors}anchor" / "results" / "metrics_summary.json"
    row = {"num_anchors": num_anchors}
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
echo "ViT-L anchor ablation completed."
echo "Aggregate summary: ${OUT_ROOT}/anchor_8_16_32_metrics_summary.csv"
