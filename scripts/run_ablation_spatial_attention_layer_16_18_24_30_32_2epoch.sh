#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:1}"
EPOCHS="${EPOCHS:-2}"
SAVE_FREQ="${SAVE_FREQ:-2}"
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
OUT_ROOT="${OUT_ROOT:-ablation/spatial-attention-2epoch}"
GROUP_NAME="layer_16_18_24_30_32"
LAYERS="16 18 24 30 32"

OUT_DIR="${OUT_ROOT}/${GROUP_NAME}"
RESULT_DIR="${OUT_DIR}/results"

echo "Running spatial-attention ablation for ${GROUP_NAME}"
echo "Device: ${DEVICE}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Image size: ${IMAGE_SIZE}"
echo "Backbone: ${BACKBONE_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Data root: ${DATA_ROOT}"
echo "Meta path: ${META_PATH}"
echo "Output: ${OUT_DIR}"
echo "Feature layers: ${LAYERS}"
echo "Residual adapters: disabled"
echo "Refined mask head: disabled"
echo "Analysis visualization: disabled"

mkdir -p "${OUT_ROOT}"

python train.py \
  --train_data_path "${DATA_ROOT}" \
  --train_meta_path "${META_PATH}" \
  --train_dataset "${DATASET_NAME}" \
  --backbone_type dinov3 \
  --backbone_name "${BACKBONE_NAME}" \
  --features_list ${LAYERS} \
  --epoch "${EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --save_freq "${SAVE_FREQ}" \
  --batch_size "${BATCH_SIZE}" \
  --image_size "${IMAGE_SIZE}" \
  --device "${DEVICE}" \
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

python - "${OUT_ROOT}" "${GROUP_NAME}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
group_name = sys.argv[2]
metrics_path = out_root / group_name / "results" / "metrics_summary.json"
summary_path = out_root / f"{group_name}_metrics_summary.csv"
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

row = {"group": group_name}
if metrics_path.exists():
    with metrics_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    mean = payload.get("metrics", {}).get("mean", {})
    for key in fields[1:]:
        value = mean.get(key, "")
        row[key] = f"{value * 100:.4f}" if isinstance(value, (int, float)) else value

with summary_path.open("w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=fields)
    writer.writeheader()
    writer.writerow(row)

print(f"Wrote metrics to {summary_path}")
PY

echo "Completed ${GROUP_NAME}."
