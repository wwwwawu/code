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

DATA_ROOT="${DATA_ROOT:-/media/data/jjh/mydatasets}"
META_PATH="${META_PATH:-/media/data/jjh/mydatasets/mydatasets_meta.json}"
DATASET_NAME="${DATASET_NAME:-mydatasets}"
BACKBONE_NAME="${BACKBONE_NAME:-facebook/dinov3-vith16plus-pretrain-lvd1689m}"
FEATURE_LAYERS="${FEATURE_LAYERS:-18 24 30 32}"
ADAPTER_LAYERS="${ADAPTER_LAYERS:-3,6,9,12}"
ADAPTER_RATIO="${ADAPTER_RATIO:-0.1}"
OUT_ROOT="${OUT_ROOT:-ablation/aaclip-adapters-spatial-attention-5epoch}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/adapter_3_6_9_12}"
RESULT_DIR="${RESULT_DIR:-${OUT_DIR}/results}"

echo "Running AA-CLIP style adapter ablation"
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
echo "Adapter type: aaclip"
echo "Adapter ratio: ${ADAPTER_RATIO}"
echo "Output: ${OUT_DIR}"
echo "Refined mask head: disabled"
echo "Analysis visualization: disabled"

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
  --adapter_layer_ids "${ADAPTER_LAYERS}" \
  --adapter_ratio "${ADAPTER_RATIO}" \
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

python - "${OUT_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
metrics_path = out_dir / "results" / "metrics_summary.json"
summary_path = out_dir / "aaclip_adapter_metrics_summary.csv"
fields = [
    "adapter_type",
    "adapter_layers",
    "adapter_ratio",
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

row = {"adapter_type": "aaclip"}
if metrics_path.exists():
    with metrics_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    row["adapter_layers"] = payload.get("adapter_layer_ids", "")
    row["adapter_ratio"] = payload.get("adapter_ratio", "")
    mean = payload.get("metrics", {}).get("mean", {})
    for key in fields[3:]:
        value = mean.get(key, "")
        row[key] = f"{value * 100:.4f}" if isinstance(value, (int, float)) else value

with summary_path.open("w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=fields)
    writer.writeheader()
    writer.writerow(row)

print(f"Wrote metrics to {summary_path}")
PY
