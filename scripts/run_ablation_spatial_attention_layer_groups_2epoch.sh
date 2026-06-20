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

GROUP_NAMES=(
  "layer_18_24"
  "layer_18_24_30"
  "layer_18_24_30_32"
)

GROUP_LAYERS=(
  "18 24"
  "18 24 30"
  "18 24 30 32"
)

echo "Running multi-layer spatial-attention ablation"
echo "Device: ${DEVICE}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Image size: ${IMAGE_SIZE}"
echo "Backbone: ${BACKBONE_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Data root: ${DATA_ROOT}"
echo "Meta path: ${META_PATH}"
echo "Output root: ${OUT_ROOT}"
echo "Residual adapters: disabled"
echo "Refined mask head: disabled"
echo "Analysis visualization: disabled"

mkdir -p "${OUT_ROOT}"

for IDX in "${!GROUP_NAMES[@]}"; do
  GROUP_NAME="${GROUP_NAMES[$IDX]}"
  LAYERS="${GROUP_LAYERS[$IDX]}"
  OUT_DIR="${OUT_ROOT}/${GROUP_NAME}"
  RESULT_DIR="${OUT_DIR}/results"

  echo
  echo "============================================================"
  echo "${GROUP_NAME}: training ${EPOCHS} epoch(s)"
  echo "Feature layers: ${LAYERS}"
  echo "Output: ${OUT_DIR}"
  echo "============================================================"

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

  echo
  echo "${GROUP_NAME}: testing"

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
summary_path = out_root / "layer_group_metrics_summary.csv"
group_names = ["layer_18_24", "layer_18_24_30", "layer_18_24_30_32"]
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
for group_name in group_names:
    metrics_path = out_root / group_name / "results" / "metrics_summary.json"
    row = {"group": group_name}
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
echo "Multi-layer spatial-attention ablation completed."
echo "Aggregate summary: ${OUT_ROOT}/layer_group_metrics_summary.csv"
