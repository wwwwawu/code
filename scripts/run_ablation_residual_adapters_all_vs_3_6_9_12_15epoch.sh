#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-15}"
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
OUT_ROOT="${OUT_ROOT:-ablation/resiual-adapters-spatial-attention-15epoch}"

GROUP_NAMES=(
  "adapter_all"
  "adapter_3_6_9_12"
)

ADAPTER_LAYER_MODES=(
  "all"
  "custom"
)

ADAPTER_LAYERS=(
  ""
  "3,6,9,12"
)

echo "Running residual-adapter 15epoch ablation"
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
echo "Output root: ${OUT_ROOT}"
echo "Residual adapters: enabled, type=mlp"
echo "Run order: adapter_all -> adapter_3_6_9_12"
echo "Refined mask head: disabled"
echo "Analysis visualization: disabled"

mkdir -p "${OUT_ROOT}"

for IDX in "${!GROUP_NAMES[@]}"; do
  GROUP_NAME="${GROUP_NAMES[$IDX]}"
  ADAPTER_MODE="${ADAPTER_LAYER_MODES[$IDX]}"
  ADAPTER_IDS="${ADAPTER_LAYERS[$IDX]}"
  OUT_DIR="${OUT_ROOT}/${GROUP_NAME}"
  RESULT_DIR="${OUT_DIR}/results"

  echo
  echo "============================================================"
  echo "${GROUP_NAME}: training ${EPOCHS} epoch(s) on ${DEVICE}"
  echo "Feature layers: ${FEATURE_LAYERS}"
  echo "Adapter mode: ${ADAPTER_MODE}"
  if [[ -n "${ADAPTER_IDS}" ]]; then
    echo "Adapter layers: ${ADAPTER_IDS}"
  else
    echo "Adapter layers: all backbone layers"
  fi
  echo "Output: ${OUT_DIR}"
  echo "============================================================"

  TRAIN_CMD=(
    python train.py
    --train_data_path "${DATA_ROOT}"
    --train_meta_path "${META_PATH}"
    --train_dataset "${DATASET_NAME}"
    --backbone_type dinov3
    --backbone_name "${BACKBONE_NAME}"
    --features_list ${FEATURE_LAYERS}
    --epoch "${EPOCHS}"
    --learning_rate "${LEARNING_RATE}"
    --save_freq "${SAVE_FREQ}"
    --batch_size "${BATCH_SIZE}"
    --image_size "${IMAGE_SIZE}"
    --device "${DEVICE}"
    --use_residual_adapters
    --adapter_type mlp
    --adapter_layers "${ADAPTER_MODE}"
    --no_iou_loss
    --save_path "${OUT_DIR}"
  )

  if [[ -n "${ADAPTER_IDS}" ]]; then
    TRAIN_CMD+=(--adapter_layer_ids "${ADAPTER_IDS}")
  fi

  "${TRAIN_CMD[@]}"

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
group_names = ["adapter_all", "adapter_3_6_9_12"]
summary_path = out_root / "residual_adapter_all_vs_3_6_9_12_metrics_summary.csv"
fields = [
    "group",
    "adapter_type",
    "adapter_layers",
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
    row = {"group": group_name, "adapter_type": "mlp"}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        row["adapter_layers"] = payload.get("adapter_layer_ids", "")
        mean = payload.get("metrics", {}).get("mean", {})
        for key in fields[3:]:
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
echo "Residual-adapter 15epoch ablation completed."
echo "Aggregate summary: ${OUT_ROOT}/residual_adapter_all_vs_3_6_9_12_metrics_summary.csv"
