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
NUM_ANCHORS="${NUM_ANCHORS:-4}"

DATA_ROOT="${DATA_ROOT:-../UW-Bench/training_set}"
META_PATH="${META_PATH:-data_meta/uwbench_meta.json}"
DATASET_NAME="${DATASET_NAME:-uwbench}"
BACKBONE_NAME="${BACKBONE_NAME:-facebook/dinov3-vitl16-pretrain-lvd1689m}"
OUT_ROOT="${OUT_ROOT:-ablation-vitl/resiual-adapters-spatial-attention-2epoch}"
FEATURES="22 24"
ADAPTER_LAYERS="${ADAPTER_LAYERS:-13 15 17 19 21 23}"
SUMMARY_NAME="${SUMMARY_NAME:-single_adapter_layer_metrics_gpu1.csv}"

echo "Running DINOv3 ViT-L single residual-adapter layer ablation"
echo "Device: ${DEVICE}"
echo "Spatial-attention features: ${FEATURES}"
echo "Spatial-attention anchors: ${NUM_ANCHORS}"
echo "Adapter layers: ${ADAPTER_LAYERS}"
echo "Epochs: ${EPOCHS}"
echo "Output root: ${OUT_ROOT}"
echo "Refined mask head: disabled"
echo "Analysis visualization: disabled"

mkdir -p "${OUT_ROOT}"

for ADAPTER_LAYER in ${ADAPTER_LAYERS}; do
  OUT_DIR="${OUT_ROOT}/adapter_layer_${ADAPTER_LAYER}"
  RESULT_DIR="${OUT_DIR}/results"

  echo
  echo "============================================================"
  echo "adapter_layer_${ADAPTER_LAYER}: features_list=${FEATURES}, anchors=${NUM_ANCHORS}"
  echo "Training ${EPOCHS} epoch(s)"
  echo "Output: ${OUT_DIR}"
  echo "============================================================"

  python train.py \
    --train_data_path "${DATA_ROOT}" \
    --train_meta_path "${META_PATH}" \
    --train_dataset "${DATASET_NAME}" \
    --backbone_type dinov3 \
    --backbone_name "${BACKBONE_NAME}" \
    --features_list ${FEATURES} \
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
    --adapter_layer_ids "${ADAPTER_LAYER}" \
    --no_iou_loss \
    --save_path "${OUT_DIR}"

  echo
  echo "adapter_layer_${ADAPTER_LAYER}: testing"

  python test.py \
    --test_data_path "${DATA_ROOT}" \
    --test_meta_path "${META_PATH}" \
    --test_dataset "${DATASET_NAME}" \
    --checkpoint_path "${OUT_DIR}/checkpoints/final_model.pth" \
    --device "${DEVICE}" \
    --sigma "${SIGMA}" \
    --eval_threshold "${EVAL_THRESHOLD}" \
    --metrics_mode all \
    --fast_metrics_stride "${FAST_METRICS_STRIDE}" \
    --save_path "${RESULT_DIR}"
done

python - "${OUT_ROOT}" "${ADAPTER_LAYERS}" "${SUMMARY_NAME}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
adapter_layers = [int(layer) for layer in sys.argv[2].split()]
summary_path = out_root / sys.argv[3]
fields = [
    "adapter_layer",
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
for adapter_layer in adapter_layers:
    metrics_path = out_root / f"adapter_layer_{adapter_layer}" / "results" / "metrics_summary.json"
    row = {"adapter_layer": adapter_layer}
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
echo "Single residual-adapter layer ablation part completed."
echo "Aggregate summary: ${OUT_ROOT}/${SUMMARY_NAME}"
