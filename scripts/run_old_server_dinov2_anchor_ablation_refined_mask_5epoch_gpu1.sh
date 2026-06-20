#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:1}"
EPOCHS="${EPOCHS:-5}"
SAVE_FREQ="${SAVE_FREQ:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
IMAGE_SIZE="${IMAGE_SIZE:-756}"
SIGMA="${SIGMA:-4}"
EVAL_THRESHOLD="${EVAL_THRESHOLD:-0.5}"
FAST_METRICS_STRIDE="${FAST_METRICS_STRIDE:-8}"
METRICS_DECIMALS="${METRICS_DECIMALS:-4}"

DATA_ROOT="${DATA_ROOT:-../UW-Bench/training_set}"
META_PATH="${META_PATH:-data_meta/uwbench_meta.json}"
DATASET_NAME="${DATASET_NAME:-uwbench}"
BACKBONE_NAME="${BACKBONE_NAME:-facebook/dinov2-large}"
TORCH_HOME="${TORCH_HOME:-$PWD/model_cache/dinov2}"
FEATURES="${FEATURES:-12 16 20 24}"
FEATURES_TAG="${FEATURES// /_}"
ADAPTER_IDS="${ADAPTER_IDS:-3,6,9,12,15,18,21,24}"
ADAPTER_TAG="${ADAPTER_IDS//,/_}"
OUT_ROOT="${OUT_ROOT:-ablation-vitl/anchor-ablation-refined-mask-5epoch/image756}"
SUMMARY_NAME="${SUMMARY_NAME:-anchor_metrics_summary_gpu1.csv}"

ANCHORS=(8 16)
GROUP_NAMES=()

echo "Running old-server DINOv2 ViT-L anchor ablation on ${DEVICE}"
echo "Anchors: ${ANCHORS[*]}"
echo "Adapter layers: ${ADAPTER_IDS}"
echo "Spatial-attention layers: ${FEATURES}"
echo "Epochs: ${EPOCHS} | save_freq=${SAVE_FREQ} | batch_size=${BATCH_SIZE} | image_size=${IMAGE_SIZE}"
echo "Data root: ${DATA_ROOT}"
echo "Output root: ${OUT_ROOT}"
echo "Refined mask head: enabled"
echo "TORCH_HOME: ${TORCH_HOME}"

mkdir -p "${OUT_ROOT}"

for ANCHOR in "${ANCHORS[@]}"; do
  GROUP_NAME="anchor_${ANCHOR}_adapter_${ADAPTER_TAG}_spatial_${FEATURES_TAG}"
  OUT_DIR="${OUT_ROOT}/${GROUP_NAME}"
  RESULT_DIR="${OUT_DIR}/results"
  GROUP_NAMES+=("${GROUP_NAME}")

  echo
  echo "============================================================"
  echo "${GROUP_NAME}"
  echo "============================================================"

  TORCH_HOME="${TORCH_HOME}" python train.py \
    --train_data_path "${DATA_ROOT}" \
    --train_meta_path "${META_PATH}" \
    --train_dataset "${DATASET_NAME}" \
    --backbone_type dinov2 \
    --backbone_name "${BACKBONE_NAME}" \
    --features_list ${FEATURES} \
    --num_anchors "${ANCHOR}" \
    --epoch "${EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --save_freq "${SAVE_FREQ}" \
    --batch_size "${BATCH_SIZE}" \
    --image_size "${IMAGE_SIZE}" \
    --device "${DEVICE}" \
    --use_residual_adapters \
    --adapter_type mlp \
    --adapter_layers custom \
    --adapter_layer_ids "${ADAPTER_IDS}" \
    --use_refined_mask \
    --no_iou_loss \
    --save_path "${OUT_DIR}"

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

python - "${OUT_ROOT}" "${SUMMARY_NAME}" "${GROUP_NAMES[@]}" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
summary_path = out_root / sys.argv[2]
groups = sys.argv[3:]
fields = ["anchor", "group", "precision", "recall", "f1", "iou", "miou", "pixel_ap", "accuracy", "pixel_auroc", "pixel_f1", "image_auroc", "image_ap", "image_f1"]

rows = []
for group in groups:
    metrics_path = out_root / group / "results" / "metrics_summary.json"
    match = re.search(r"anchor_(\d+)_", group)
    row = {"anchor": match.group(1) if match else "", "group": group}
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

echo "Completed anchor ablation on ${DEVICE}."
echo "Aggregate summary: ${OUT_ROOT}/${SUMMARY_NAME}"
