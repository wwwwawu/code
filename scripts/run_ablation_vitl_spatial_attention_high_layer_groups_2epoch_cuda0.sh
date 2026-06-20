#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
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
OUT_ROOT="${OUT_ROOT:-ablation-vitl/spatial-attention-2epoch}"

GROUP_NAMES=(
  "layer_22_24"
  "layer_20_22_24"
  "layer_18_20_22_24"
  "layer_16_18_20_22_24"
)

GROUP_LAYERS=(
  "22 24"
  "20 22 24"
  "18 20 22 24"
  "16 18 20 22 24"
)

SUMMARY_NAME="${SUMMARY_NAME:-high_layer_group_metrics_summary.csv}"

echo "Running DINOv3 ViT-L high-layer spatial-attention group ablation"
echo "Device: ${DEVICE}"
echo "Epochs: ${EPOCHS}"
echo "Output root: ${OUT_ROOT}"
echo "Residual adapters: disabled"
echo "Refined mask head: disabled"
echo "Analysis visualization: disabled"

mkdir -p "${OUT_ROOT}"

for IDX in "${!GROUP_NAMES[@]}"; do
  GROUP_NAME="${GROUP_NAMES[$IDX]}"
  FEATURES="${GROUP_LAYERS[$IDX]}"
  OUT_DIR="${OUT_ROOT}/${GROUP_NAME}"
  RESULT_DIR="${OUT_DIR}/results"

  echo
  echo "============================================================"
  echo "${GROUP_NAME}: features_list=${FEATURES}"
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
    --metrics_mode all \
    --fast_metrics_stride "${FAST_METRICS_STRIDE}" \
    --save_path "${RESULT_DIR}"
done

python - "${OUT_ROOT}" "${SUMMARY_NAME}" "${GROUP_NAMES[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
summary_path = out_root / sys.argv[2]
group_names = sys.argv[3:]
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
echo "High-layer group ablation completed."
echo "Aggregate summary: ${OUT_ROOT}/${SUMMARY_NAME}"
