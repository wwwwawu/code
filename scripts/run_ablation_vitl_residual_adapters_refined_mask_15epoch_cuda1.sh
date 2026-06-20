#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:1}"
EPOCHS="${EPOCHS:-15}"
SAVE_FREQ="${SAVE_FREQ:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
IMAGE_SIZE="${IMAGE_SIZE:-518}"
SIGMA="${SIGMA:-4}"
EVAL_THRESHOLD="${EVAL_THRESHOLD:-0.5}"
FAST_METRICS_STRIDE="${FAST_METRICS_STRIDE:-8}"
METRICS_DECIMALS="${METRICS_DECIMALS:-4}"

DATA_ROOT="${DATA_ROOT:-../UW-Bench/training_set}"
META_PATH="${META_PATH:-data_meta/uwbench_meta.json}"
DATASET_NAME="${DATASET_NAME:-uwbench}"
BACKBONE_NAME="${BACKBONE_NAME:-facebook/dinov3-vitl16-pretrain-lvd1689m}"
OUT_ROOT="${OUT_ROOT:-ablation-vitl/resiual-adapters-spatial-attention-refined-mask-15epoch}"

GROUP_NAMES=(
  "adapter_3_6_9_12_layer_6_12_18_24"
  "adapter_3_5_7_9_11_layer_22_24"
)

FEATURE_LAYERS=(
  "6 12 18 24"
  "22 24"
)

ADAPTER_LAYERS=(
  "3,6,9,12"
  "3,5,7,9,11"
)

ANCHORS=(
  "4"
  "4"
)

SUMMARY_NAME="${SUMMARY_NAME:-vitl_15epoch_residual_refined_metrics_summary.csv}"

echo "Running DINOv3 ViT-L residual-adapter + refined-mask 15epoch ablation"
echo "Device: ${DEVICE}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Backbone: ${BACKBONE_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Output root: ${OUT_ROOT}"
echo "Residual adapters: enabled, type=mlp"
echo "Refined mask head: enabled"
echo "Run order:"
echo "  1. adapter 3,6,9,12 + features 6,12,18,24"
echo "  2. adapter 3,5,7,9,11 + features 22,24"

mkdir -p "${OUT_ROOT}"

for IDX in "${!GROUP_NAMES[@]}"; do
  GROUP_NAME="${GROUP_NAMES[$IDX]}"
  FEATURES="${FEATURE_LAYERS[$IDX]}"
  ADAPTER_IDS="${ADAPTER_LAYERS[$IDX]}"
  NUM_ANCHORS="${ANCHORS[$IDX]}"
  OUT_DIR="${OUT_ROOT}/${GROUP_NAME}"
  RESULT_DIR="${OUT_DIR}/results"

  echo
  echo "============================================================"
  echo "${GROUP_NAME}: training ${EPOCHS} epoch(s) on ${DEVICE}"
  echo "Feature layers: ${FEATURES}"
  echo "Spatial anchors: ${NUM_ANCHORS}"
  echo "Adapter layers: ${ADAPTER_IDS}"
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
    --adapter_layer_ids "${ADAPTER_IDS}" \
    --use_refined_mask \
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
    --metrics_decimals "${METRICS_DECIMALS}" \
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
    "features_list",
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
    config_path = out_root / group_name / "train_config.json"
    row = {"group": group_name}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fp:
            config = json.load(fp)
        row["features_list"] = " ".join(str(v) for v in config.get("features_list", []))
        row["adapter_layers"] = ",".join(str(v) for v in config.get("adapter_layer_ids", []))
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
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
echo "DINOv3 ViT-L 15epoch refined-mask ablation completed."
echo "Aggregate summary: ${OUT_ROOT}/${SUMMARY_NAME}"
