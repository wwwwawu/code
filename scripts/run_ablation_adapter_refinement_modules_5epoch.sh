#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:1}"
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
ADAPTER_REFINED_ROOT="${ADAPTER_REFINED_ROOT:-ablation/resiual-adapters-spatial-attention-refined-mask-5epoch}"
NO_ADAPTER_REFINED_ROOT="${NO_ADAPTER_REFINED_ROOT:-ablation/spatial-attention-refined-mask-5epoch}"

GROUP_NAMES=(
  "adapter_3_6_9_12_refined"
  "no_adapter_refined"
)

OUTPUT_ROOTS=(
  "${ADAPTER_REFINED_ROOT}"
  "${NO_ADAPTER_REFINED_ROOT}"
)

USE_ADAPTERS=(
  "1"
  "0"
)

USE_REFINED_MASKS=(
  "1"
  "1"
)

echo "Running adapter/refined-mask module ablation"
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
echo "Adapter + refined output root: ${ADAPTER_REFINED_ROOT}"
echo "No-adapter + refined output root: ${NO_ADAPTER_REFINED_ROOT}"
echo "Analysis visualization: disabled"

mkdir -p "${ADAPTER_REFINED_ROOT}" "${NO_ADAPTER_REFINED_ROOT}"

for IDX in "${!GROUP_NAMES[@]}"; do
  GROUP_NAME="${GROUP_NAMES[$IDX]}"
  USE_ADAPTER="${USE_ADAPTERS[$IDX]}"
  USE_REFINED="${USE_REFINED_MASKS[$IDX]}"
  OUT_DIR="${OUTPUT_ROOTS[$IDX]}"
  RESULT_DIR="${OUT_DIR}/results"

  echo
  echo "============================================================"
  echo "${GROUP_NAME}: training ${EPOCHS} epoch(s) on ${DEVICE}"
  echo "Feature layers: ${FEATURE_LAYERS}"
  echo "Use residual adapter: ${USE_ADAPTER}"
  echo "Use refined mask head: ${USE_REFINED}"
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
    --no_iou_loss
    --save_path "${OUT_DIR}"
  )

  if [[ "${USE_ADAPTER}" == "1" ]]; then
    TRAIN_CMD+=(
      --use_residual_adapters
      --adapter_type mlp
      --adapter_layers custom
      --adapter_layer_ids "${ADAPTER_LAYERS}"
    )
  fi

  if [[ "${USE_REFINED}" == "1" ]]; then
    TRAIN_CMD+=(--use_refined_mask)
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

python - "${ADAPTER_REFINED_ROOT}" "${NO_ADAPTER_REFINED_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

adapter_root = Path(sys.argv[1])
no_adapter_root = Path(sys.argv[2])
groups = [
    (adapter_root, "adapter_3_6_9_12_refined", "1", "1"),
    (no_adapter_root, "no_adapter_refined", "0", "1"),
]
summary_path = adapter_root.parent / "adapter_refinement_module_metrics_summary.csv"
fields = [
    "group",
    "result_dir",
    "use_residual_adapter",
    "use_refined_mask",
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
for result_dir, group_name, use_adapter, use_refined in groups:
    metrics_path = result_dir / "results" / "metrics_summary.json"
    row = {
        "group": group_name,
        "result_dir": str(result_dir),
        "use_residual_adapter": use_adapter,
        "use_refined_mask": use_refined,
    }
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
echo "Adapter/refined-mask module ablation completed."
echo "Adapter + refined results: ${ADAPTER_REFINED_ROOT}"
echo "No-adapter + refined results: ${NO_ADAPTER_REFINED_ROOT}"
echo "Aggregate summary: ablation/adapter_refinement_module_metrics_summary.csv"
