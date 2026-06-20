#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-10}"
SAVE_FREQ="${SAVE_FREQ:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
IMAGE_SIZE="${IMAGE_SIZE:-756}"
SIGMA="${SIGMA:-4}"
EVAL_THRESHOLD="${EVAL_THRESHOLD:-0.5}"
FAST_METRICS_STRIDE="${FAST_METRICS_STRIDE:-8}"
METRICS_DECIMALS="${METRICS_DECIMALS:-4}"

# ── Dataset ──────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-../UW-Bench/UW-Bench/training_set}"
META_PATH="${META_PATH:-data_meta/uwbench_meta.json}"
DATASET_NAME="${DATASET_NAME:-uwbench}"

# ── Backbone ─────────────────────────────────────────────
BACKBONE_TYPE="${BACKBONE_TYPE:-dinov2}"
BACKBONE_NAME="${BACKBONE_NAME:-facebook/dinov2-large}"
TORCH_HOME="${TORCH_HOME:-$PWD/model_cache/dinov2}"

# ── Layers (spatial-aware cross-attention) ───────────────
FEATURES="${FEATURES:-12 16 20 24}"
FEATURES_TAG="${FEATURES// /_}"

# ── Residual adapters ────────────────────────────────────
ADAPTER_LAYER_IDS="${ADAPTER_LAYER_IDS:-3,6,9,12,15,18,21,24}"
ADAPTER_TAG="${ADAPTER_LAYER_IDS//,/_}"

# ── Save root ────────────────────────────────────────────
OUT_ROOT="${OUT_ROOT:-ablation-vitl/resiual-adapters-spatial-attention-refined-mask-10epoch/image756}"

# ── Anchor values to ablate ──────────────────────────────
ANCHORS=(6 4)
GROUP_NAMES=()

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Anchor ablation: 6 vs 4                                    ║"
echo "║  DINOv2 ViT-L | residual adapters | spatial attention       ║"
echo "║  refined mask head | 10 epochs | image_size=756             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Device:          ${DEVICE}"
echo "Epochs:          ${EPOCHS} (save_freq=${SAVE_FREQ})"
echo "Batch size:      ${BATCH_SIZE}"
echo "Learning rate:   ${LEARNING_RATE}"
echo "Image size:      ${IMAGE_SIZE}"
echo "Backbone:        ${BACKBONE_TYPE} / ${BACKBONE_NAME}"
echo "Spatial layers:  ${FEATURES}"
echo "Adapter layers:  ${ADAPTER_LAYER_IDS}"
echo "Refined mask:    enabled"
echo "Anchors:         ${ANCHORS[*]}"
echo "Data root:       ${DATA_ROOT}"
echo "Meta path:       ${META_PATH}"
echo "Dataset:         ${DATASET_NAME}"
echo "Output root:     ${OUT_ROOT}"
echo ""

mkdir -p "${OUT_ROOT}"

for ANCHOR in "${ANCHORS[@]}"; do
  GROUP_NAME="anchor_${ANCHOR}_adapter_${ADAPTER_TAG}_spatial_${FEATURES_TAG}"
  OUT_DIR="${OUT_ROOT}/${GROUP_NAME}"
  RESULT_DIR="${OUT_DIR}/results"
  GROUP_NAMES+=("${GROUP_NAME}")

  echo "════════════════════════════════════════════════════════════════"
  echo "  ▶ anchor = ${ANCHOR}"
  echo "    train dir: ${OUT_DIR}"
  echo "════════════════════════════════════════════════════════════════"
  echo ""

  # ── Training ────────────────────────────────────────────
  TORCH_HOME="${TORCH_HOME}" python train.py \
    --train_data_path "${DATA_ROOT}" \
    --train_meta_path "${META_PATH}" \
    --train_dataset "${DATASET_NAME}" \
    --backbone_type "${BACKBONE_TYPE}" \
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
    --adapter_layer_ids "${ADAPTER_LAYER_IDS}" \
    --use_refined_mask \
    --no_iou_loss \
    --save_path "${OUT_DIR}"

  echo ""
  echo "  ▶ anchor = ${ANCHOR}: testing final checkpoint"
  echo ""

  # ── Testing ─────────────────────────────────────────────
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

  echo ""
done

# ── Aggregate metrics ────────────────────────────────────
python - "${OUT_ROOT}" "anchor_6_vs_4_metrics_summary.csv" "${GROUP_NAMES[@]}" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
summary_path = out_root / sys.argv[2]
groups = sys.argv[3:]
fields = [
    "anchor",
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

print(f"\nAggregate metrics → {summary_path}")
PY

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Anchor ablation completed."
echo "  Results: ${OUT_ROOT}/"
echo "  Summary: ${OUT_ROOT}/anchor_6_vs_4_metrics_summary.csv"
echo "════════════════════════════════════════════════════════════════"
