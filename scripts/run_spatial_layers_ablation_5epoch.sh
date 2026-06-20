#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-5}"
SAVE_FREQ="${SAVE_FREQ:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
IMAGE_SIZE="${IMAGE_SIZE:-756}"
SIGMA="${SIGMA:-4}"
EVAL_THRESHOLD="${EVAL_THRESHOLD:-0.5}"
FAST_METRICS_STRIDE="${FAST_METRICS_STRIDE:-8}"
NUM_ANCHORS="${NUM_ANCHORS:-4}"

# ── Dataset ──────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-../UW-Bench/UW-Bench/training_set}"
META_PATH="${META_PATH:-data_meta/uwbench_meta.json}"
DATASET_NAME="${DATASET_NAME:-uwbench}"

# ── Backbone ─────────────────────────────────────────────
BACKBONE_TYPE="${BACKBONE_TYPE:-dinov2}"
BACKBONE_NAME="${BACKBONE_NAME:-facebook/dinov2-large}"
TORCH_HOME="${TORCH_HOME:-$PWD/model_cache/dinov2}"

# ── Residual adapters (shared) ───────────────────────────
ADAPTER_LAYER_IDS="${ADAPTER_LAYER_IDS:-3,6,9,12,15,18,21,24}"
ADAPTER_TAG="${ADAPTER_LAYER_IDS//,/_}"

# ── Save root ────────────────────────────────────────────
OUT_ROOT="${OUT_ROOT:-ablation-vitl/resiual-adapters-spatial-attention-refined-mask-5epoch/image756}"

# ── Spatial attention layer configurations ───────────────
SPATIAL_LAYERS=(16 20 24)
GROUP_NAMES=()

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Spatial-attention layer ablation: 16 vs 20 vs 24           ║"
echo "║  DINOv2 ViT-L | residual adapters | refined mask | 5 epoch  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Device:          ${DEVICE}"
echo "Epochs:          ${EPOCHS} (save_freq=${SAVE_FREQ})"
echo "Batch size:      ${BATCH_SIZE}"
echo "Image size:      ${IMAGE_SIZE}"
echo "Backbone:        ${BACKBONE_TYPE} / ${BACKBONE_NAME}"
echo "Adapter layers:  ${ADAPTER_LAYER_IDS}"
echo "Refined mask:    enabled"
echo "Num anchors:     ${NUM_ANCHORS}"
echo "Data root:       ${DATA_ROOT}"
echo "Output root:     ${OUT_ROOT}"
echo ""

mkdir -p "${OUT_ROOT}"

for LAYERS in "${SPATIAL_LAYERS[@]}"; do
  GROUP_NAME="adapter_${ADAPTER_TAG}_spatial_${LAYERS}"
  OUT_DIR="${OUT_ROOT}/${GROUP_NAME}"
  RESULT_DIR="${OUT_DIR}/results"
  GROUP_NAMES+=("${GROUP_NAME}")

  echo "════════════════════════════════════════════════════════════════"
  echo "  ▶ ${GROUP_NAME}"
  echo "    spatial layers: ${LAYERS}"
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
    --features_list ${LAYERS} \
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
    --adapter_layer_ids "${ADAPTER_LAYER_IDS}" \
    --use_refined_mask \
    --no_iou_loss \
    --save_path "${OUT_DIR}"

  echo ""
  echo "  ▶ ${GROUP_NAME}: testing"
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
    --save_path "${RESULT_DIR}"

  echo ""
done

# ── Aggregate metrics ────────────────────────────────────
python - "${OUT_ROOT}" "spatial_layers_metrics_summary.csv" "${GROUP_NAMES[@]}" <<'PY'
import csv, json, re, sys
from pathlib import Path

out_root = Path(sys.argv[1])
summary_path = out_root / sys.argv[2]
groups = sys.argv[3:]
fields = [
    "group", "spatial_layers",
    "precision", "recall", "f1", "iou", "miou", "pixel_ap", "accuracy",
    "pixel_auroc", "pixel_f1", "image_auroc", "image_ap", "image_f1",
]

rows = []
for group in groups:
    metrics_path = out_root / group / "results" / "metrics_summary.json"
    m = re.search(r"spatial_(\d+)", group)
    spatial = m.group(1) if m else ""
    row = {"group": group, "spatial_layers": spatial}
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        mean = payload.get("metrics", {}).get("mean", {})
        for key in fields[2:]:
            value = mean.get(key, "")
            row[key] = f"{value * 100:.4f}" if isinstance(value, (int, float)) else value
    rows.append(row)

summary_path.write_text(
    "\n".join([",".join(fields)]
              + [",".join(row.get(f, "") for f in fields) for row in rows]),
    encoding="utf-8",
)
print(f"\nAggregate metrics → {summary_path}")
PY

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Spatial-attention layer ablation completed."
echo "  Results: ${OUT_ROOT}/"
echo "  Summary: ${OUT_ROOT}/spatial_layers_metrics_summary.csv"
echo "════════════════════════════════════════════════════════════════"
