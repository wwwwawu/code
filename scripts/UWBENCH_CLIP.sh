#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="${UWBENCH_PATH:-"${REPO_DIR}/../UW-Bench/training_set"}"
META_PATH="${UWBENCH_META_PATH:-"${REPO_DIR}/data_meta/uwbench_meta.json"}"

GPU="${GPU:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-5}"
IMAGE_SIZE="${IMAGE_SIZE:-336}"
BACKBONE="${BACKBONE:-ViT-L/14@336px}"
SIGMAS="${SIGMAS:-0 1 2 4}"

EXP_DIR="${EXP_DIR:-"${REPO_DIR}/experiments/uwbench_clip_l14"}"
CKPT_DIR="${EXP_DIR}/checkpoints"
RESULT_DIR="${EXP_DIR}/results"

cd "${REPO_DIR}"

python generate_dataset_json/uwbench.py

python train.py \
    --train_data_path "${DATA_PATH}" \
    --train_meta_path "${META_PATH}" \
    --save_path "${CKPT_DIR}" \
    --train_dataset uwbench \
    --backbone "${BACKBONE}" \
    --features_list 6 12 18 24 \
    --epoch "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --image_size "${IMAGE_SIZE}" \
    --device "${GPU}"

for sigma in ${SIGMAS}; do
    python test.py \
        --test_data_path "${DATA_PATH}" \
        --test_meta_path "${META_PATH}" \
        --checkpoint_path "${CKPT_DIR}/epoch_${EPOCHS}.pth" \
        --test_dataset uwbench \
        --save_path "${RESULT_DIR}/sigma_${sigma}" \
        --sigma "${sigma}" \
        --device "${GPU}"
done
