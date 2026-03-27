#!/bin/bash
#SBATCH --job-name=openpi_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80        # A100 80 GB — required for full fine-tuning (>70 GB)
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/network/%u/openpi_logs/%x-%j.out
#SBATCH --error=/scratch/network/%u/openpi_logs/%x-%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=is1893@princeton.edu

# ============================================================
# CONFIG — edit these before submitting
# ============================================================
CONFIG_NAME="pi05_libero"          # training config (see src/openpi/training/config.py)
EXP_NAME="my_experiment"           # experiment name (checkpoints saved under this name)
OVERWRITE=true                     # set to false to resume from existing checkpoint
# ============================================================

set -euo pipefail

REPO_DIR="/home/${USER}/vla-stu/openpi"
SCRATCH="/scratch/network/${USER}"
CKPT_DIR="${SCRATCH}/openpi_checkpoints"
LOG_DIR="${SCRATCH}/openpi_logs"

mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

export PATH="${HOME}/.local/bin:${PATH}"

module purge
module load cudatoolkit/12.6      # adjust version: `module avail cudatoolkit`

cd "${REPO_DIR}"

# Sync dependencies (skips git-lfs blobs in LeRobot submodule)
echo "Syncing dependencies with uv..."
GIT_LFS_SKIP_SMUDGE=1 uv sync

# Point checkpoint output to scratch (not home)
export OPENPI_DATA_HOME="${SCRATCH}/.cache/openpi"
mkdir -p "${OPENPI_DATA_HOME}"

# JAX: allow up to 90% of GPU memory
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Step 1: Compute normalization stats (skipped if already done)
NORM_STATS_FILE="${CKPT_DIR}/${CONFIG_NAME}/assets/norm_stats.json"
if [ ! -f "${NORM_STATS_FILE}" ]; then
    echo "Computing normalization stats for ${CONFIG_NAME}..."
    uv run scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
else
    echo "Norm stats already exist, skipping."
fi

# Step 2: Train
OVERWRITE_FLAG=""
if [ "${OVERWRITE}" = true ]; then
    OVERWRITE_FLAG="--overwrite"
fi

echo "Starting training: config=${CONFIG_NAME}, exp=${EXP_NAME}"
uv run scripts/train.py "${CONFIG_NAME}" \
    --exp-name="${EXP_NAME}" \
    --checkpoint-base-dir="${CKPT_DIR}" \
    ${OVERWRITE_FLAG}

echo "Training complete. Checkpoints in: ${CKPT_DIR}/${CONFIG_NAME}/${EXP_NAME}"
