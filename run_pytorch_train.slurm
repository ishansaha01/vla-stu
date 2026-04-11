#!/bin/bash
#SBATCH --job-name=pi05_pytorch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80        # A100 80 GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/network/%u/openpi_logs/%x-%j.out
#SBATCH --error=/scratch/network/%u/openpi_logs/%x-%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=sk3686@princeton.edu

# ============================================================
# CONFIG — edit these before submitting
# ============================================================
# Available configs:
#   pi05_libero            — baseline (no STU)
#   pi05_libero_stu_k4     — STU with K=4 spectral filters
#   pi05_libero_stu_k8     — STU with K=8 spectral filters
#   pi05_libero_stu_k16    — STU with K=16 spectral filters
CONFIG_NAME="pi05_libero"
EXP_NAME="baseline_v1"
OVERWRITE=true
# ============================================================

set -euo pipefail

REPO_DIR="/home/${USER}/ece534/final/vla-stu/openpi"
SCRATCH="/scratch/network/${USER}"
CKPT_DIR="${SCRATCH}/openpi_checkpoints"
LOG_DIR="${SCRATCH}/openpi_logs"

mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

CONDA_ENV="${HOME}/.conda/envs/openpi"
export PATH="${HOME}/.local/bin:${CONDA_ENV}/bin:${PATH}"
export PKG_CONFIG_PATH="${CONDA_ENV}/lib/pkgconfig:${CONDA_ENV}/share/pkgconfig"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

module purge
module load cudatoolkit/12.6
export CC=/usr/bin/gcc

cd "${REPO_DIR}"

echo "Syncing dependencies with uv..."
GIT_LFS_SKIP_SMUDGE=1 uv sync --python "${CONDA_ENV}/bin/python"

# Install transformers and apply openpi patches for PyTorch training
uv pip install transformers==4.53.2
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/

export OPENPI_DATA_HOME="${SCRATCH}/.cache/openpi"
export HF_HOME="/scratch/network/${USER}/.cache/huggingface"
export HF_TOKEN=$(cat /scratch/network/sk3686/.cache/huggingface/token)
mkdir -p "${OPENPI_DATA_HOME}"

# Compute normalization stats if needed
echo "Computing normalization stats for ${CONFIG_NAME}..."
uv run scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}" || echo "Norm stats computation skipped or already done"

# Train with PyTorch
OVERWRITE_FLAG=""
if [ "${OVERWRITE}" = true ]; then
    OVERWRITE_FLAG="--overwrite"
fi

echo "Starting PyTorch training: config=${CONFIG_NAME}, exp=${EXP_NAME}"
uv run scripts/train_pytorch.py "${CONFIG_NAME}" \
    --exp-name="${EXP_NAME}" \
    --checkpoint-base-dir="${CKPT_DIR}" \
    --no-wandb-enabled \
    ${OVERWRITE_FLAG}

echo "Training complete. Checkpoints in: ${CKPT_DIR}/${CONFIG_NAME}/${EXP_NAME}"
