#!/bin/bash
#SBATCH --job-name=stu_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/network/%u/openpi_logs/%x-%j.out
#SBATCH --error=/scratch/network/%u/openpi_logs/%x-%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=sk3686@princeton.edu

# Smoke test for the STU-augmented PI0 model (PyTorch training path).
# Uses debug config with fake data and dummy weights.

set -euo pipefail

REPO_DIR="/home/${USER}/ece534/final/vla-stu/openpi"
SCRATCH="/scratch/network/${USER}"
LOG_DIR="${SCRATCH}/openpi_logs"

mkdir -p "${LOG_DIR}"

CONDA_ENV="${HOME}/.conda/envs/openpi"
export PATH="${HOME}/.local/bin:${CONDA_ENV}/bin:${PATH}"
export PKG_CONFIG_PATH="${CONDA_ENV}/lib/pkgconfig:${CONDA_ENV}/share/pkgconfig"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

module purge
module load cudatoolkit/12.6
export CC=/usr/bin/gcc

cd "${REPO_DIR}"

echo "Syncing dependencies..."
GIT_LFS_SKIP_SMUDGE=1 uv sync --python "${CONDA_ENV}/bin/python"

# Install transformers and apply openpi patches for PyTorch training
uv pip install transformers==4.53.2
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/

export OPENPI_DATA_HOME="${SCRATCH}/.cache/openpi"
mkdir -p "${OPENPI_DATA_HOME}"

echo "Running STU debug training (10 steps, fake data, dummy model, K=4 filters)..."
uv run scripts/train_pytorch.py debug_stu \
    --exp-name=stu_smoketest \
    --checkpoint-base-dir="${SCRATCH}/openpi_checkpoints"

echo "SUCCESS — STU smoke test passed."
