#!/bin/bash
#SBATCH --job-name=openpi_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/network/%u/openpi_logs/%x-%j.out
#SBATCH --error=/scratch/network/%u/openpi_logs/%x-%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=is1893@princeton.edu

# Quick smoke test using the built-in "debug" config:
#   - fake data (no downloads)
#   - dummy model weights (no checkpoint download)
#   - 10 train steps
#   - wandb disabled
# Any GPU works — no --constraint needed.

set -euo pipefail

REPO_DIR="/home/${USER}/vla-stu/openpi"
SCRATCH="/scratch/network/${USER}"
LOG_DIR="${SCRATCH}/openpi_logs"

mkdir -p "${LOG_DIR}"

CONDA_ENV="${HOME}/.conda/envs/openpi"
export PATH="${HOME}/.local/bin:${CONDA_ENV}/bin:${PATH}"
export PKG_CONFIG_PATH="${CONDA_ENV}/lib/pkgconfig:${CONDA_ENV}/share/pkgconfig"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

module purge
module load cudatoolkit/12.6      # adjust: run `module avail cudatoolkit` on login node

cd "${REPO_DIR}"

echo "Syncing dependencies..."
GIT_LFS_SKIP_SMUDGE=1 uv sync --python "${CONDA_ENV}/bin/python"

export OPENPI_DATA_HOME="${SCRATCH}/.cache/openpi"
mkdir -p "${OPENPI_DATA_HOME}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo "Running debug training (10 steps, fake data, dummy model)..."
uv run scripts/train.py debug \
    --exp-name=smoketest \
    --checkpoint-base-dir="${SCRATCH}/openpi_checkpoints"

echo "SUCCESS — smoke test passed."
