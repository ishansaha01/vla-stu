# vla-stu
Spectral Filtering for Vision-Language-Action Models

---

## Setup on Princeton Adroit

These instructions are specific to the [Adroit cluster](https://researchcomputing.princeton.edu/systems/adroit). All steps below run on the **login node** unless stated otherwise.

### 1. Clone the repo

```bash
git clone --recurse-submodules https://github.com/ishansaha01/vla-stu
cd vla-stu
```

### 2. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

> Do this on the login node — compute nodes have no internet access.

### 3. Create a conda environment with `ffmpeg 7` and `av`

The `lerobot` dependency requires `av`, which must be built from source against ffmpeg 7.
The system ffmpeg on Adroit is too old (5.1), so we install ffmpeg 7 via conda-forge.

```bash
module load anaconda3/2025.6

conda create -y -n openpi python=3.11
conda install -y -n openpi -c conda-forge "ffmpeg=7.*"
```

### 4. Sync Python dependencies

```bash
CONDA_ENV="$HOME/.conda/envs/openpi"
export PATH="$HOME/.local/bin:$CONDA_ENV/bin:$PATH"
export PKG_CONFIG_PATH="$CONDA_ENV/lib/pkgconfig:$CONDA_ENV/share/pkgconfig"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"

cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --python "$CONDA_ENV/bin/python"
```

> `GIT_LFS_SKIP_SMUDGE=1` is required to skip git-lfs blobs in the LeRobot submodule.
> This only needs to be run once (or after dependency changes).

---

## Running on Adroit

Two Slurm scripts are provided at the repo root:

| Script | Purpose |
|---|---|
| `run_openpi_test.slurm` | Smoke test: 10 steps, fake data, dummy model weights, no wandb (~1 min) |
| `run_openpi.slurm` | Full training run (edit `CONFIG_NAME` and `EXP_NAME` before submitting) |

### Smoke test

```bash
sbatch run_openpi_test.slurm
```

Watch the output:
```bash
squeue --me
tail -f /scratch/network/$USER/openpi_logs/openpi_test-<JOBID>.out
```

Expected output on success:
```
Step 0: grad_norm=..., loss=..., param_norm=...
SUCCESS — smoke test passed.
```

### Full training

Edit the config section at the top of `run_openpi.slurm`:

```bash
CONFIG_NAME="pi05_libero"   # see openpi/src/openpi/training/config.py for all options
EXP_NAME="my_experiment"
OVERWRITE=true
```

Then submit:
```bash
sbatch run_openpi.slurm
```

Checkpoints are saved to `/scratch/network/$USER/openpi_checkpoints/<CONFIG_NAME>/<EXP_NAME>/`.

### GPU memory requirements

| Mode | Min VRAM | Slurm constraint |
|---|---|---|
| Inference | 8 GB | any GPU |
| Fine-tuning (LoRA) | 22.5 GB | `--constraint=a100` |
| Fine-tuning (full) | 70 GB | `--constraint=gpu80` |

---

## Notes

- Checkpoints and model cache are written to `/scratch/network/$USER/` — not `/home`, which has a small quota.
- The `debug` and `debug_pi05` configs use fake data and dummy weights, useful for testing without downloading anything.
- For available training configs, see `openpi/src/openpi/training/config.py`.
- openpi documentation: [openpi/README.md](openpi/README.md)
