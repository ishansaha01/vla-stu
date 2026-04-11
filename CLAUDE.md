# Project Context for Claude

## What This Is
ECE 534 final project: Integrating Spectral Transfer Units (STUs) into the π₀.5 VLA model's action prediction head. Team: Shivam Kak (sk3686), Ishan Saha (is1893).

## What's Been Implemented
- **STU layer**: `openpi/src/openpi/models_pytorch/stu_layer.py` — PyTorch STU (Hankel eigendecomposition + FFT convolution)
- **STU model**: `openpi/src/openpi/models_pytorch/pi0_stu_pytorch.py` — `PI0STUPytorch` extends `PI0Pytorch`, adds residual STU block between transformer output and action projection
- **Training configs**: In `openpi/src/openpi/training/config.py` — added `stu_num_filters` field to `TrainConfig`, plus configs: `pi05_libero_stu_k4`, `pi05_libero_stu_k8`, `pi05_libero_stu_k16`, `debug_stu`
- **Train script modified**: `openpi/scripts/train_pytorch.py` — auto-selects STU model when `stu_num_filters > 0`, uses `strict=False` for weight loading
- **Preprocessing fix**: `openpi/src/openpi/models_pytorch/preprocessing_pytorch.py` — always outputs NCHW for SigLIP
- **Slurm scripts**: `run_pi05_libero.slurm`, `run_pi05_libero_stu_k4.slurm`, `run_pi05_libero_stu_k8.slurm`, `run_pytorch_train.slurm`
- **Report draft**: `latex/final/final_report.tex` — 2-page report, compiles with pdflatex (no fontawesome5), needs results table filled in

## Current Status
All 3 training jobs (baseline, STU-K4, STU-K8) completed 5000 steps. The HF datasets cache issue was resolved by killing stale NFS processes and cleaning corrupted `.incomplete` directories. Note: `HF_DATASETS_CACHE` env var does NOT redirect where `load_dataset("parquet", data_dir=...)` caches — it uses `$HF_HOME/datasets` regardless. The fix was to let the pre-built cache at the default path be reused.

## Disk Quota
- Scratch quota: 100GB, currently using ~87GB
- Main consumers: HF lerobot cache (33GB), openpi cache (12GB), checkpoints (11GB)
- Each checkpoint save is ~2.5GB (model + optimizer)
- Save interval set to 5000 steps, total steps 5000, so only 1 checkpoint per run

## What's Already Done
- Conda env `openpi` with Python 3.11 + ffmpeg 7 ✓
- `uv` installed, dependencies synced ✓
- π₀.5 base checkpoint downloaded and converted to PyTorch at `/scratch/network/sk3686/openpi_checkpoints/pi05_base_pytorch/` ✓
- LIBERO dataset downloaded at `/scratch/network/sk3686/.cache/huggingface/lerobot/physical-intelligence/libero/` ✓
- Norm stats computed for all configs in `openpi/assets/pi05_libero*/` ✓
- STU layer unit tests pass (K=4,8,16; H=10,50; gradient flow) ✓
- Training confirmed working: baseline reached step 999 (loss ~0.055), STU-K4 reached step 918 (loss ~0.075) before disk quota crash

## What's Done (as of 2026-03-28)
1. **Training completed**: baseline, STU-K4, STU-K8 all ran 5000 steps
2. **Loss curves extracted** and report table filled in
3. **Report compiled**: `latex/final/final_report.pdf` — 2 pages, clean compile
4. **Checkpoints**: STU-K4 saved at step 5000; baseline and STU-K8 checkpoints failed to save (disk quota) but training logs are complete

## Key Paths
- Repo: `/home/sk3686/ece534/final/vla-stu`
- openpi submodule: `/home/sk3686/ece534/final/vla-stu/openpi`
- Scratch: `/scratch/network/sk3686/`
- Logs: `/scratch/network/sk3686/openpi_logs/`
- Checkpoints: `/scratch/network/sk3686/openpi_checkpoints/`
- PyTorch base weights: `/scratch/network/sk3686/openpi_checkpoints/pi05_base_pytorch/`
- HF token: stored at `/scratch/network/sk3686/.cache/huggingface/token`

## Environment Setup (for slurm scripts)
```bash
CONDA_ENV="${HOME}/.conda/envs/openpi"
export PATH="${HOME}/.local/bin:${CONDA_ENV}/bin:${PATH}"
export PKG_CONFIG_PATH="${CONDA_ENV}/lib/pkgconfig:${CONDA_ENV}/share/pkgconfig"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export OPENPI_DATA_HOME="/scratch/network/sk3686/.cache/openpi"
export HF_HOME="/scratch/network/sk3686/.cache/huggingface"
export HF_TOKEN=$(cat /scratch/network/sk3686/.cache/huggingface/token)
module purge && module load cudatoolkit/12.6 && export CC=/usr/bin/gcc
```

## Training Results (5000 steps, 2026-03-28)
- **Baseline**: final avg loss 0.029 (last 100 steps), 1.53 s/step
- **STU-K4**: final avg loss 0.031, 1.48 s/step. Started high (0.141 for first 1K steps), caught up by step 2K
- **STU-K8**: final avg loss 0.031, 1.54 s/step. Started at 0.101 for first 1K, caught up by step 2K
- Both STU variants converge to within 7% of baseline
- K4 and K8 reach identical final loss — short horizon (H=10) likely saturates benefit of more filters
- Log files: `/scratch/network/sk3686/openpi_logs/pi05_pytorch-{3055456,3055457,3055458}.err`

## STU Reference Implementation
From `/home/sk3686/ece534/final/STUZero/ez/agents/models/stu_layer.py` — the `MiniSTU` class adapted into our `STULayer`.

## Note on the Debug Configs
The `dummy` model variant (used in debug configs) does NOT work with PyTorch training — SigLIP outputs 2048-dim but dummy Gemma expects 64-dim. This is a pre-existing openpi bug. Only real configs (gemma_2b) work.
