#!/usr/bin/env python3
"""Analyze training results from all VLA experiments.

Parses training logs, extracts loss curves, and generates:
1. H=10 comparison plot (baseline vs STU-K4/K8 vs Mamba-K4/K8)
2. Horizon ablation plot (H=5,10,20,50 for baseline, STU-K4, Mamba-K4)
3. Summary table for the report
"""
import re
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

LOG_DIR = Path("/scratch/gpfs/EHAZAN/sk3686/openpi_logs")
OUTPUT_DIR = Path("/home/sk3686/ece534/vla-stu/latex/final")

# Experiment configs: (log_file_pattern, label)
H10_EXPERIMENTS = {
    "baseline":   ("pi05_libero_baseline_v2.log", "Baseline"),
    "stu_k4":     ("pi05_libero_stu_k4_stu_k4_v2.log", "STU-K4 (post-hoc)"),
    "stu_k8":     ("pi05_libero_stu_k8_stu_k8_v2.log", "STU-K8 (post-hoc)"),
    "mamba_k4":   ("pi05_libero_mamba_k4_mamba_k4_v2.log", "Mamba-K4"),
    "mamba_k8":   ("pi05_libero_mamba_k8_mamba_k8_v2.log", "Mamba-K8"),
    "stu_v2_k4":  ("pi05_libero_stu_v2_k4_stu_v2_k4.log", "STU-v2-K4 (pre-input)"),
    "stu_v2_k8":  ("pi05_libero_stu_v2_k8_stu_v2_k8.log", "STU-v2-K8 (pre-input)"),
}

HORIZON_EXPERIMENTS = {}
STU_V2_LOGS = {
    10: "pi05_libero_stu_v2_k4_stu_v2_k4.log",
    20: "pi05_libero_stu_v2_h20_k4_stu_v2_h20_k4.log",
    50: "pi05_libero_stu_v2_h50_k4_stu_v2_h50_k4.log",
}
for h in [5, 10, 20, 50]:
    if h == 10:
        HORIZON_EXPERIMENTS[f"h{h}_baseline"] = ("pi05_libero_baseline_v2.log", f"H={h} Baseline")
        HORIZON_EXPERIMENTS[f"h{h}_stu"] = ("pi05_libero_stu_k4_stu_k4_v2.log", f"H={h} STU-K4")
        HORIZON_EXPERIMENTS[f"h{h}_mamba"] = ("pi05_libero_mamba_k4_mamba_k4_v2.log", f"H={h} Mamba-K4")
    else:
        HORIZON_EXPERIMENTS[f"h{h}_baseline"] = (f"pi05_libero_h{h}_h{h}_baseline.log", f"H={h} Baseline")
        HORIZON_EXPERIMENTS[f"h{h}_stu"] = (f"pi05_libero_h{h}_stu_k4_h{h}_stu_k4.log", f"H={h} STU-K4")
        HORIZON_EXPERIMENTS[f"h{h}_mamba"] = (f"pi05_libero_h{h}_mamba_k4_h{h}_mamba_k4.log", f"H={h} Mamba-K4")
    if h in STU_V2_LOGS:
        HORIZON_EXPERIMENTS[f"h{h}_stu_v2"] = (STU_V2_LOGS[h], f"H={h} STU-v2-K4")


def parse_log(log_path):
    """Parse training log and extract (step, loss) pairs.

    Logs contain two formats:
    - Coarse: ``step=N loss=X`` written every 100 steps via logger.info
    - Fine: ``loss=X, lr=Y, step=N`` written by tqdm every step but separated
      by \\r instead of \\n.
    Prefer the fine-grained tqdm format when available (per-step resolution).
    """
    if not log_path.exists():
        print(f"  WARNING: Log not found: {log_path}")
        return np.array([]), np.array([])

    with open(log_path, errors='replace') as f:
        text = f.read().replace('\r', '\n')

    # Fine: tqdm postfix format
    fine = re.findall(r'loss=([\d.]+),\s*lr=[\de.+-]+,\s*step=(\d+)', text)
    if fine:
        # tqdm emits two lines per step (mid-step then end-step); dedup by keeping
        # the last loss value seen for each step.
        by_step = {}
        for loss_s, step_s in fine:
            by_step[int(step_s)] = float(loss_s)
        steps = np.array(sorted(by_step.keys()))
        losses = np.array([by_step[s] for s in steps])
        print(f"  Parsed {len(steps)} steps from {log_path.name} (final loss: {losses[-1]:.4f})")
        return steps, losses

    # Coarse fallback
    coarse = re.findall(r'step=(\d+)\s+loss=([\d.]+)', text)
    if coarse:
        steps = np.array([int(s) for s, _ in coarse])
        losses = np.array([float(l) for _, l in coarse])
        print(f"  Parsed {len(steps)} steps (coarse) from {log_path.name} (final loss: {losses[-1]:.4f})")
        return steps, losses

    print(f"  WARNING: No training data in: {log_path}")
    return np.array([]), np.array([])


def smooth(losses, window=50):
    """Rolling average smoothing."""
    if len(losses) < window:
        return losses
    kernel = np.ones(window) / window
    return np.convolve(losses, kernel, mode='valid')


def compute_stats(losses, last_n=100):
    """Compute final loss stats from last N steps."""
    if len(losses) < last_n:
        last_n = len(losses)
    final = losses[-last_n:]
    return {
        'mean': np.mean(final),
        'std': np.std(final),
        'min': np.min(final),
        'max': np.max(final),
    }


def plot_h10_comparison():
    """Plot H=10 model comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8), gridspec_kw={'width_ratios': [2, 1]})

    colors = {
        'Baseline':              '#1f77b4',
        'STU-K4 (post-hoc)':     '#ff7f0e',
        'STU-K8 (post-hoc)':     '#d62728',
        'Mamba-K4':              '#2ca02c',
        'Mamba-K8':              '#9467bd',
        'STU-v2-K4 (pre-input)': '#8c564b',
        'STU-v2-K8 (pre-input)': '#e377c2',
    }

    print("\n=== H=10 Comparison ===")
    stats_table = []

    for key, (logfile, label) in H10_EXPERIMENTS.items():
        steps, losses = parse_log(LOG_DIR / logfile)
        if len(steps) == 0:
            continue

        smoothed = smooth(losses, 50)
        smooth_steps = steps[49:] if len(steps) >= 50 else steps

        color = colors.get(label, '#333333')
        ax1.plot(smooth_steps, smoothed, label=label, color=color, linewidth=1.2, alpha=0.9)

        # Zoomed view
        mask = smooth_steps >= 3000
        if mask.any():
            ax2.plot(smooth_steps[mask], smoothed[mask], label=label, color=color, linewidth=1.2, alpha=0.9)

        stats = compute_stats(losses)
        stats_table.append((label, stats['mean'], stats['std'], len(steps)))

    ax1.set_xlabel('Training Step', fontsize=9)
    ax1.set_ylabel('Training Loss (MSE)', fontsize=9)
    ax1.set_title('H=10: Model Comparison', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.set_xlim(0, 5000)
    ax1.tick_params(labelsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Training Step', fontsize=9)
    ax2.set_title('Zoomed: Steps 3K–5K', fontsize=10, fontweight='bold')
    ax2.set_xlim(3000, 5000)
    ax2.tick_params(labelsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'h10_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_DIR / 'h10_comparison.png', bbox_inches='tight', dpi=150)
    plt.close()

    print("\n  Model         | Final Loss (mean±std) | Steps")
    print("  " + "-" * 50)
    for label, mean, std, n in stats_table:
        print(f"  {label:<14} | {mean:.4f} ± {std:.4f}         | {n}")

    return stats_table


def plot_horizon_ablation():
    """Plot horizon ablation results."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), sharey=True)

    base_colors = {'Baseline': '#1f77b4', 'STU-K4': '#ff7f0e', 'Mamba-K4': '#2ca02c', 'STU-v2-K4': '#8c564b'}
    horizons = [5, 10, 20, 50]

    print("\n=== Horizon Ablation ===")
    all_stats = {}

    for idx, h in enumerate(horizons):
        ax = axes[idx]
        ax.set_title(f'H={h}', fontsize=10, fontweight='bold')

        for model_type, model_label in [('baseline', 'Baseline'), ('stu', 'STU-K4'), ('mamba', 'Mamba-K4'), ('stu_v2', 'STU-v2-K4')]:
            key = f"h{h}_{model_type}"
            if key not in HORIZON_EXPERIMENTS:
                continue
            logfile, label = HORIZON_EXPERIMENTS[key]
            steps, losses = parse_log(LOG_DIR / logfile)
            if len(steps) == 0:
                continue

            smoothed = smooth(losses, 50)
            smooth_steps = steps[49:] if len(steps) >= 50 else steps

            color = base_colors[model_label]
            ax.plot(smooth_steps, smoothed, label=model_label, color=color, linewidth=1.2, alpha=0.9)

            stats = compute_stats(losses)
            all_stats[f"H={h} {model_label}"] = stats['mean']

        ax.set_xlabel('Step', fontsize=8)
        if idx == 0:
            ax.set_ylabel('Loss', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='upper right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'horizon_ablation.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_DIR / 'horizon_ablation.png', bbox_inches='tight', dpi=150)
    plt.close()

    # Print table
    print("\n  Horizon | Baseline | STU-K4  | Mamba-K4 | STU vs Base | Mamba vs Base")
    print("  " + "-" * 70)
    for h in horizons:
        base = all_stats.get(f"H={h} Baseline", float('nan'))
        stu = all_stats.get(f"H={h} STU-K4", float('nan'))
        mamba = all_stats.get(f"H={h} Mamba-K4", float('nan'))
        stu_diff = ((stu - base) / base * 100) if not np.isnan(base) and not np.isnan(stu) else float('nan')
        mamba_diff = ((mamba - base) / base * 100) if not np.isnan(base) and not np.isnan(mamba) else float('nan')
        print(f"  H={h:<3}   | {base:.4f}  | {stu:.4f} | {mamba:.4f}  | {stu_diff:+.1f}%       | {mamba_diff:+.1f}%")

    return all_stats


def plot_horizon_summary():
    """Bar chart showing final loss by horizon for each model type."""
    fig, ax = plt.subplots(figsize=(5, 3))

    horizons = [5, 10, 20, 50]
    models = ['Baseline', 'STU-K4', 'Mamba-K4']
    colors = {'Baseline': '#1f77b4', 'STU-K4': '#ff7f0e', 'Mamba-K4': '#2ca02c'}

    x = np.arange(len(horizons))
    width = 0.25

    for i, model in enumerate(models):
        vals = []
        for h in horizons:
            key = f"h{h}_{'baseline' if model == 'Baseline' else 'stu' if model == 'STU-K4' else 'mamba'}"
            if key not in HORIZON_EXPERIMENTS:
                vals.append(0)
                continue
            logfile, _ = HORIZON_EXPERIMENTS[key]
            steps, losses = parse_log(LOG_DIR / logfile)
            if len(losses) > 0:
                vals.append(np.mean(losses[-100:]))
            else:
                vals.append(0)

        ax.bar(x + (i - 1) * width, vals, width, label=model, color=colors[model], alpha=0.85)

    ax.set_xlabel('Action Horizon (H)')
    ax.set_ylabel('Final Loss (last 100 steps)')
    ax.set_title('Final Loss vs Action Horizon')
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'horizon_bar.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_DIR / 'horizon_bar.png', bbox_inches='tight', dpi=150)
    plt.close()


if __name__ == '__main__':
    print(f"Log directory: {LOG_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # List available logs
    print("\nAvailable log files:")
    for f in sorted(LOG_DIR.glob("pi05_libero*.log")):
        size = f.stat().st_size / 1024
        print(f"  {f.name} ({size:.0f} KB)")

    print("\n" + "=" * 60)
    h10_stats = plot_h10_comparison()

    print("\n" + "=" * 60)
    horizon_stats = plot_horizon_ablation()

    print("\n" + "=" * 60)
    plot_horizon_summary()

    print(f"\nPlots saved to {OUTPUT_DIR}:")
    print("  - h10_comparison.pdf/png")
    print("  - horizon_ablation.pdf/png")
    print("  - horizon_bar.pdf/png")
