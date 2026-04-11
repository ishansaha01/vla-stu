#!/usr/bin/env python3
"""Generate training loss curve plot for the final report."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_and_smooth(path, window=50):
    data = np.loadtxt(path, delimiter=',')
    steps, losses = data[:, 0], data[:, 1]
    # Rolling average
    kernel = np.ones(window) / window
    smoothed = np.convolve(losses, kernel, mode='valid')
    smooth_steps = steps[window-1:]
    return steps, losses, smooth_steps, smoothed

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8), gridspec_kw={'width_ratios': [2, 1]})

colors = {'Baseline': '#1f77b4', 'STU-K4': '#ff7f0e', 'STU-K8': '#2ca02c'}
files = [
    ('/tmp/baseline_loss.csv', 'Baseline'),
    ('/tmp/stu_k4_loss.csv', 'STU-K4'),
    ('/tmp/stu_k8_loss.csv', 'STU-K8'),
]

for path, label in files:
    steps, losses, smooth_steps, smoothed = load_and_smooth(path, window=50)
    ax1.plot(smooth_steps, smoothed, label=label, color=colors[label], linewidth=1.2, alpha=0.9)

ax1.set_xlabel('Training Step', fontsize=9)
ax1.set_ylabel('Training Loss (MSE)', fontsize=9)
ax1.set_title('Training Loss Curves', fontsize=10, fontweight='bold')
ax1.legend(fontsize=8, loc='upper right')
ax1.set_xlim(0, 5000)
ax1.set_ylim(0, 0.20)
ax1.tick_params(labelsize=8)
ax1.grid(True, alpha=0.3)

# Zoomed view: last 2000 steps
for path, label in files:
    steps, losses, smooth_steps, smoothed = load_and_smooth(path, window=50)
    mask = smooth_steps >= 3000
    ax2.plot(smooth_steps[mask], smoothed[mask], label=label, color=colors[label], linewidth=1.2, alpha=0.9)

ax2.set_xlabel('Training Step', fontsize=9)
ax2.set_title('Zoomed: Steps 3K\u20135K', fontsize=10, fontweight='bold')
ax2.set_xlim(3000, 5000)
ax2.set_ylim(0.015, 0.045)
ax2.tick_params(labelsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/sk3686/ece534/final/vla-stu/latex/final/loss_curves.pdf', bbox_inches='tight', dpi=300)
plt.savefig('/home/sk3686/ece534/final/vla-stu/latex/final/loss_curves.png', bbox_inches='tight', dpi=150)
print("Saved loss_curves.pdf and loss_curves.png")
