#!/usr/bin/env python3
"""Extract last-100-step mean/std losses from v4 logs and write a comparison
table to /scratch/gpfs/EHAZAN/sk3686/openpi_logs/stu_v4_results.txt.

Run after the v4 round2 sweep finishes. Includes baselines and earlier-round
runs for context.
"""
from __future__ import annotations

import os
import re
import statistics
from pathlib import Path

LOG_DIR = Path("/scratch/gpfs/EHAZAN/sk3686/openpi_logs")
OUT = LOG_DIR / "stu_v4_results.txt"

# Each entry: (label, log filename, comparison-baseline label).
RUNS = [
    # baselines
    ("baseline H=10 warmup10k", "pi05_libero_baseline_v2.log", None),
    ("baseline H=10 warmup1k",  "pi05_libero_warmup1k_warmup1k.log", None),
    ("baseline H=20",           "pi05_libero_h20_h20_baseline.log", None),
    # v4 runs
    ("STU-v4 K=4 warmup10k",    "pi05_libero_stu_v4_k4_stu_v4_k4.log",                     "baseline H=10 warmup10k"),
    ("STU-v4 K=4 warmup1k",     "pi05_libero_stu_v4_k4_warmup1k_stu_v4_k4_warmup1k.log",   "baseline H=10 warmup1k"),
    ("STU-v4 K=8 warmup10k",    "pi05_libero_stu_v4_k8_stu_v4_k8.log",                     "baseline H=10 warmup10k"),
    ("STU-v4 K=4 H=20",         "pi05_libero_stu_v4_h20_k4_stu_v4_h20_k4.log",             "baseline H=20"),
    ("Mamba-v4 K=4 warmup10k",  "pi05_libero_mamba_v4_k4_mamba_v4_k4.log",                 "baseline H=10 warmup10k"),
    # earlier runs included for context
    ("STU-v2 K=4 warmup10k",    "pi05_libero_stu_v2_k4_stu_v2_k4.log",                     "baseline H=10 warmup10k"),
    ("STU-v3 K=4 warmup10k",    "pi05_libero_stu_v3_k4_stu_v3_k4.log",                     "baseline H=10 warmup10k"),
]

LAST_N = 100
# Match the per-step tqdm progress-bar format: "loss=0.0381, lr=4.5e-05, step=4901"
PATTERN = re.compile(r"loss=([0-9.]+), lr=[^,]+, step=(\d+)")


def last_n_loss(path: Path, n: int = LAST_N) -> tuple[float, float, int] | None:
    if not path.exists():
        return None
    text = path.read_text()
    pairs = [(int(m.group(2)), float(m.group(1))) for m in PATTERN.finditer(text)]
    if not pairs:
        return None
    # Dedup by step (tqdm rewrites the same step many times as the bar updates).
    by_step: dict[int, float] = {}
    for s, l in pairs:
        by_step[s] = l
    steps = sorted(by_step.keys())
    last = [by_step[s] for s in steps[-n:]]
    if len(last) < 2:
        return last[0], 0.0, len(last)
    return statistics.mean(last), statistics.stdev(last), len(last)


def main() -> None:
    results: dict[str, tuple[float, float, int]] = {}
    lines = ["STU-v4 sweep results (last-100-step mean ± std)\n"]
    lines.append(f"{'run':<32} {'mean':>8} {'std':>8} {'n':>5}  vs-baseline")
    lines.append("-" * 80)
    for label, fname, base in RUNS:
        r = last_n_loss(LOG_DIR / fname)
        if r is None:
            lines.append(f"{label:<32}  (log missing or empty)")
            continue
        mean, std, n = r
        results[label] = r
        if base and base in results:
            base_mean = results[base][0]
            delta = (mean - base_mean) / base_mean * 100
            cmp_str = f"{delta:+.1f}% vs {base}"
        else:
            cmp_str = ""
        lines.append(f"{label:<32} {mean:8.4f} {std:8.4f} {n:5d}  {cmp_str}")
    out = "\n".join(lines) + "\n"
    print(out)
    OUT.write_text(out)
    print(f"\n[wrote {OUT}]")


if __name__ == "__main__":
    main()
