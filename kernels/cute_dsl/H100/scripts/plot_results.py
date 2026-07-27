"""Generate publication-quality benchmark plots from results.json."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "lines.linewidth": 2.2,
    "figure.dpi": 200,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

with open(os.path.join(RESULTS_DIR, "results.json")) as f:
    data = json.load(f)

results = data["results"]
SHAPES = ["128", "256", "512", "1K", "2K", "4K", "5K", "8K"]
KERNEL_ORDER = ["cublas", "k1", "k2", "k3", "k4", "k5", "k6"]
KERNEL_NAMES = {
    "cublas": "cuBLAS",
    "k1": "K1: Single-CTA TMA+WGMMA",
    "k2": "K2: Multi-stage (3)",
    "k3": "K3: Mainloop+Epilogue",
    "k4": "K4: Warp-Specialized",
    "k5": "K5: WS2 tile (128,256,64)",
    "k6": "K6: TMA Multicast (2,1)",
}

COLORS = {
    "cublas": "#4361ee", "k1": "#f72585", "k2": "#4cc9f0",
    "k3": "#7209b7", "k4": "#f77f00", "k5": "#06d6a0", "k6": "#e63946",
}
MARKERS = {
    "cublas": "s", "k1": "o", "k2": "D", "k3": "^", "k4": "v", "k5": "P", "k6": "*",
}


# ═══════════════════════════════════════════════════════════════════════
# Chart 1: Grouped bar chart (TFLOPS)
# ═══════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(15, 7))
n_shapes = len(SHAPES)
n_kernels = len(KERNEL_ORDER)
bar_width = 0.82 / n_kernels
x = np.arange(n_shapes)

for i, key in enumerate(KERNEL_ORDER):
    vals = [results[key][s] if results[key][s] is not None else 0 for s in SHAPES]
    offset = (i - n_kernels / 2 + 0.5) * bar_width
    bars = ax1.bar(x + offset, vals, bar_width * 0.92, label=KERNEL_NAMES[key],
                   color=COLORS[key], edgecolor="white", linewidth=0.4, zorder=3)
    for bar, val in zip(bars, vals):
        if val > 15:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     f"{val:.0f}", ha="center", va="bottom",
                     fontsize=6.5, fontweight="bold", color=COLORS[key], rotation=90)

ax1.set_xlabel("Matrix Shape  (M = N = K)", fontsize=14, fontweight="bold", labelpad=10)
ax1.set_ylabel("TFLOPS", fontsize=14, fontweight="bold", labelpad=10)
ax1.set_title("H100 FP16 GEMM — CuTe DSL Kernels vs cuBLAS",
              fontsize=18, fontweight="bold", pad=18, color="#0f172a")
ax1.set_xticks(x)
ax1.set_xticklabels(SHAPES, fontsize=13, fontweight="medium")
ax1.tick_params(axis="y", labelsize=12)
ax1.set_ylim(0, 900)
ax1.legend(fontsize=10, loc="upper left", ncol=2, framealpha=0.95, edgecolor="#cbd5e1", fancybox=True)
ax1.grid(axis="y", alpha=0.2, linestyle="--", zorder=0)
ax1.set_axisbelow(True)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
fig1.tight_layout()
fig1.savefig(os.path.join(RESULTS_DIR, "h100_tflops.png"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig1)
print("Saved h100_tflops.png")


# ═══════════════════════════════════════════════════════════════════════
# Chart 2: Line chart (% of cuBLAS)
# ═══════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(14, 7))
cublas_vals = {s: results["cublas"][s] for s in SHAPES}

for key in KERNEL_ORDER:
    if key == "cublas":
        continue
    pcts = []
    for s in SHAPES:
        cv, kv = cublas_vals[s], results[key][s]
        pcts.append(kv / cv * 100 if cv and kv and cv > 0 else None)
    valid_x = [j for j, p in enumerate(pcts) if p is not None]
    valid_y = [p for p in pcts if p is not None]

    ax2.plot(valid_x, valid_y, marker=MARKERS[key], markersize=9, linewidth=2.4,
             label=KERNEL_NAMES[key], color=COLORS[key], markeredgecolor="white",
             markeredgewidth=1, zorder=4)
    for xi, yi in zip(valid_x, valid_y):
        ax2.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=7.5, fontweight="bold",
                     color=COLORS[key])

ax2.axhline(100, color=COLORS["cublas"], linewidth=2.5, linestyle="--",
            alpha=0.6, label="cuBLAS (100%)", zorder=2)
ax2.set_xlabel("Matrix Shape  (M = N = K)", fontsize=14, fontweight="bold", labelpad=10)
ax2.set_ylabel("% of cuBLAS TFLOPS", fontsize=14, fontweight="bold", labelpad=10)
ax2.set_title("CuTe DSL Kernels — % of cuBLAS Performance",
              fontsize=18, fontweight="bold", pad=18, color="#0f172a")
ax2.set_xticks(range(len(SHAPES)))
ax2.set_xticklabels(SHAPES, fontsize=13, fontweight="medium")
ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
ax2.tick_params(axis="y", labelsize=12)
ax2.set_ylim(0, 130)
ax2.legend(fontsize=10, loc="lower left", framealpha=0.95, edgecolor="#cbd5e1", fancybox=True)
ax2.grid(alpha=0.2, linestyle="--", zorder=0)
ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_xlim(-0.5, len(SHAPES) - 0.5)
fig2.tight_layout()
fig2.savefig(os.path.join(RESULTS_DIR, "h100_percent.png"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print("Saved h100_percent.png")


# ═══════════════════════════════════════════════════════════════════════
# Chart 3: Line chart (TFLOPS progression)
# ═══════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(14, 7))
for key in KERNEL_ORDER:
    vals = [results[key][s] if results[key][s] is not None else 0 for s in SHAPES]
    ax3.plot(SHAPES, vals, marker=MARKERS[key], markersize=8, linewidth=2.2,
             label=KERNEL_NAMES[key], color=COLORS[key], markeredgecolor="white",
             markeredgewidth=0.8, zorder=4)

ax3.set_xlabel("Matrix Shape  (M = N = K)", fontsize=14, fontweight="bold", labelpad=10)
ax3.set_ylabel("TFLOPS", fontsize=14, fontweight="bold", labelpad=10)
ax3.set_title("Performance Progression — TFLOPS Scaling with Matrix Size",
              fontsize=18, fontweight="bold", pad=18, color="#0f172a")
ax3.tick_params(axis="both", labelsize=12)
ax3.legend(fontsize=10, loc="upper left", framealpha=0.95, edgecolor="#cbd5e1", fancybox=True)
ax3.grid(alpha=0.2, linestyle="--", zorder=0)
ax3.set_axisbelow(True)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
fig3.tight_layout()
fig3.savefig(os.path.join(RESULTS_DIR, "h100_progression.png"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig3)
print("Saved h100_progression.png")
