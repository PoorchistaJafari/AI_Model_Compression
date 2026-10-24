
from pathlib import Path

import matplotlib.pyplot as plt


# Resolve project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Validation results from the same 5,000-image validation subset
models = [
    "Extended Baseline\nFP32",
    "Pruning 20%",
    "Pruning 40%",
    "Static INT8",
    "Pruning 20% +\nStatic INT8",
]

accuracy = [39.00, 38.84, 38.24, 38.88, 38.80]
macro_f1 = [0.3767, 0.3764, 0.3743, 0.3762, 0.3768]

# Serialized checkpoint/state-dictionary sizes in decimal MB
size_mb = [
    44.960587,
    44.960715,
    44.960715,
    11.355075,
    11.356617,
]

# Median throughput from the recorded benchmark session.
# Throughput results are available for these three models only.
speed_models = [
    "Extended Baseline\nFP32",
    "Static INT8",
    "Pruning 20% +\nStatic INT8",
]

throughput_batch1 = [57.37, 88.30, 86.55]
throughput_batch32 = [102.02, 437.81, 436.19]

# Create four comparison charts
fig, axes = plt.subplots(2, 2, figsize=(17, 12))

# 1. Validation accuracy
axes[0, 0].bar(models, accuracy)
axes[0, 0].set_title("Validation Accuracy")
axes[0, 0].set_ylabel("Accuracy (%)")
axes[0, 0].set_ylim(0, 45)
axes[0, 0].tick_params(axis="x", rotation=30, labelsize=8)

for i, value in enumerate(accuracy):
    axes[0, 0].text(
        i,
        value + 0.4,
        f"{value:.2f}%",
        ha="center",
        fontsize=8,
    )

# 2. Validation Macro-F1
axes[0, 1].bar(models, macro_f1)
axes[0, 1].set_title("Validation Macro-F1")
axes[0, 1].set_ylabel("Macro-F1")
axes[0, 1].set_ylim(0, 0.42)
axes[0, 1].tick_params(axis="x", rotation=30, labelsize=8)

for i, value in enumerate(macro_f1):
    axes[0, 1].text(
        i,
        value + 0.005,
        f"{value:.4f}",
        ha="center",
        fontsize=8,
    )

# 3. Serialized checkpoint/state-dictionary size
axes[1, 0].bar(models, size_mb)
axes[1, 0].set_title("Serialized File Size")
axes[1, 0].set_ylabel("Size (decimal MB)")
axes[1, 0].tick_params(axis="x", rotation=30, labelsize=8)

for i, value in enumerate(size_mb):
    axes[1, 0].text(
        i,
        value + 0.6,
        f"{value:.2f} MB",
        ha="center",
        fontsize=8,
    )

# 4. CPU inference throughput
x_positions = list(range(len(speed_models)))
bar_width = 0.36

axes[1, 1].bar(
    [x - bar_width / 2 for x in x_positions],
    throughput_batch1,
    width=bar_width,
    label="Batch size 1",
)

axes[1, 1].bar(
    [x + bar_width / 2 for x in x_positions],
    throughput_batch32,
    width=bar_width,
    label="Batch size 32",
)

axes[1, 1].set_title("CPU Inference Throughput")
axes[1, 1].set_ylabel("Images per second")
axes[1, 1].set_xticks(x_positions)
axes[1, 1].set_xticklabels(
    speed_models,
    rotation=20,
    ha="right",
    fontsize=8,
)
axes[1, 1].legend()

fig.suptitle(
    "Model Compression Comparison - Extended Baseline",
    fontsize=15,
)

fig.tight_layout(rect=[0, 0, 1, 0.96])

output_path = FIGURES_DIR / "compression_comparison.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Compression comparison charts saved successfully.")
print(f"Output file: {output_path}")