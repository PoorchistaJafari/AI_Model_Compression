
from pathlib import Path

import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Results collected from completed experiments
models = [
    "Baseline FP32",
    "Pruning 20%",
    "Pruning 40%",
    "Dynamic INT8",
    "Static INT8",
    "Pruning 20% +\nStatic INT8",
]

accuracy = [34.50, 34.64, 34.36, 34.48, 34.36, 34.48]
macro_f1 = [0.3177, 0.3196, 0.3219, 0.3175, 0.3158, 0.3187]

# Serialized checkpoint/state-dict sizes in decimal MB
size_mb = [44.959307, 44.959435, 44.959435, 44.807075, 11.353661, 11.355075]

# Create three comparison charts
fig, axes = plt.subplots(1, 3, figsize=(19, 6))

# 1. Validation accuracy
axes[0].bar(models, accuracy)
axes[0].set_title("Validation Accuracy")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_ylim(0, 40)
axes[0].tick_params(axis="x", rotation=65, labelsize=8)

for i, value in enumerate(accuracy):
    axes[0].text(i, value + 0.35, f"{value:.2f}%", ha="center", fontsize=8)

# 2. Validation Macro-F1
axes[1].bar(models, macro_f1)
axes[1].set_title("Validation Macro-F1")
axes[1].set_ylabel("Macro-F1")
axes[1].set_ylim(0, 0.36)
axes[1].tick_params(axis="x", rotation=65, labelsize=8)

for i, value in enumerate(macro_f1):
    axes[1].text(i, value + 0.003, f"{value:.4f}", ha="center", fontsize=8)

# 3. Serialized checkpoint/state-dict size
axes[2].bar(models, size_mb)
axes[2].set_title("Serialized File Size")
axes[2].set_ylabel("Size (MB)")
axes[2].tick_params(axis="x", rotation=65, labelsize=8)

for i, value in enumerate(size_mb):
    axes[2].text(i, value + 0.35, f"{value:.2f}", ha="center", fontsize=8)

fig.suptitle(
    "Comparison of Model Compression Experiments",
    fontsize=15,
)
fig.tight_layout()

output_path = FIGURES_DIR / "compression_comparison.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Compression comparison charts saved successfully.")
print(f"Output file: {output_path}")
