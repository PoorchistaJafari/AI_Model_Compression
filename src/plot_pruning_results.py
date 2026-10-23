
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Results collected from the baseline and pruning experiments
models = ["Baseline", "Pruning 20%", "Pruning 40%"]

accuracy = [34.50, 34.64, 34.36]
macro_f1 = [0.3177, 0.3196, 0.3219]
model_size_mb = [44.959307, 44.959435, 44.959435]

# Create three separate comparison charts
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Validation accuracy
axes[0].bar(models, accuracy)
axes[0].set_title("Validation Accuracy")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_ylim(0, 40)
axes[0].tick_params(axis="x", rotation=15)

for i, value in enumerate(accuracy):
    axes[0].text(i, value + 0.4, f"{value:.2f}%", ha="center")

# 2. Macro-F1
axes[1].bar(models, macro_f1)
axes[1].set_title("Validation Macro-F1")
axes[1].set_ylabel("Macro-F1")
axes[1].set_ylim(0, 0.36)
axes[1].tick_params(axis="x", rotation=15)

for i, value in enumerate(macro_f1):
    axes[1].text(i, value + 0.004, f"{value:.4f}", ha="center")

# 3. Checkpoint size
axes[2].bar(models, model_size_mb)
axes[2].set_title("Checkpoint Size")
axes[2].set_ylabel("Size (MB)")
axes[2].tick_params(axis="x", rotation=15)

for i, value in enumerate(model_size_mb):
    axes[2].text(i, value + 0.15, f"{value:.3f}", ha="center")

fig.suptitle("Baseline vs. Global Magnitude Pruning", fontsize=14)
fig.tight_layout()

output_path = FIGURES_DIR / "pruning_comparison.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Pruning comparison charts saved successfully.")
print(f"Output file: {output_path}")
