
from pathlib import Path

import matplotlib.pyplot as plt
import torchvision

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"

train_dataset = torchvision.datasets.CIFAR100(
    root=str(DATA_DIR),
    train=True,
    download=False,
)

test_dataset = torchvision.datasets.CIFAR100(
    root=str(DATA_DIR),
    train=False,
    download=False,
)

class_ids = list(range(100))

train_counts = [train_dataset.targets.count(i) for i in class_ids]
test_counts = [test_dataset.targets.count(i) for i in class_ids]

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].bar(class_ids, train_counts)
axes[0].set_title("CIFAR-100 Training Set: Images per Class")
axes[0].set_ylabel("Number of Images")
axes[0].set_xticks(range(0, 100, 10))
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(class_ids, test_counts)
axes[1].set_title("CIFAR-100 Test Set: Images per Class")
axes[1].set_xlabel("Class ID")
axes[1].set_ylabel("Number of Images")
axes[1].set_xticks(range(0, 100, 10))
axes[1].grid(axis="y", alpha=0.3)

fig.tight_layout()

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
output_path = FIGURES_DIR / "class_distribution.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"Chart saved to: {output_path}")

plt.show()
