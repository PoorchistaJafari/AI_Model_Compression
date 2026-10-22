
from pathlib import Path

import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

# 1. Find the project folder and create the data folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 2. Convert images into PyTorch tensors
transform = transforms.ToTensor()

# 3. Download and load the training dataset
print("Loading CIFAR-100 training dataset...")

train_dataset = torchvision.datasets.CIFAR100(
    root=str(DATA_DIR),
    train=True,
    download=True,
    transform=transform,
)

# 4. Download and load the test dataset
print("Loading CIFAR-100 test dataset...")

test_dataset = torchvision.datasets.CIFAR100(
    root=str(DATA_DIR),
    train=False,
    download=True,
    transform=transform,
)

# 5. Print basic dataset information
print("\n--- Dataset Summary ---")
print(f"Training images: {len(train_dataset)}")
print(f"Test images: {len(test_dataset)}")
print(f"Number of classes: {len(train_dataset.classes)}")
print(f"Image tensor shape: {train_dataset[0][0].shape}")

first_image, first_label = train_dataset[0]
print(f"First image label: {train_dataset.classes[first_label]}")

# 6. Display ten example images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    image, label = train_dataset[i]
    image = image.permute(1, 2, 0).numpy()

    ax.imshow(image)
    ax.set_title(train_dataset.classes[label])
    ax.axis("off")

plt.tight_layout()

# 7. Save the visualization
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

figure_path = FIGURES_DIR / "cifar100_samples.png"
plt.savefig(figure_path, dpi=150, bbox_inches="tight")

print(f"\nSample visualization saved to: {figure_path}")

plt.show()
