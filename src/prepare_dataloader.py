
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

mean = (0.5071, 0.4866, 0.4409)
std = (0.2673, 0.2564, 0.2762)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Training dataset
train_dataset = torchvision.datasets.CIFAR100(
    root=str(DATA_DIR),
    train=True,
    download=False,
    transform=transform,
)

# Separate dataset object for validation.
# It uses the same images but keeps preprocessing deterministic.
validation_dataset = torchvision.datasets.CIFAR100(
    root=str(DATA_DIR),
    train=True,
    download=False,
    transform=transform,
)

# Test dataset, reserved for final evaluation
test_dataset = torchvision.datasets.CIFAR100(
    root=str(DATA_DIR),
    train=False,
    download=False,
    transform=transform,
)

# These loaders remain available for inspection and later evaluation.
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
)

if __name__ == "__main__":
    images, labels = next(iter(train_loader))

    print("DataLoader preparation successful")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation dataset images: {len(validation_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")
    print(
        f"Image value range: {images.min().item():.2f} "
        f"to {images.max().item():.2f}"
    )
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
