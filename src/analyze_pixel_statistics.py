
from pathlib import Path

import torch
import torchvision
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

transform = transforms.ToTensor()

train_dataset = torchvision.datasets.CIFAR100(
    root=str(DATA_DIR),
    train=True,
    download=False,
    transform=transform,
)

images = torch.stack([image for image, _ in train_dataset])

mean = images.mean(dim=(0, 2, 3))
std = images.std(dim=(0, 2, 3))

print("CIFAR-100 Training Set Pixel Statistics")
print(f"Mean (RGB): {mean.tolist()}")
print(f"Standard deviation (RGB): {std.tolist()}")
