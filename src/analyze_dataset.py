
from pathlib import Path
from collections import Counter

import torchvision

# 1. Locate the project and dataset folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 2. Load the training dataset
train_dataset = torchvision.datasets.CIFAR100(
    root=str(DATA_DIR),
    train=True,
    download=False,
)
#Load the test dataset
test_dataset = torchvision.datasets.CIFAR100(
    root = str(DATA_DIR),
    train=False,
    download=False,
)

# 3. Count the images belonging to each class
class_counts = Counter(train_dataset.targets)
test_class_counts = Counter(test_dataset.targets)

# 4. Print the class distribution
print("--- CIFAR-100 Class Distribution ---")
print(f"Total training images: {len(train_dataset)}")
print(f"Total classes: {len(train_dataset.classes)}")


for class_id, class_name in enumerate(train_dataset.classes):
    train_count = class_counts[class_id]
    test_count = test_class_counts[class_id]

    print(
        f"{class_name}: "
        f"Train = {train_count}, "
        f"Test = {test_count}"
    )
