
import torch
from torch.utils.data import Subset, random_split

from prepare_dataloader import train_dataset, validation_dataset

validation_size = 5000
training_size = len(train_dataset) - validation_size

# Generate reproducible indices for the split
training_indices, validation_indices = random_split(
    range(len(train_dataset)),
    [training_size, validation_size],
    generator=torch.Generator().manual_seed(42),
)

# Use the same indices with datasets that have separate transforms
training_subset = Subset(train_dataset, list(training_indices))
validation_subset = Subset(validation_dataset, list(validation_indices))

print("Training and validation split successful")
print(f"Training images: {len(training_subset)}")
print(f"Validation images: {len(validation_subset)}")
