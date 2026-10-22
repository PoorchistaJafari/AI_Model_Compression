
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from model import create_model
from prepare_dataloader import train_dataset
from prepare_validation import training_subset, validation_subset

# Create the directory for model checkpoints
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility and CPU configuration
torch.manual_seed(42)
torch.set_num_threads(2)

device = torch.device("cpu")

# DataLoaders
train_loader = DataLoader(
    training_subset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
)

validation_loader = DataLoader(
    validation_subset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
)

# Model, loss function, and optimizer
model = create_model(num_classes=100).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
)

# Full training configuration
num_epochs = 3
best_validation_accuracy = -1.0

model_path = MODELS_DIR / "resnet18_baseline.pth"

for epoch in range(num_epochs):
    model.train()

    running_loss = 0.0
    training_correct = 0
    training_total = 0

    start_time = time.perf_counter()

    # Training loop
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        training_total += labels.size(0)
        training_correct += (predictions == labels).sum().item()

        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(train_loader):
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}",
                flush=True,
            )

    epoch_loss = running_loss / training_total
    training_accuracy = 100.0 * training_correct / training_total

    # Validation loop
    model.eval()

    validation_correct = 0
    validation_total = 0
    validation_loss_sum = 0.0

    with torch.inference_mode():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            validation_loss_sum += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            validation_total += labels.size(0)
            validation_correct += (predictions == labels).sum().item()

    validation_loss = validation_loss_sum / validation_total
    validation_accuracy = 100.0 * validation_correct / validation_total

    elapsed = time.perf_counter() - start_time

    print(f"\nResults for epoch {epoch + 1}/{num_epochs}")
    print(f"Training loss: {epoch_loss:.4f}")
    print(f"Training accuracy: {training_accuracy:.2f}%")
    print(f"Validation loss: {validation_loss:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.2f}%")
    print(f"Epoch time: {elapsed / 60:.2f} minutes")

    # Save only when validation accuracy improves
    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        torch.save(model.state_dict(), model_path)
        print(f"Best model checkpoint saved: {model_path}")

    print("-" * 50, flush=True)

print("\nTraining completed")
print(f"Best validation accuracy: {best_validation_accuracy:.2f}%")
print(f"Best checkpoint: {model_path}")
