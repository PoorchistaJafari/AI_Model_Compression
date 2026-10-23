
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from model import create_model
from prepare_validation import validation_subset

# File paths
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_PATH = MODELS_DIR / "resnet18_baseline.pth"
PRUNED_MODEL_PATH = MODELS_DIR / "resnet18_pruned_20.pth"

# Reproducibility and CPU configuration
torch.manual_seed(42)
torch.set_num_threads(2)
device = torch.device("cpu")

# Validation DataLoader
validation_loader = DataLoader(
    validation_subset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
)

# Load the trained baseline model
model = create_model(num_classes=100).to(device)

state_dict = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True,
)
model.load_state_dict(state_dict)
model.eval()

# Select convolutional weights and final classifier weights
modules_to_prune = [
    (module, "weight")
    for module in model.modules()
    if isinstance(module, nn.Conv2d)
]
modules_to_prune.append((model.fc, "weight"))

# Apply 20% global magnitude pruning
prune.global_unstructured(
    modules_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.20,
)

# Make pruning permanent in the model parameters
for module, parameter_name in modules_to_prune:
    prune.remove(module, parameter_name)

# Measure sparsity in the selected weights
total_weights = 0
zero_weights = 0

for module, parameter_name in modules_to_prune:
    parameter = getattr(module, parameter_name)
    total_weights += parameter.numel()
    zero_weights += (parameter == 0).sum().item()

sparsity = 100.0 * zero_weights / total_weights

# Evaluate the pruned model
all_predictions = []
all_labels = []
total_loss = 0.0
total_images = 0
criterion = nn.CrossEntropyLoss()

start_time = time.perf_counter()

with torch.inference_mode():
    for images, labels in validation_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total_images += labels.size(0)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

evaluation_time = time.perf_counter() - start_time

accuracy = accuracy_score(all_labels, all_predictions)
macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)
average_loss = total_loss / total_images
throughput = total_images / evaluation_time

# Save the pruned model weights
torch.save(model.state_dict(), PRUNED_MODEL_PATH)

# Measure checkpoint size
file_size_bytes = PRUNED_MODEL_PATH.stat().st_size
file_size_mb = file_size_bytes / 1_000_000
file_size_mib = file_size_bytes / (1024 ** 2)

# Report results
print("\nPruned model evaluation completed")
print(f"Baseline checkpoint: {CHECKPOINT_PATH}")
print(f"Pruned checkpoint: {PRUNED_MODEL_PATH}")
print(f"Validation images: {total_images}")
print(f"Pruning target: 20%")
print(f"Selected weights: {total_weights:,}")
print(f"Zero weights: {zero_weights:,}")
print(f"Measured sparsity: {sparsity:.2f}%")
print(f"Validation loss: {average_loss:.4f}")
print(f"Validation accuracy: {accuracy * 100:.2f}%")
print(f"Validation Macro-F1: {macro_f1:.4f}")
print(f"Evaluation time: {evaluation_time:.2f} seconds")
print(f"Throughput: {throughput:.2f} images/second")
print(f"Pruned checkpoint size: {file_size_bytes:,} bytes")
print(f"Pruned checkpoint size: {file_size_mb:.2f} MB")
print(f"Pruned checkpoint size: {file_size_mib:.2f} MiB")
