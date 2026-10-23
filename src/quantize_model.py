
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model import create_model
from prepare_validation import validation_subset

# File paths
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_PATH = MODELS_DIR / "resnet18_baseline.pth"
QUANTIZED_PATH = MODELS_DIR / "resnet18_dynamic_int8.pth"

# CPU configuration and reproducibility
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

# Load the original FP32 baseline model
model = create_model(num_classes=100).to(device)

state_dict = torch.load(
    BASELINE_PATH,
    map_location=device,
    weights_only=True,
)
model.load_state_dict(state_dict)
model.eval()

# Apply dynamic INT8 quantization to supported Linear layers
quantized_model = torch.ao.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8,
)
quantized_model.eval()

# Evaluate the quantized model
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

        outputs = quantized_model(images)
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

# Save the quantized model
torch.save(quantized_model.state_dict(), QUANTIZED_PATH)

# Measure checkpoint size
baseline_size = BASELINE_PATH.stat().st_size
quantized_size = QUANTIZED_PATH.stat().st_size

baseline_mb = baseline_size / 1_000_000
quantized_mb = quantized_size / 1_000_000
size_reduction = 100 * (1 - quantized_size / baseline_size)

# Report results
print("\nDynamic INT8 quantization evaluation completed")
print(f"Baseline checkpoint: {BASELINE_PATH}")
print(f"Quantized checkpoint: {QUANTIZED_PATH}")
print(f"Validation images: {total_images}")
print(f"Validation loss: {average_loss:.4f}")
print(f"Validation accuracy: {accuracy * 100:.2f}%")
print(f"Validation Macro-F1: {macro_f1:.4f}")
print(f"Evaluation time: {evaluation_time:.2f} seconds")
print(f"Throughput: {throughput:.2f} images/second")

print(f"\nBaseline checkpoint size: {baseline_size:,} bytes")
print(f"Baseline size: {baseline_mb:.2f} MB")
print(f"Quantized checkpoint size: {quantized_size:,} bytes")
print(f"Quantized size: {quantized_mb:.2f} MB")
print(f"Checkpoint size reduction: {size_reduction:.2f}%")

# Confirm which layer was quantized
print("\nQuantized model structure:")
print(quantized_model.fc)
