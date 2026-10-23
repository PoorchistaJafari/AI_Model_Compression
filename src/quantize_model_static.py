
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score

from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model import create_model
from prepare_validation import training_subset, validation_subset

# --------------------------------------------------
# 1. File paths and CPU configuration
# --------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_PATH = MODELS_DIR / "resnet18_baseline.pth"
QUANTIZED_PATH = MODELS_DIR / "resnet18_static_int8.pth"

torch.manual_seed(42)
torch.set_num_threads(2)
device = torch.device("cpu")

torch.backends.quantized.engine = "onednn"

# --------------------------------------------------
# 2. Prepare calibration and validation DataLoaders
# --------------------------------------------------

calibration_size = min(1024, len(training_subset))

calibration_subset = Subset(
    training_subset,
    range(calibration_size),
)

calibration_loader = DataLoader(
    calibration_subset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
)

validation_loader = DataLoader(
    validation_subset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
)

# --------------------------------------------------
# 3. Load the trained FP32 baseline model
# --------------------------------------------------

model = create_model(num_classes=100).to(device)

state_dict = torch.load(
    BASELINE_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(state_dict)
model.eval()

# --------------------------------------------------
# 4. Prepare the model for static quantization
# --------------------------------------------------

example_inputs = (torch.randn(1, 3, 32, 32),)

qconfig_mapping = get_default_qconfig_mapping("onednn")

prepared_model = prepare_fx(
    model,
    qconfig_mapping,
    example_inputs,
)

prepared_model.eval()

# --------------------------------------------------
# 5. Calibrate the model using training images
# --------------------------------------------------

print("\nStarting calibration...")
print(f"Calibration images: {calibration_size}")

with torch.inference_mode():
    for images, _ in calibration_loader:
        prepared_model(images.to(device))

print("Calibration completed.")

# --------------------------------------------------
# 6. Convert supported operations to quantized form
# --------------------------------------------------

print("\nConverting model to quantized form...")

quantized_model = convert_fx(prepared_model)
quantized_model.eval()

print("Model conversion completed.")

# --------------------------------------------------
# 7. Evaluate the quantized model
# --------------------------------------------------

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

# --------------------------------------------------
# 8. Save the quantized model
# --------------------------------------------------

# Save the converted model's state dictionary.
# This file is suitable for measuring the serialized weights,
# but is not necessarily a standalone deployable model.
torch.save(
    quantized_model.state_dict(),
    QUANTIZED_PATH,
)

# --------------------------------------------------
# 9. Measure checkpoint sizes
# --------------------------------------------------

baseline_size = BASELINE_PATH.stat().st_size
quantized_size = QUANTIZED_PATH.stat().st_size

baseline_mb = baseline_size / 1_000_000
quantized_mb = quantized_size / 1_000_000

size_reduction = 100.0 * (
    1 - quantized_size / baseline_size
)

# --------------------------------------------------
# 10. Report results
# --------------------------------------------------

print("\nStatic INT8 quantization evaluation completed")
print(f"Baseline checkpoint: {BASELINE_PATH}")
print(f"Quantized checkpoint: {QUANTIZED_PATH}")

print(f"Calibration images: {calibration_size}")
print(f"Validation images: {total_images}")

print(f"Validation loss: {average_loss:.4f}")
print(f"Validation accuracy: {accuracy * 100:.2f}%")
print(f"Validation Macro-F1: {macro_f1:.4f}")

print(f"Evaluation time: {evaluation_time:.2f} seconds")
print(f"Throughput: {throughput:.2f} images/second")

print(f"\nBaseline checkpoint size: {baseline_size:,} bytes")
print(f"Baseline size: {baseline_mb:.2f} MB")

print(f"Quantized state-dict size: {quantized_size:,} bytes")
print(f"Quantized state-dict size: {quantized_mb:.2f} MB")

print(f"Measured size change: {size_reduction:.2f}%")

print("\nQuantized model structure:")
print(quantized_model)
