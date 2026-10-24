
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

# Locate the project root and source directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from model import create_model
from prepare_validation import training_subset, validation_subset

# Define model file paths
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODELS_DIR / "resnet18_baseline_extended.pth"
QUANTIZED_MODEL_PATH = MODELS_DIR / "resnet18_extended_static_int8.pth"

# Set random seed and CPU configuration
torch.manual_seed(42)
torch.set_num_threads(2)
device = torch.device("cpu")

# Select the quantization backend
if "onednn" not in torch.backends.quantized.supported_engines:
    raise RuntimeError(
        "The oneDNN quantization backend is not supported in this environment."
    )

torch.backends.quantized.engine = "onednn"

# Prepare data loaders
calibration_loader = DataLoader(
    training_subset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
)

validation_loader = DataLoader(
    validation_subset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
)

# Create the model and load the extended baseline weights
model = create_model(num_classes=100).to(device)

state_dict = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True,
)
model.load_state_dict(state_dict)
model.eval()

print("Extended baseline model loaded successfully.")

# Configure FX static quantization
qconfig_mapping = get_default_qconfig_mapping("onednn")
example_inputs = (torch.randn(1, 3, 32, 32),)

prepared_model = prepare_fx(
    model,
    qconfig_mapping,
    example_inputs,
)

# Calibrate the model using 1,024 training images
calibration_images = 0
calibration_limit = 1024

with torch.inference_mode():
    for images, _ in calibration_loader:
        images = images.to(device)

        prepared_model(images)
        calibration_images += images.size(0)

        if calibration_images >= calibration_limit:
            break

print(f"Calibration completed using {calibration_images} training images.")

# Convert the prepared model to a quantized INT8 model
quantized_model = convert_fx(prepared_model)
quantized_model.eval()

print("Static INT8 quantization completed.")

# Evaluate the quantized model on the validation dataset
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

# Calculate evaluation metrics
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

# Save the quantized model state dictionary
torch.save(quantized_model.state_dict(), QUANTIZED_MODEL_PATH)

# Measure the saved checkpoint size
file_size_bytes = QUANTIZED_MODEL_PATH.stat().st_size
file_size_mb = file_size_bytes / 1_000_000
file_size_mib = file_size_bytes / (1024 ** 2)

# Calculate size reduction relative to the extended FP32 baseline
baseline_size_bytes = CHECKPOINT_PATH.stat().st_size
size_reduction = (
    100.0 * (baseline_size_bytes - file_size_bytes) / baseline_size_bytes
)

# Report experimental results
print("\nExtended model static INT8 evaluation completed.")
print(f"Input model: {CHECKPOINT_PATH.name}")
print(f"Output model: {QUANTIZED_MODEL_PATH.name}")
print(f"Quantization backend: {torch.backends.quantized.engine}")
print(f"Calibration images: {calibration_images}")
print(f"Validation images: {total_images}")
print(f"Validation loss: {average_loss:.4f}")
print(f"Validation accuracy: {accuracy * 100:.2f}%")
print(f"Validation Macro-F1: {macro_f1:.4f}")
print(f"Evaluation time: {evaluation_time:.2f} seconds")
print(f"Throughput: {throughput:.2f} images/second")
print(f"Baseline checkpoint size: {baseline_size_bytes:,} bytes")
print(f"Quantized checkpoint size: {file_size_bytes:,} bytes")
print(f"Quantized checkpoint size: {file_size_mb:.2f} MB")
print(f"Quantized checkpoint size: {file_size_mib:.2f} MiB")
print(f"Measured size reduction: {size_reduction:.2f}%")
