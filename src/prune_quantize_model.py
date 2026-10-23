
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
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
# 1. Paths and configuration
# --------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_PATH = MODELS_DIR / "resnet18_baseline.pth"
OUTPUT_PATH = MODELS_DIR / "resnet18_pruned20_static_int8.pth"

torch.manual_seed(42)
torch.set_num_threads(2)
device = torch.device("cpu")
torch.backends.quantized.engine = "onednn"

# --------------------------------------------------
# 2. DataLoaders
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
# 3. Load the original FP32 baseline
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
# 4. Apply 20% global magnitude pruning
# --------------------------------------------------

modules_to_prune = [
    (module, "weight")
    for module in model.modules()
    if isinstance(module, nn.Conv2d)
]
modules_to_prune.append((model.fc, "weight"))

prune.global_unstructured(
    modules_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.20,
)

# Make pruning permanent before quantization
for module, parameter_name in modules_to_prune:
    prune.remove(module, parameter_name)

total_selected_weights = sum(
    getattr(module, parameter_name).numel()
    for module, parameter_name in modules_to_prune
)

zero_selected_weights = sum(
    (getattr(module, parameter_name) == 0).sum().item()
    for module, parameter_name in modules_to_prune
)

sparsity = 100.0 * zero_selected_weights / total_selected_weights

print("\nPruning completed")
print(f"Selected weights: {total_selected_weights:,}")
print(f"Zero weights after pruning: {zero_selected_weights:,}")
print(f"Measured sparsity: {sparsity:.2f}%")

# --------------------------------------------------
# 5. Prepare static quantization
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
# 6. Calibrate using training images
# --------------------------------------------------

print("\nStarting calibration...")
print(f"Calibration images: {calibration_size}")

with torch.inference_mode():
    for images, _ in calibration_loader:
        prepared_model(images.to(device))

print("Calibration completed.")

# --------------------------------------------------
# 7. Convert to static INT8
# --------------------------------------------------

print("\nConverting pruned model to static INT8...")

quantized_model = convert_fx(prepared_model)
quantized_model.eval()

print("Quantization conversion completed.")

# --------------------------------------------------
# 8. Evaluate the combined model
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
# 9. Save the combined model state dictionary
# --------------------------------------------------

torch.save(
    quantized_model.state_dict(),
    OUTPUT_PATH,
)

baseline_size = BASELINE_PATH.stat().st_size
combined_size = OUTPUT_PATH.stat().st_size

baseline_mb = baseline_size / 1_000_000
combined_mb = combined_size / 1_000_000
size_reduction = 100.0 * (1 - combined_size / baseline_size)

# --------------------------------------------------
# 10. Report results
# --------------------------------------------------

print("\nPruning + Static INT8 evaluation completed")
print(f"Output checkpoint: {OUTPUT_PATH}")
print(f"Calibration images: {calibration_size}")
print(f"Validation images: {total_images}")

print(f"Pruning target: 20%")
print(f"Measured selected-weight sparsity: {sparsity:.2f}%")

print(f"Validation loss: {average_loss:.4f}")
print(f"Validation accuracy: {accuracy * 100:.2f}%")
print(f"Validation Macro-F1: {macro_f1:.4f}")

print(f"Evaluation time: {evaluation_time:.2f} seconds")
print(f"Throughput: {throughput:.2f} images/second")

print(f"\nBaseline checkpoint size: {baseline_size:,} bytes")
print(f"Baseline size: {baseline_mb:.2f} MB")
print(f"Combined state-dict size: {combined_size:,} bytes")
print(f"Combined state-dict size: {combined_mb:.2f} MB")
print(f"Measured size reduction: {size_reduction:.2f}%")
