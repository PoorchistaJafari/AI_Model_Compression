
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from model import create_model

# Path to the trained baseline checkpoint
model_path = PROJECT_ROOT / "models" / "resnet18_baseline.pth"

device = torch.device("cpu")

# Create the model architecture
model = create_model(num_classes=100)

# Load the trained weights
state_dict = torch.load(
    model_path,
    map_location=device,
    weights_only=True,
)
model.load_state_dict(state_dict)
model.eval()

# Count model parameters
total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

# Measure checkpoint file size
file_size_bytes = model_path.stat().st_size
file_size_mb = file_size_bytes / (1024 ** 2)
file_size_mib = file_size_bytes / (1024 ** 2)

print("Baseline model analysis completed")
print(f"Checkpoint: {model_path}")
print(f"Total parameters: {total_parameters:,}")
print(f"Trainable parameters: {trainable_parameters:,}")
print(f"Checkpoint size: {file_size_bytes:,} bytes")
print(f"Checkpoint size: {file_size_bytes / 1_000_000:.2f} MB")
print(f"Checkpoint size: {file_size_mib:.2f} MiB")
