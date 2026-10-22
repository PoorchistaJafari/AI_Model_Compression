
from pathlib import Path

import torch

from model import create_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

model = create_model(num_classes=100)

total_params = sum(
    parameter.numel() for parameter in model.parameters()
)

trainable_params = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

parameter_bytes = sum(
    parameter.numel() * parameter.element_size()
    for parameter in model.parameters()
)

buffer_bytes = sum(
    buffer.numel() * buffer.element_size()
    for buffer in model.buffers()
)

total_bytes = parameter_bytes + buffer_bytes

print("Baseline ResNet-18 Model Size Analysis")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Parameter storage: {parameter_bytes / (1024 ** 2):.2f} MiB")
print(f"Buffer storage: {buffer_bytes / (1024 ** 2):.2f} MiB")
print(f"Total model state: {total_bytes / (1024 ** 2):.2f} MiB")
print(f"Parameter data type: {next(model.parameters()).dtype}")
