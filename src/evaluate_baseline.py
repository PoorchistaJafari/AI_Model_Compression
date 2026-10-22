
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

MODELS_DIR = PROJECT_ROOT / "models"
model_path = MODELS_DIR / "resnet18_baseline.pth"

torch.manual_seed(42)
torch.set_num_threads(2)

device = torch.device("cpu")

validation_loader = DataLoader(
    validation_subset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
)

model = create_model(num_classes=100).to(device)

state_dict = torch.load(
    model_path,
    map_location=device,
    weights_only=True,
)
model.load_state_dict(state_dict)
model.eval()

criterion = nn.CrossEntropyLoss()

all_predictions = []
all_labels = []
total_loss = 0.0
total_images = 0

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

elapsed = time.perf_counter() - start_time

accuracy = accuracy_score(all_labels, all_predictions)
macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)
average_loss = total_loss / total_images
throughput = total_images / elapsed

print("Baseline evaluation completed")
print(f"Model checkpoint: {model_path}")
print(f"Validation images: {total_images}")
print(f"Validation loss: {average_loss:.4f}")
print(f"Validation accuracy: {accuracy * 100:.2f}%")
print(f"Validation Macro-F1: {macro_f1:.4f}")
print(f"Evaluation time: {elapsed:.2f} seconds")
print(f"Throughput: {throughput:.2f} images/second")
