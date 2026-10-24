
from pathlib import Path
import sys
import csv
import time

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

sys.path.insert(0, str(SRC_DIR))

from model import create_model
from prepare_dataloader import test_dataset

# Configuration
MODEL_PATH = MODELS_DIR / "resnet18_baseline_extended.pth"
CSV_PATH = REPORTS_DIR / "class_error_analysis.csv"
FIGURE_PATH = FIGURES_DIR / "class_accuracy_worst15.png"

DEVICE = torch.device("cpu")
BATCH_SIZE = 128

torch.set_num_threads(2)

# Check files and dataset
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if len(test_dataset.classes) != 100:
    raise ValueError("Expected 100 CIFAR-100 classes.")

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# Load the extended baseline model
model = create_model(num_classes=100).to(DEVICE)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(state_dict, strict=True)
model.eval()

# Count correct predictions for each class
class_correct = [0] * 100
class_total = [0] * 100

total_correct = 0
total_images = 0
start_time = time.perf_counter()

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        total_correct += (predictions == labels).sum().item()
        total_images += labels.size(0)

        for label, prediction in zip(labels, predictions):
            class_id = label.item()
            class_total[class_id] += 1

            if prediction.item() == class_id:
                class_correct[class_id] += 1

elapsed = time.perf_counter() - start_time

# Calculate per-class accuracy
results = []

for class_id, class_name in enumerate(test_dataset.classes):
    correct = class_correct[class_id]
    total = class_total[class_id]
    accuracy = 100.0 * correct / total if total else 0.0

    results.append({
        "class_id": class_id,
        "class_name": class_name,
        "correct": correct,
        "total": total,
        "accuracy_percent": round(accuracy, 2),
    })

results.sort(key=lambda item: item["accuracy_percent"])

# Save results for all 100 classes
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# Plot the 15 classes with the lowest accuracy
worst15 = results[:15]

plt.figure(figsize=(10, 7))
names = [item["class_name"] for item in worst15]
accuracies = [item["accuracy_percent"] for item in worst15]

plt.barh(names, accuracies)
plt.xlabel("Class Accuracy (%)")
plt.ylabel("CIFAR-100 Class")
plt.title("15 Classes with Lowest Accuracy - Extended FP32 Baseline")
plt.tight_layout()
plt.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
plt.close()

# Print summary
overall_accuracy = 100.0 * total_correct / total_images

print("\nCLASS ERROR ANALYSIS COMPLETED")
print("-" * 45)
print(f"Test images: {total_images}")
print(f"Overall test accuracy: {overall_accuracy:.2f}%")
print(f"Evaluation time: {elapsed:.2f} seconds")

print("\n15 classes with the lowest accuracy:")
for item in worst15:
    print(
        f"{item['class_name']:<20} "
        f"{item['accuracy_percent']:>6.2f}% "
        f"({item['correct']}/{item['total']})"
    )

print(f"\nFull class results saved to: {CSV_PATH}")
print(f"Chart saved to: {FIGURE_PATH}")