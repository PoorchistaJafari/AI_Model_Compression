
from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

sys.path.insert(0, str(SRC_DIR))

from model import create_model
from prepare_dataloader import test_dataset

MODEL_PATH = MODELS_DIR / "resnet18_baseline_extended.pth"
CSV_PATH = REPORTS_DIR / "confusion_matrix.csv"
FIGURE_PATH = FIGURES_DIR / "confusion_matrix.png"

DEVICE = torch.device("cpu")
BATCH_SIZE = 128

torch.set_num_threads(2)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

model = create_model(num_classes=100).to(DEVICE)
state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)
model.load_state_dict(state_dict, strict=True)
model.eval()

all_labels = []
all_predictions = []

print("Generating predictions on the official test set...")

with torch.inference_mode():
    for images, labels in loader:
        outputs = model(images.to(DEVICE))
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().tolist())

class_names = test_dataset.classes

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
)

# Save the full 100 x 100 confusion matrix as CSV.
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as file:
    import csv
    writer = csv.writer(file)
    writer.writerow(["true_class"] + class_names)

    for index, row in enumerate(cm):
        writer.writerow([class_names[index]] + row.tolist())

# Plot a readable full matrix.
fig, ax = plt.subplots(figsize=(18, 18))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)
display.plot(
    ax=ax,
    include_values=False,
    xticks_rotation="vertical",
    colorbar=False,
)
ax.set_title("Confusion Matrix - Extended FP32 Baseline")
fig.tight_layout()
fig.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
plt.close(fig)

# Report the most common incorrect predictions for the 5 weakest classes.
weak_classes = ["otter", "squirrel", "turtle", "crab", "raccoon"]
class_to_index = {name: i for i, name in enumerate(class_names)}

print("\nMost common incorrect predictions:")
for name in weak_classes:
    true_index = class_to_index[name]
    row = cm[true_index].copy()
    row[true_index] = 0

    top_indices = row.argsort()[::-1][:3]

    print(f"\n{name}:")
    for predicted_index in top_indices:
        print(
            f"  predicted as {class_names[predicted_index]}: "
            f"{row[predicted_index]} images"
        )

print("\nConfusion matrix analysis completed.")
print(f"CSV saved to: {CSV_PATH}")
print(f"Chart saved to: {FIGURE_PATH}")