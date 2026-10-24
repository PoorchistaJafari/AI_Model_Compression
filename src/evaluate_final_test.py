
from pathlib import Path
import csv
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

# --------------------------------------------------
# 1. Project paths and imports
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

sys.path.insert(0, str(SRC_DIR))

from model import create_model
from prepare_dataloader import test_dataset
from prepare_validation import training_subset

BASELINE_PATH = MODELS_DIR / "resnet18_baseline_extended.pth"
INT8_PATH = MODELS_DIR / "resnet18_extended_static_int8.pth"
COMBINED_PATH = (
    MODELS_DIR / "resnet18_extended_pruned20_static_int8.pth"
)

CSV_PATH = REPORTS_DIR / "final_test_results.csv"

# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------
SEED = 42
BATCH_SIZE = 128
CALIBRATION_IMAGES = 1024
BACKEND = "onednn"
DEVICE = torch.device("cpu")

torch.manual_seed(SEED)
torch.set_num_threads(2)

if BACKEND not in torch.backends.quantized.supported_engines:
    raise RuntimeError(
        f"Quantization backend {BACKEND!r} is unavailable. "
        f"Available engines: {torch.backends.quantized.supported_engines}"
    )

torch.backends.quantized.engine = BACKEND

# --------------------------------------------------
# 3. Validate files and dataset
# --------------------------------------------------
for path in (BASELINE_PATH, INT8_PATH, COMBINED_PATH):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

if len(test_dataset) != 10000:
    raise ValueError(
        f"Expected 10000 official test images, found {len(test_dataset)}."
    )

if len(training_subset) < CALIBRATION_IMAGES:
    raise ValueError("Not enough training images for calibration.")

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

example_inputs = (torch.randn(1, 3, 32, 32),)
qconfig_mapping = get_default_qconfig_mapping(BACKEND)


# --------------------------------------------------
# 4. Load the extended FP32 baseline
# --------------------------------------------------
def load_baseline():
    model = create_model(num_classes=100).to(DEVICE)

    state_dict = torch.load(
        BASELINE_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# --------------------------------------------------
# 5. Prepare and calibrate a quantized model
# --------------------------------------------------
def calibrate_and_convert(model, calibration_loader):
    model.eval()

    prepared = prepare_fx(
        model,
        qconfig_mapping,
        example_inputs,
    )
    prepared.eval()

    calibrated_images = 0

    with torch.inference_mode():
        for images, _ in calibration_loader:
            remaining = CALIBRATION_IMAGES - calibrated_images
            if remaining <= 0:
                break

            images = images[:remaining].to(DEVICE)
            prepared(images)
            calibrated_images += images.size(0)

    if calibrated_images != CALIBRATION_IMAGES:
        raise RuntimeError(
            f"Expected {CALIBRATION_IMAGES} calibration images, "
            f"used {calibrated_images}."
        )

    quantized = convert_fx(prepared)
    quantized.eval()
    return quantized


def build_static_int8():
    # Match the original static INT8 experiment:
    # the first 1024 images of training_subset.
    calibration_loader = DataLoader(
        training_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = load_baseline()
    quantized = calibrate_and_convert(model, calibration_loader)

    saved_state = torch.load(
        INT8_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    quantized.load_state_dict(saved_state, strict=True)
    quantized.eval()
    return quantized


# --------------------------------------------------
# 6. Reconstruct the 20% pruning + INT8 model
# --------------------------------------------------
def apply_original_pruning(model):
    import torch.nn.utils.prune as prune

    parameters_to_prune = [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
    ]
    parameters_to_prune.append((model.fc, "weight"))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.20,
    )

    # Make the pruning permanent, matching the original experiment.
    for module, name in parameters_to_prune:
        prune.remove(module, name)


def build_combined_model():
    # Recreate the pruning operation using the same FP32 baseline.
    model = load_baseline()
    apply_original_pruning(model)

    # Match the original combined experiment's calibration indices.
    generator = torch.Generator().manual_seed(SEED)
    calibration_indices = torch.randperm(
        len(training_subset),
        generator=generator,
    )[:CALIBRATION_IMAGES].tolist()

    calibration_subset = Subset(
        training_subset,
        calibration_indices,
    )
    calibration_loader = DataLoader(
        calibration_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    quantized = calibrate_and_convert(model, calibration_loader)

    saved_state = torch.load(
        COMBINED_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    quantized.load_state_dict(saved_state, strict=True)
    quantized.eval()
    return quantized


# --------------------------------------------------
# 7. Evaluate a model on the official test set
# --------------------------------------------------
def evaluate_model(name, model, checkpoint_path):
    model.eval()

    criterion = nn.CrossEntropyLoss()
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    total_images = 0

    start_time = time.perf_counter()

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_images += batch_size

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    elapsed = time.perf_counter() - start_time

    result = {
        "model": name,
        "test_images": total_images,
        "test_loss": total_loss / total_images,
        "accuracy_percent": 100.0 * accuracy_score(
            all_labels, all_predictions
        ),
        "macro_f1": f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
        "evaluation_seconds": elapsed,
        "throughput_images_per_second": total_images / elapsed,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_size_MB": checkpoint_path.stat().st_size / 1_000_000,
    }

    print(f"\n{name}")
    print("-" * 45)
    print(f"Test images: {result['test_images']}")
    print(f"Test loss: {result['test_loss']:.4f}")
    print(f"Test accuracy: {result['accuracy_percent']:.2f}%")
    print(f"Test Macro-F1: {result['macro_f1']:.4f}")
    print(f"Evaluation time: {result['evaluation_seconds']:.2f} seconds")
    print(
        "Throughput: "
        f"{result['throughput_images_per_second']:.2f} images/second"
    )
    print(f"Checkpoint size: {result['checkpoint_size_MB']:.2f} MB")

    return result


# --------------------------------------------------
# 8. Run the final evaluation and save results
# --------------------------------------------------
def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    print("Evaluating extended FP32 baseline...")
    baseline = load_baseline()
    results.append(
        evaluate_model("Extended FP32 baseline", baseline, BASELINE_PATH)
    )
    del baseline

    print("\nReconstructing static INT8 model...")
    static_int8 = build_static_int8()
    results.append(
        evaluate_model("Static INT8", static_int8, INT8_PATH)
    )
    del static_int8

    print("\nReconstructing pruning + INT8 model...")
    combined = build_combined_model()
    results.append(
        evaluate_model("20% pruning + Static INT8", combined, COMBINED_PATH)
    )
    del combined

    # Save all final test metrics to a separate CSV file.
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 55)
    print("FINAL TEST EVALUATION COMPLETED")
    print(f"Results saved to: {CSV_PATH}")
    print("Use these results separately from validation results.")


if __name__ == "__main__":
    main()