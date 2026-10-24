
import os
import time

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score

from model import create_model
from prepare_validation import training_subset, validation_subset
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx


# Configuration
SEED = 42
PRUNING_AMOUNT = 0.20
CALIBRATION_IMAGES = 1024
BATCH_SIZE = 128
BACKEND = "onednn"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

INPUT_PATH = os.path.join(
    MODEL_DIR, "resnet18_baseline_extended.pth"
)
OUTPUT_PATH = os.path.join(
    MODEL_DIR, "resnet18_extended_pruned20_static_int8.pth"
)

DEVICE = torch.device("cpu")


def load_model(path):
    model = create_model(num_classes=100)

    state_dict = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def get_data():
    generator = torch.Generator().manual_seed(SEED)

    calibration_indices = torch.randperm(
        len(training_subset),
        generator=generator
    )[:CALIBRATION_IMAGES].tolist()

    calibration_dataset = Subset(
        training_subset,
        calibration_indices
    )

    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    val_loader = DataLoader(
        validation_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return calibration_loader, val_loader


def apply_global_pruning(model, amount):
    parameters_to_prune = []

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            parameters_to_prune.append((module, "weight"))

    parameters_to_prune.append((model.fc, "weight"))

    total_weights = sum(
        module.weight.numel()
        for module, _ in parameters_to_prune
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )

    # Make pruning permanent.
    for module, name in parameters_to_prune:
        prune.remove(module, name)

    zero_weights = sum(
        int((module.weight == 0).sum().item())
        for module, _ in parameters_to_prune
    )

    sparsity = 100.0 * zero_weights / total_weights

    print("\nPruning Results")
    print("----------------")
    print(f"Selected weights: {total_weights:,}")
    print(f"Zero weights: {zero_weights:,}")
    print(f"Measured sparsity: {sparsity:.2f}%")

    return sparsity


def evaluate_model(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_images = 0
    all_targets = []
    all_predictions = []

    start_time = time.perf_counter()

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, targets)
            predictions = outputs.argmax(dim=1)

            batch_size = targets.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (
                predictions == targets
            ).sum().item()
            total_images += batch_size

            all_targets.extend(targets.tolist())
            all_predictions.extend(predictions.tolist())

    elapsed_time = time.perf_counter() - start_time

    average_loss = total_loss / total_images
    accuracy = 100.0 * total_correct / total_images

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    throughput = total_images / elapsed_time

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "elapsed_time": elapsed_time,
        "throughput": throughput,
        "images": total_images
    }


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(2)

    if not os.path.isfile(INPUT_PATH):
        raise FileNotFoundError(
            f"Input checkpoint not found: {INPUT_PATH}"
        )

    if len(training_subset) < CALIBRATION_IMAGES:
        raise ValueError(
            "Training subset has fewer images than required "
            "for calibration."
        )

    if len(validation_subset) != 5000:
        raise ValueError(
            f"Expected 5000 validation images, "
            f"but found {len(validation_subset)}."
        )

    supported_engines = torch.backends.quantized.supported_engines

    if BACKEND not in supported_engines:
        raise RuntimeError(
            f"Backend '{BACKEND}' is not supported. "
            f"Available backends: {supported_engines}"
        )

    torch.backends.quantized.engine = BACKEND

    print("Loading extended FP32 baseline...")
    model = load_model(INPUT_PATH)

    print("\nPreparing calibration and validation data...")
    calibration_loader, val_loader = get_data()

    print(f"Calibration images: {len(calibration_loader.dataset):,}")
    print(f"Validation images: {len(val_loader.dataset):,}")

    print("\nEvaluating extended FP32 baseline...")
    baseline_results = evaluate_model(model, val_loader)
    baseline_size = os.path.getsize(INPUT_PATH)

    print("\nApplying 20% global magnitude pruning...")
    sparsity = apply_global_pruning(model, PRUNING_AMOUNT)

    print("\nPreparing static INT8 quantization...")
    model.eval()

    qconfig_mapping = get_default_qconfig_mapping(BACKEND)
    example_inputs = (torch.randn(1, 3, 32, 32),)

    prepared_model = prepare_fx(
        model,
        qconfig_mapping,
        example_inputs
    )

    print(
        f"\nCalibrating with {CALIBRATION_IMAGES:,} training images..."
    )

    calibrated_images = 0

    with torch.inference_mode():
        for images, _ in calibration_loader:
            remaining = CALIBRATION_IMAGES - calibrated_images

            if remaining <= 0:
                break

            images = images[:remaining].to(DEVICE)
            prepared_model(images)
            calibrated_images += images.size(0)

    if calibrated_images != CALIBRATION_IMAGES:
        raise RuntimeError(
            f"Expected {CALIBRATION_IMAGES} calibration images, "
            f"but used {calibrated_images}."
        )

    print(f"Calibration images used: {calibrated_images:,}")

    print("\nConverting model to INT8...")
    quantized_model = convert_fx(prepared_model)
    quantized_model.eval()

    print("\nEvaluating combined model on validation subset...")
    combined_results = evaluate_model(quantized_model, val_loader)

    torch.save(quantized_model.state_dict(), OUTPUT_PATH)

    quantized_size = os.path.getsize(OUTPUT_PATH)

    size_reduction = (
        100.0 * (1 - quantized_size / baseline_size)
    )

    accuracy_change = (
        combined_results["accuracy"]
        - baseline_results["accuracy"]
    )

    print("\n" + "=" * 55)
    print("COMBINED PRUNING + STATIC INT8 RESULTS")
    print("=" * 55)

    print("\nExtended FP32 Baseline")
    print(f"Validation images: {baseline_results['images']:,}")
    print(f"Validation loss: {baseline_results['loss']:.4f}")
    print(f"Validation accuracy: {baseline_results['accuracy']:.2f}%")
    print(f"Validation Macro-F1: {baseline_results['macro_f1']:.4f}")
    print(f"Evaluation time: {baseline_results['elapsed_time']:.2f} seconds")
    print(f"Throughput: {baseline_results['throughput']:.2f} images/second")
    print(f"Checkpoint size: {baseline_size:,} bytes")

    print("\n20% Pruning + Static INT8")
    print(f"Pruning sparsity before quantization: {sparsity:.2f}%")
    print(f"Validation images: {combined_results['images']:,}")
    print(f"Validation loss: {combined_results['loss']:.4f}")
    print(f"Validation accuracy: {combined_results['accuracy']:.2f}%")
    print(f"Validation Macro-F1: {combined_results['macro_f1']:.4f}")
    print(f"Evaluation time: {combined_results['elapsed_time']:.2f} seconds")
    print(f"Throughput: {combined_results['throughput']:.2f} images/second")
    print(f"Quantized checkpoint size: {quantized_size:,} bytes")
    print(f"Size reduction: {size_reduction:.2f}%")

    print("\nComparison with extended FP32 baseline")
    print(f"Accuracy change: {accuracy_change:+.2f} percentage points")

    print(f"\nOutput checkpoint: {OUTPUT_PATH}")
    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()