
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from model import create_model
from prepare_validation import training_subset, validation_subset
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

FP32_PATH = MODELS_DIR / "resnet18_baseline_extended.pth"
STATIC_INT8_PATH = MODELS_DIR / "resnet18_extended_static_int8.pth"
COMBINED_INT8_PATH = (
    MODELS_DIR / "resnet18_extended_pruned20_static_int8.pth"
)

NUM_THREADS = 2
BATCH_SIZES = [1, 32]
WARMUP_RUNS = 10
REPEATS = 5
MAX_BATCHES_PER_REPEAT = 100
CALIBRATION_IMAGES = 1024

torch.set_num_threads(NUM_THREADS)

if "onednn" not in torch.backends.quantized.supported_engines:
    raise RuntimeError(
        "The oneDNN quantization backend is not supported."
    )

torch.backends.quantized.engine = "onednn"


# --------------------------------------------------
# Load FP32 model
# --------------------------------------------------

def load_fp32_model():
    model = create_model(num_classes=100)

    state_dict = torch.load(
        FP32_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


# --------------------------------------------------
# Reconstruct and load FX static INT8 model
# --------------------------------------------------

def load_quantized_model(checkpoint_path, apply_pruning=False):
    model = create_model(num_classes=100)

    state_dict = torch.load(
        FP32_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    # Reproduce the pruning step used by the combined experiment.
    if apply_pruning:
        import torch.nn.utils.prune as prune

        parameters_to_prune = []

        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                parameters_to_prune.append((module, "weight"))

        parameters_to_prune.append((model.fc, "weight"))

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.20,
        )

        for module, parameter_name in parameters_to_prune:
            prune.remove(module, parameter_name)

    qconfig_mapping = get_default_qconfig_mapping("onednn")
    example_inputs = (torch.randn(1, 3, 32, 32),)

    prepared_model = prepare_fx(
        model,
        qconfig_mapping,
        example_inputs,
    )

    # Calibrate with a deterministic subset of training images.
    calibration_subset = Subset(
        training_subset,
        list(range(min(CALIBRATION_IMAGES, len(training_subset)))),
    )

    calibration_loader = DataLoader(
        calibration_subset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    calibrated_images = 0

    with torch.inference_mode():
        for images, _ in calibration_loader:
            prepared_model(images)
            calibrated_images += images.size(0)

    print(
        f"Calibration for {checkpoint_path.name}: "
        f"{calibrated_images} training images"
    )

    quantized_model = convert_fx(prepared_model)

    quantized_state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    quantized_model.load_state_dict(
        quantized_state_dict,
        strict=True,
    )

    quantized_model.eval()

    return quantized_model


# --------------------------------------------------
# Benchmark one model
# --------------------------------------------------

def benchmark_model(model, batch_size, validation_loader):
    model.eval()

    # Warm-up using the requested batch size.
    warmup_input = torch.randn(batch_size, 3, 32, 32)

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(warmup_input)

    latency_results = []
    throughput_results = []

    for repeat_index in range(REPEATS):
        processed_images = 0
        start_time = time.perf_counter()

        with torch.inference_mode():
            for batch_index, (images, _) in enumerate(validation_loader):
                if batch_index >= MAX_BATCHES_PER_REPEAT:
                    break

                # Keep the same batch size for each experiment.
                images = images[:batch_size]

                if images.size(0) == 0:
                    continue

                model(images)
                processed_images += images.size(0)

        elapsed = time.perf_counter() - start_time

        if processed_images == 0 or elapsed <= 0:
            raise RuntimeError(
                "Benchmark processed no images or elapsed time was invalid."
            )

        latency_ms = elapsed * 1000 / processed_images
        throughput = processed_images / elapsed

        latency_results.append(latency_ms)
        throughput_results.append(throughput)

        print(
            f"  Repeat {repeat_index + 1}/{REPEATS}: "
            f"{latency_ms:.3f} ms/image, "
            f"{throughput:.2f} images/second"
        )

    return {
        "latency_median": statistics.median(latency_results),
        "throughput_median": statistics.median(throughput_results),
        "latency_runs": latency_results,
        "throughput_runs": throughput_results,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    required_files = [
        FP32_PATH,
        STATIC_INT8_PATH,
        COMBINED_INT8_PATH,
    ]

    missing_files = [
        path for path in required_files if not path.exists()
    ]

    if missing_files:
        print("ERROR: Missing model checkpoint files:")
        for path in missing_files:
            print(f"  {path}")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    validation_loader = DataLoader(
        validation_subset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    print("Loading Extended Baseline FP32...")
    fp32_model = load_fp32_model()

    print("Loading Static INT8...")
    static_int8_model = load_quantized_model(
        STATIC_INT8_PATH,
        apply_pruning=False,
    )

    print("Loading Pruning 20% + Static INT8...")
    combined_model = load_quantized_model(
        COMBINED_INT8_PATH,
        apply_pruning=True,
    )

    models = [
        ("Extended Baseline FP32", fp32_model),
        ("Static INT8", static_int8_model),
        ("Pruning 20% + Static INT8", combined_model),
    ]

    output_lines = [
        "Repeated CPU Inference Benchmark",
        f"CPU threads: {torch.get_num_threads()}",
        "Quantization backend: onednn",
        f"Warm-up runs: {WARMUP_RUNS}",
        f"Repeats: {REPEATS}",
        f"Maximum batches per repeat: {MAX_BATCHES_PER_REPEAT}",
        f"Calibration images: {CALIBRATION_IMAGES}",
        "Reported latency and throughput are medians across repeats.",
        "",
    ]

    for model_name, model in models:
        for batch_size in BATCH_SIZES:
            print(
                f"\nBenchmarking {model_name}, "
                f"batch size={batch_size}"
            )

            result = benchmark_model(
                model,
                batch_size,
                validation_loader,
            )

            output_lines.extend([
                f"Model: {model_name}",
                f"Batch size: {batch_size}",
                (
                    f"Median latency: "
                    f"{result['latency_median']:.3f} ms/image"
                ),
                (
                    f"Median throughput: "
                    f"{result['throughput_median']:.2f} images/second"
                ),
                (
                    "Latency per repeat (ms/image): "
                    + ", ".join(
                        f"{value:.3f}"
                        for value in result["latency_runs"]
                    )
                ),
                (
                    "Throughput per repeat (images/second): "
                    + ", ".join(
                        f"{value:.2f}"
                        for value in result["throughput_runs"]
                    )
                ),
                "",
            ])

    results_path = REPORTS_DIR / "benchmark_results.txt"

    results_path.write_text(
        "\n".join(output_lines),
        encoding="utf-8",
    )

    print("\nBenchmark completed.")
    print(f"Results saved to: {results_path}")
    print("\n" + "\n".join(output_lines))


if __name__ == "__main__":
    main()