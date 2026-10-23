
from pathlib import Path
import sys

import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model import create_model

MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_PATH = MODELS_DIR / "resnet18_baseline.pth"
STATE_DICT_PATH = MODELS_DIR / "resnet18_static_int8.pth"

torch.manual_seed(42)
torch.set_num_threads(2)
torch.backends.quantized.engine = "onednn"

device = torch.device("cpu")
example_inputs = (torch.randn(1, 3, 32, 32),)
qconfig_mapping = get_default_qconfig_mapping("onednn")


def build_quantized_model():
    model = create_model(num_classes=100).to(device)

    baseline_state = torch.load(
        BASELINE_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(baseline_state)
    model.eval()

    prepared = prepare_fx(
        model,
        qconfig_mapping,
        example_inputs,
    )
    prepared.eval()

    # The original experiment used 1,024 real training images.
    # This reconstruction uses random calibration data only as a
    # preliminary serialization test, not for scientific evaluation.
    with torch.inference_mode():
        for _ in range(2):
            prepared(torch.randn(1, 3, 32, 32))

    quantized = convert_fx(prepared)
    quantized.eval()
    return quantized


print("Building quantized model...")
model_before = build_quantized_model()

print("Loading saved quantized state_dict...")
saved_state = torch.load(
    STATE_DICT_PATH,
    map_location=device,
    weights_only=True,
)

try:
    model_before.load_state_dict(saved_state, strict=True)
except RuntimeError as exc:
    print("\nSTRICT STATE-DICT LOAD FAILED")
    print(exc)
    print(
        "\nThis indicates that the saved state_dict cannot be "
        "loaded into this reconstructed graph as-is."
    )
    raise

print("Building a second quantized model...")
model_after = build_quantized_model()
model_after.load_state_dict(saved_state, strict=True)
model_after.eval()

test_input = torch.randn(2, 3, 32, 32)

with torch.inference_mode():
    output_before = model_before(test_input)
    output_after = model_after(test_input)

max_difference = (output_before - output_after).abs().max().item()

print("\nReload test results")
print("Output shape:", tuple(output_after.shape))
print(f"Maximum output difference: {max_difference:.8f}")
print(
    "Output comparison:",
    "PASSED"
    if torch.allclose(output_before, output_after, atol=1e-5, rtol=1e-5)
    else "FAILED",
)
