
import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=100):
    model = models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.maxpool = nn.Identity()

    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes,
    )

    return model


if __name__ == "__main__":
    model = create_model(num_classes=100)
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        output = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("ResNet-18 model created successfully")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
