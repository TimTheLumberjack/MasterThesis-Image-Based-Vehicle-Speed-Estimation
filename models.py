from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torchvision import models


class SpeedRegressionModel(nn.Module):
    """Wraps a backbone with a regression head for speed prediction."""

    def __init__(
        self, backbone: nn.Module, in_features: int, dropout_rate: float = 0.2
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.regressor(features)


def _efficientnet_v2_l() -> nn.Module:
    model = models.efficientnet_v2_l(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Identity()
    return SpeedRegressionModel(model, in_features)


def _efficientnet_v2_s() -> nn.Module:
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Identity()
    return SpeedRegressionModel(model, in_features)


def _resnet18() -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Identity()
    return SpeedRegressionModel(model, in_features)


def _resnet50() -> nn.Module:
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Identity()
    return SpeedRegressionModel(model, in_features)


MODEL_REGISTRY: Dict[str, callable] = {
    "efficientnetv2-L": _efficientnet_v2_l,
    "efficientnetv2-S": _efficientnet_v2_s,
    "resnet18": _resnet18,
    "resnet50": _resnet50,
}


class SimpleFlowCNN(nn.Module):
    """Lightweight CNN backbone for RGB optical-flow inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.regressor(x)


def _simple_cnn() -> nn.Module:
    return SimpleFlowCNN()


MODEL_REGISTRY["simple"] = _simple_cnn


def create_model(model_name: str) -> nn.Module:
    """Create a regression model for the requested backbone name."""

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")
    return MODEL_REGISTRY[model_name]()
