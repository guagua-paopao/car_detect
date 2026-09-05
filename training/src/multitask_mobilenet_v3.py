from __future__ import annotations

from torch import Tensor, nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_V2_S_Weights,
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    efficientnet_v2_s,
    convnext_tiny,
    mobilenet_v3_large,
    resnet50,
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


ARCHITECTURES = (
    "mobilenet_v3_large",
    "mobilenet_v3_large_dual",
    "mobilenet_v3_large_foreground_dual",
    "efficientnet_v2_s",
    "resnet50",
    "convnext_tiny",
)


class MultiTaskVehicleAttributes(nn.Module):
    def __init__(
        self,
        body_type_count: int,
        color_count: int,
        *,
        architecture: str = "mobilenet_v3_large",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if body_type_count <= 0 or color_count <= 0:
            raise ValueError("attribute class counts must be positive")
        if architecture in {
            "mobilenet_v3_large",
            "mobilenet_v3_large_dual",
            "mobilenet_v3_large_foreground_dual",
        }:
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_large(weights=weights)
        elif architecture == "efficientnet_v2_s":
            weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_v2_s(weights=weights)
        elif architecture == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = resnet50(weights=weights)
        elif architecture == "convnext_tiny":
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.backbone = convnext_tiny(weights=weights)
        else:
            raise ValueError(f"unsupported attribute architecture: {architecture}")
        self.architecture = architecture
        if architecture == "resnet50":
            feature_count = int(self.backbone.fc.in_features)
            self.backbone.fc = nn.Identity()
        else:
            feature_count = int(self.backbone.classifier[-1].in_features)
            self.backbone.classifier[-1] = nn.Identity()
        if architecture in {
            "mobilenet_v3_large_dual",
            "mobilenet_v3_large_foreground_dual",
        }:
            # Separate projection towers reduce destructive interference between
            # type and color gradients while retaining the lightweight backbone.
            self.body_type_head = nn.Sequential(
                nn.Linear(feature_count, 512), nn.Hardswish(), nn.Dropout(0.20),
                nn.Linear(512, body_type_count),
            )
            self.color_head = nn.Sequential(
                nn.Linear(feature_count, 512), nn.Hardswish(), nn.Dropout(0.20),
                nn.Linear(512, color_count),
            )
        else:
            self.body_type_head = nn.Linear(feature_count, body_type_count)
            self.color_head = nn.Linear(feature_count, color_count)

    def forward_features(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Return body/color embeddings before their classification heads."""

        if self.architecture == "mobilenet_v3_large_foreground_dual":
            # Preserve the detector-facing tensor contract while exposing a
            # color-specific feature path.  Body type uses the full crop;
            # color removes the outer feature-map ring, which suppresses road,
            # sky and neighboring-object context introduced by bbox margins.
            spatial = self.backbone.features(images)
            body_features = self.backbone.avgpool(spatial).flatten(1)
            body_features = self.backbone.classifier(body_features)
            color_spatial = spatial[:, :, 1:-1, 1:-1]
            color_features = self.backbone.avgpool(color_spatial).flatten(1)
            color_features = self.backbone.classifier(color_features)
            return body_features, color_features
        features = self.backbone(images)
        return features, features

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        body_features, color_features = self.forward_features(images)
        return self.body_type_head(body_features), self.color_head(color_features)


class MultiTaskMobileNetV3(MultiTaskVehicleAttributes):
    """Backward-compatible constructor for existing checkpoints and callers."""

    def __init__(self, body_type_count: int, color_count: int, *, pretrained: bool = True) -> None:
        super().__init__(
            body_type_count,
            color_count,
            architecture="mobilenet_v3_large",
            pretrained=pretrained,
        )


def architecture_from_checkpoint(checkpoint: dict) -> str:
    value = str(checkpoint.get("architecture", "mobilenet_v3_large_multitask"))
    return value.removesuffix("_multitask")


def model_from_checkpoint(checkpoint: dict, *, pretrained: bool = False) -> MultiTaskVehicleAttributes:
    return MultiTaskVehicleAttributes(
        len(checkpoint["body_types"]),
        len(checkpoint["colors"]),
        architecture=architecture_from_checkpoint(checkpoint),
        pretrained=pretrained,
    )
