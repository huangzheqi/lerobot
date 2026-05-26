from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


class ResnetCubePoseEstimator:
    """ResNet18-based cube XY estimator from fixed_camera RGB."""

    def __init__(self, model_path: str | Path, device: str | torch.device = "cuda"):
        self.device = torch.device(device)
        self.model = self._build_model().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @staticmethod
    def _build_model() -> nn.Module:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model

    def estimate(self, fixed_rgb: torch.Tensor, z_value: torch.Tensor) -> torch.Tensor:
        """Estimate object xyz from fixed-camera rgb.

        Args:
            fixed_rgb: Tensor [N,H,W,C] or [N,C,H,W], uint8 or float.
            z_value: Tensor [N] or [N,1], robot-frame z to reuse.

        Returns:
            Tensor [N,3] = [x,y,z].
        """
        rgb = fixed_rgb.to(device=self.device, non_blocking=True)
        if rgb.ndim != 4:
            raise ValueError(f"Expected 4D rgb tensor, got shape={tuple(rgb.shape)}")
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.shape[-1] == 3:
            rgb = rgb.permute(0, 3, 1, 2)

        rgb = rgb.float()
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        rgb = torch.nn.functional.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
        rgb = (rgb - self._mean) / self._std

        with torch.inference_mode():
            xy = self.model(rgb)

        z = z_value.to(device=self.device, non_blocking=True).reshape(-1, 1)
        return torch.cat([xy, z], dim=1)
