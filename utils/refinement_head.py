"""Lightweight mask reconstruction head for ProtoWD response maps."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class RefinementHead(nn.Module):
    """Fuse multi-layer patch features and a coarse anomaly map into a refined mask."""

    def __init__(self, input_dim: int, num_layers: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        mid_dim = max(32, self.hidden_dim // 2)
        norm_groups = 8 if self.hidden_dim % 8 == 0 else 1
        mid_norm_groups = 8 if mid_dim % 8 == 0 else 1
        self.projections = nn.ModuleList([
            nn.Linear(self.input_dim, self.hidden_dim) for _ in range(self.num_layers)
        ])
        fusion_dim = self.hidden_dim * self.num_layers + 1
        self.decoder = nn.Sequential(
            nn.Conv2d(fusion_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, self.hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(self.hidden_dim, mid_dim, kernel_size=3, padding=1),
            nn.GroupNorm(mid_norm_groups, mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, 1, kernel_size=1),
        )

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
        batch, num_patches, channels = tokens.shape
        side = int(num_patches ** 0.5)
        if side * side != num_patches:
            raise RuntimeError(f"RefinementHead expects square patch tokens, got {num_patches}")
        return tokens.transpose(1, 2).reshape(batch, channels, side, side)

    def forward(
        self,
        patch_features_list: Sequence[torch.Tensor],
        anomaly_maps_list: Sequence[torch.Tensor],
        image_size: int,
    ) -> torch.Tensor:
        if len(patch_features_list) != self.num_layers:
            raise RuntimeError(
                f"RefinementHead expected {self.num_layers} feature maps, got {len(patch_features_list)}"
            )
        projected_maps = []
        target_hw = None
        for tokens, projection in zip(patch_features_list, self.projections):
            projected = projection(tokens)
            feature_map = self._tokens_to_map(projected)
            if target_hw is None:
                target_hw = feature_map.shape[-2:]
            elif feature_map.shape[-2:] != target_hw:
                feature_map = F.interpolate(feature_map, size=target_hw, mode="bilinear", align_corners=False)
            projected_maps.append(feature_map)

        coarse_map = torch.stack(anomaly_maps_list).sum(dim=0).unsqueeze(1)
        coarse_map = F.interpolate(coarse_map, size=target_hw, mode="bilinear", align_corners=False)
        fused = torch.cat(projected_maps + [coarse_map], dim=1)
        refined_logits = self.decoder(fused)
        refined_logits = F.interpolate(
            refined_logits,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        return refined_logits.squeeze(1)
