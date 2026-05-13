"""Residual adapter helpers for AA-CLIP style backbone adaptation."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import torch

from .feature_transform import create_residual_adapter


ADAPTER_STATE_KEYS = ("residual_adapter", "residual_adapters")


def parse_adapter_layer_ids(value) -> List[int]:
    """Parse adapter layer ids from a comma string, list, tuple, or int."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        items = value.replace(";", ",").split(",")
    elif isinstance(value, Iterable):
        items = list(value)
    else:
        return []

    layer_ids: List[int] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        layer_ids.append(int(text))
    return layer_ids


def resolve_adapter_layer_ids(
    mode: str,
    custom_layer_ids,
    feature_layers: Sequence[int],
    total_layers: int,
) -> List[int]:
    """Resolve adapter placement into valid 1-based transformer layer ids."""
    if total_layers <= 0:
        return []

    mode = (mode or "shallow").lower()
    custom_ids = parse_adapter_layer_ids(custom_layer_ids)

    if mode == "custom":
        requested = custom_ids
    elif mode == "all":
        requested = list(range(1, total_layers + 1))
    elif mode == "middle":
        requested = [round(total_layers * ratio) for ratio in (0.25, 0.5, 0.75, 1.0)]
    elif custom_ids:
        requested = custom_ids
    elif feature_layers:
        requested = list(feature_layers[: min(4, len(feature_layers))])
    else:
        upper = max(1, total_layers // 2)
        requested = [round(v) for v in np.linspace(1, upper, num=min(4, upper))]

    resolved = sorted({min(max(1, int(layer)), total_layers) for layer in requested if int(layer) > 0})
    return resolved


def attach_clip_residual_adapters(
    model,
    layer_ids: Sequence[int],
    adapter_ratio: float,
    bottleneck_ratio: float,
    dropout: float,
    adapter_type: str,
    device,
) -> None:
    """Attach residual adapters to selected CLIP ViT residual blocks."""
    if not layer_ids:
        return

    width = int(model.visual.embed_dim)
    hidden_dim = max(16, int(width * bottleneck_ratio))
    blocks = model.visual.transformer.resblocks

    for layer_id in layer_ids:
        block = blocks[layer_id - 1]
        block.residual_adapter = create_residual_adapter(
            adapter_type=adapter_type,
            input_dim=width,
            hidden_dim=hidden_dim,
            output_dim=width,
            dropout=dropout,
            init_scale=adapter_ratio,
            patch_start_idx=3,
            sequence_first=True,
        ).to(device)


def configure_residual_adapters(model, args, backbone_spec, device, logger=None) -> List[int]:
    """Configure residual adapters for CLIP, SAM, or DINOv3 backbones."""
    if not getattr(args, "use_residual_adapters", False):
        args.adapter_layer_ids = []
        return []

    layer_ids = resolve_adapter_layer_ids(
        getattr(args, "adapter_layers", "shallow"),
        getattr(args, "adapter_layer_ids", ""),
        getattr(args, "features_list", []),
        int(backbone_spec.num_layers),
    )
    args.adapter_layer_ids = layer_ids

    adapter_ratio = float(getattr(args, "adapter_ratio", 0.01))
    bottleneck_ratio = float(getattr(args, "adapter_bottleneck_ratio", 0.25))
    dropout = float(getattr(args, "adapter_dropout", 0.0))
    adapter_type = getattr(args, "adapter_type", "mlp")
    args.adapter_type = adapter_type

    backbone_type = getattr(args, "backbone_type", "clip")
    if backbone_type == "clip":
        attach_clip_residual_adapters(model, layer_ids, adapter_ratio, bottleneck_ratio, dropout, adapter_type, device)
        args.residual_adapter_mode = "internal_block"
    elif hasattr(model, "adapter") and hasattr(model.adapter, "configure_residual_adapters"):
        model.adapter.configure_residual_adapters(
            layer_ids=layer_ids,
            adapter_ratio=adapter_ratio,
            bottleneck_ratio=bottleneck_ratio,
            dropout=dropout,
            adapter_type=adapter_type,
            device=device,
        )
        mode = getattr(model.adapter, "residual_adapter_mode", None)
        args.residual_adapter_mode = mode
        if logger is not None and backbone_type == "dinov3" and mode == "patch_output":
            logger.warning(
                "DINOv3 transformer blocks were not found; residual adapters are applied to "
                "selected patch-token outputs as a fallback."
            )
    else:
        raise ValueError(f"Residual adapters are not supported for backbone {getattr(args, 'backbone_type', 'unknown')}")

    if logger is not None:
        logger.info(
            "Residual adapters enabled | type=%s | layers=%s | ratio=%.6f | bottleneck_ratio=%.3f",
            adapter_type,
            layer_ids,
            adapter_ratio,
            bottleneck_ratio,
        )
    return layer_ids


def iter_residual_adapter_parameters(model):
    """Yield trainable residual adapter parameters from any supported backbone."""
    for name, param in model.named_parameters():
        if any(key in name for key in ADAPTER_STATE_KEYS):
            yield param


def collect_residual_adapter_state(model):
    """Return only residual adapter tensors from a model state dict."""
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if any(key in name for key in ADAPTER_STATE_KEYS)
    }


def load_residual_adapter_state(model, state_dict, logger=None) -> None:
    """Load residual adapter weights with non-strict matching."""
    if not state_dict:
        return
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    adapter_missing = [key for key in missing if any(marker in key for marker in ADAPTER_STATE_KEYS)]
    if logger is not None:
        if adapter_missing:
            logger.warning(f"Missing residual adapter keys when loading checkpoint: {adapter_missing}")
        if unexpected:
            logger.warning(f"Unexpected residual adapter checkpoint keys: {unexpected}")
