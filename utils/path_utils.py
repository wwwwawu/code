"""Path helpers for experiment names and model caches."""

from __future__ import annotations

import os
import re


def safe_name(value: str) -> str:
    """Return a filesystem-safe, readable name for model identifiers."""
    value = str(value).strip()
    value = re.sub(r"[\\/@:\s]+", "_", value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_.")
    return value or "model"


def backbone_tag(backbone_type: str, backbone_name: str) -> str:
    return f"{safe_name(backbone_type)}_{safe_name(backbone_name)}"


def residual_model_dir_name(backbone_type: str, backbone_name: str) -> str:
    """Return the compact model folder name used by residual-adapter experiments."""
    backbone_type = str(backbone_type).lower()
    name = str(backbone_name).lower()

    if backbone_type == "clip" and "vit-l" in name:
        return "clip-vit-l"
    if backbone_type == "sam" and "vit_l" in name:
        return "sam-vit-l"
    if backbone_type == "dinov3" and ("vitl" in name or "vit-l" in name):
        return "dinov3-vit-l"
    return f"{safe_name(backbone_type)}-{safe_name(backbone_name)}"


def default_train_dir(root: str, dataset: str, tag: str, image_size: int, epochs: int) -> str:
    return os.path.join(root, f"{safe_name(dataset)}_{tag}_{image_size}_e{epochs}")


def default_residual_train_dir(
    root: str,
    dataset: str,
    epochs: int,
    backbone_type: str,
    backbone_name: str,
    experiment_name: str = "residual-adapters",
) -> str:
    return os.path.join(
        root,
        experiment_name,
        f"{safe_name(dataset)}-epoch{epochs}",
        residual_model_dir_name(backbone_type, backbone_name),
    )


def checkpoint_dir(train_dir: str) -> str:
    return os.path.join(train_dir, "checkpoints")


def default_test_dir(checkpoint_path: str, tag: str, sigma: int) -> str:
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    return os.path.join(checkpoint_dir, f"results_{tag}_sigma{sigma}")


def experiment_dir_from_checkpoint(checkpoint_path: str) -> str:
    path = os.path.abspath(checkpoint_path)
    parent = os.path.dirname(path)
    if os.path.basename(parent).lower() == "checkpoints":
        return os.path.dirname(parent)
    return parent


def next_results_dir(experiment_dir: str, prefix: str = "results") -> str:
    """Return the next resultsN directory under an experiment directory."""
    os.makedirs(experiment_dir, exist_ok=True)
    used = []
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for name in os.listdir(experiment_dir):
        match = pattern.match(name)
        if match:
            used.append(int(match.group(1)))
    next_id = 0
    if used:
        next_id = max(used) + 1
    return os.path.join(experiment_dir, f"{prefix}{next_id}")
