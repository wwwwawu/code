"""Smoke-test a VisualAD backbone adapter with one synthetic image."""

import argparse
import json
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.anomaly_detection import generate_anomaly_map_from_tokens
from utils.backbone_adapters import load_visualad_model, resolve_backbone_name
from utils.backbone_config import resolve_features_list


def main():
    parser = argparse.ArgumentParser("VisualAD backbone smoke test")
    parser.add_argument("--backbone_type", default="clip", choices=["clip", "dinov3", "sam"])
    parser.add_argument("--backbone_name", default=None)
    parser.add_argument("--backbone", default="ViT-L/14@336px")
    parser.add_argument("--sam_checkpoint", default="")
    parser.add_argument("--features_list", type=int, nargs="*", default=[6, 12, 18, 24])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    args.backbone_name = resolve_backbone_name(args.backbone_type, args.backbone_name, args.backbone)
    device = torch.device(args.device)
    model, _, spec = load_visualad_model(args, device=device)
    model.eval()
    args.features_list = resolve_features_list(args.features_list, spec.num_layers)

    dummy = torch.rand(1, 3, args.image_size, args.image_size, device=device)
    with torch.no_grad():
        out = model.encode_image(dummy, args.features_list)
        patch_tokens = out["patch_tokens"]
        patch_start_idx = out["patch_start_idx"]
        anomaly_map = generate_anomaly_map_from_tokens(
            out["anomaly_features"],
            out["normal_features"],
            patch_tokens[-1][:, patch_start_idx:, :],
            args.image_size,
        )

    payload = {
        "backbone_type": args.backbone_type,
        "backbone_name": args.backbone_name,
        "embed_dim": spec.embed_dim,
        "num_layers": spec.num_layers,
        "features_list": args.features_list,
        "patch_shapes": [list(tokens.shape) for tokens in patch_tokens],
        "anomaly_map_shape": list(anomaly_map.shape),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
