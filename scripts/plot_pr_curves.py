import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    import types

    def tqdm(iterable, **kwargs):
        return iterable

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = tqdm
    sys.modules.setdefault("tqdm", tqdm_module)

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    def gaussian_filter(array, sigma=0):
        return array


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import Dataset
from utils.anomaly_detection import generate_anomaly_map_from_tokens
from utils.backbone_adapters import load_visualad_model, resolve_backbone_name
from utils.backbone_config import resolve_features_list
from utils.checkpoint_io import load_trusted_checkpoint
from utils.feature_transform import create_feature_transform
from utils.refinement_head import RefinementHead
from utils.residual_adapters import configure_residual_adapters, load_residual_adapter_state
from utils.transforms import get_transform


DEFAULT_DATA_ROOT = Path("..") / "UW-Bench" / "UW-Bench" / "training_set"
DEFAULT_META_PATH = Path("data_meta") / "uwbench_meta.json"
DEFAULT_OUTPUT_DIR = Path("paper_assets") / "pr_curves"
DEFAULT_OURS_CHECKPOINT = (
    Path("ablation-vitl")
    / "resiual-adapters-spatial-attention-refined-mask"
    / "image756"
    / "adapter_3_6_9_12_spatial_12_16_20_24"
    / "checkpoints"
    / "final_model.pth"
)

METHOD_LABELS = {
    "deeplabv3plus": "DeepLabV3+",
    "segformer": "SegFormer",
    "setr": "SETR",
    "ours": "Ours",
}


class PrintLogger:
    def info(self, message, *args):
        print(message % args if args else message)

    def warning(self, message, *args):
        print("WARNING: " + (message % args if args else message))

    def error(self, message, *args):
        print("ERROR: " + (message % args if args else message))


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def repo_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_baseline_module(method):
    module_path = REPO_ROOT / "baselines" / method / "train.py"
    spec = importlib.util.spec_from_file_location(f"_pr_{method}_train", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_torch_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def update_pr_histogram(pos_hist, neg_hist, scores, masks, num_bins, score_stride=1):
    if scores.dim() == 4:
        scores = scores[:, 0]
    if masks.dim() == 4:
        masks = masks[:, 0]
    elif masks.dim() == 3:
        masks = masks
    else:
        raise RuntimeError(f"Unexpected mask shape: {tuple(masks.shape)}")

    if score_stride > 1:
        scores = scores[..., ::score_stride, ::score_stride]
        masks = masks[..., ::score_stride, ::score_stride]

    scores = scores.detach().float().clamp(0.0, 1.0).cpu().reshape(-1)
    gt = (masks.detach().cpu().reshape(-1) > 0.5)
    bin_ids = torch.floor(scores * (num_bins - 1)).long().clamp(0, num_bins - 1)

    if gt.any():
        pos_hist += torch.bincount(bin_ids[gt], minlength=num_bins).cpu().numpy().astype(np.float64)
    if (~gt).any():
        neg_hist += torch.bincount(bin_ids[~gt], minlength=num_bins).cpu().numpy().astype(np.float64)


def curve_from_histograms(pos_hist, neg_hist):
    thresholds = np.linspace(0.0, 1.0, len(pos_hist), dtype=np.float64)
    tp = np.cumsum(pos_hist[::-1])[::-1]
    fp = np.cumsum(neg_hist[::-1])[::-1]
    total_pos = max(float(pos_hist.sum()), 1.0)
    fn = total_pos - tp

    precision = np.divide(tp, tp + fp, out=np.ones_like(tp, dtype=np.float64), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fn) > 0)

    order = np.argsort(recall)
    pr_auc = float(np.trapz(precision[order], recall[order]))
    best_f1 = float(np.max((2.0 * precision * recall) / np.maximum(precision + recall, 1e-12)))
    best_idx = int(np.argmax((2.0 * precision * recall) / np.maximum(precision + recall, 1e-12)))
    return {
        "thresholds": thresholds,
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
        "best_f1": best_f1,
        "best_threshold": float(thresholds[best_idx]),
        "positive_pixels": int(pos_hist.sum()),
        "negative_pixels": int(neg_hist.sum()),
    }


def build_baseline_model(method, checkpoint, args, device):
    module = load_baseline_module(method)
    config = checkpoint.get("config", {})
    image_size = int(args.image_size or config.get("image_size", 518))
    stretch_to_square = bool(args.stretch_to_square or config.get("stretch_to_square", False))
    image_transform, mask_transform = module.build_transforms(image_size, stretch_to_square)

    if method == "deeplabv3plus":
        os.environ["TORCH_HOME"] = str(repo_path(args.deeplab_model_cache))
        os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
        model = module.DeepLabV3PlusBinary(backbone="resnet101", pretrained_backbone=False)
    elif method == "segformer":
        model_cache = str(repo_path(args.segformer_model_cache))
        os.makedirs(model_cache, exist_ok=True)
        model = module.SegFormerBinary(variant="b2", model_name="", cache_dir=model_cache)
    elif method == "setr":
        model_cache = str(repo_path(args.setr_model_cache))
        os.makedirs(model_cache, exist_ok=True)
        backbone = config.get("backbone", "vit_b16")
        model_name = config.get("model_name", "")
        decoder_dim = int(config.get("decoder_dim", 256))
        model = module.SETRPUPBinary(backbone=backbone, model_name=model_name, cache_dir=model_cache, decoder_dim=decoder_dim)
    else:
        raise ValueError(f"Unsupported baseline method: {method}")

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, image_transform, mask_transform


def evaluate_baseline_curve(method, checkpoint_path, args, device):
    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    model, image_transform, mask_transform = build_baseline_model(method, checkpoint, args, device)
    dataset = Dataset(
        root=args.data_path,
        transform=image_transform,
        target_transform=mask_transform,
        dataset_name=args.dataset_name,
        mode=args.test_mode,
        meta_path=args.meta_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    pos_hist = np.zeros(args.num_thresholds, dtype=np.float64)
    neg_hist = np.zeros(args.num_thresholds, dtype=np.float64)
    with torch.no_grad():
        for items in tqdm(loader, desc=METHOD_LABELS[method]):
            images = items["img"].to(device)
            masks = items["img_mask"].to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            logits = model(images)
            probs = torch.sigmoid(logits)
            update_pr_histogram(pos_hist, neg_hist, probs, masks, args.num_thresholds, args.score_stride)
    return curve_from_histograms(pos_hist, neg_hist)


def make_ours_args(checkpoint, args):
    ours_args = SimpleNamespace(
        backbone_type=None,
        backbone_name=None,
        backbone="ViT-L/14@336px",
        prototype_token_mode="append",
        sam_checkpoint="",
        image_size=518,
        features_list=[6, 12, 18, 24],
        use_residual_adapters=False,
        adapter_layers="shallow",
        adapter_layer_ids="3,6,9,12",
        adapter_type="mlp",
        residual_adapter_mode="",
        adapter_ratio=0.01,
        adapter_bottleneck_ratio=0.25,
        adapter_dropout=0.0,
        use_iou_loss=False,
        use_refined_mask=False,
        refinement_hidden_dim=256,
        refinement_dropout=0.0,
        num_anchors=4,
        stretch_to_square=args.stretch_to_square,
    )
    ours_args.backbone_type = checkpoint.get("backbone_type", "clip")
    ours_args.backbone_name = resolve_backbone_name(
        ours_args.backbone_type,
        checkpoint.get("backbone_name"),
        checkpoint.get("backbone", "ViT-L/14@336px"),
    )
    ours_args.backbone = checkpoint.get("backbone", "ViT-L/14@336px")
    ours_args.prototype_token_mode = checkpoint.get(
        "prototype_token_mode",
        "backbone" if ours_args.backbone_type in {"clip", "dinov2"} else "external",
    )
    ours_args.sam_checkpoint = checkpoint.get("sam_checkpoint", "")
    ours_args.image_size = int(args.image_size or checkpoint.get("image_size", 518))
    ours_args.features_list = checkpoint.get("features_list", ours_args.features_list)
    ours_args.use_residual_adapters = bool(checkpoint.get("use_residual_adapters", False))
    ours_args.adapter_layers = checkpoint.get("adapter_layers", ours_args.adapter_layers)
    ours_args.adapter_layer_ids = checkpoint.get("adapter_layer_ids", ours_args.adapter_layer_ids)
    ours_args.adapter_type = checkpoint.get("adapter_type", ours_args.adapter_type)
    ours_args.residual_adapter_mode = checkpoint.get("residual_adapter_mode", ours_args.residual_adapter_mode)
    ours_args.adapter_ratio = checkpoint.get("adapter_ratio", ours_args.adapter_ratio)
    ours_args.adapter_bottleneck_ratio = checkpoint.get("adapter_bottleneck_ratio", ours_args.adapter_bottleneck_ratio)
    ours_args.adapter_dropout = checkpoint.get("adapter_dropout", ours_args.adapter_dropout)
    ours_args.use_iou_loss = bool(checkpoint.get("use_iou_loss", False))
    ours_args.use_refined_mask = bool(checkpoint.get("use_refined_mask", False))
    ours_args.refinement_hidden_dim = checkpoint.get("refinement_hidden_dim", ours_args.refinement_hidden_dim)
    ours_args.refinement_dropout = checkpoint.get("refinement_dropout", ours_args.refinement_dropout)
    ours_args.num_anchors = int(
        checkpoint.get(
            "num_anchors",
            checkpoint.get("cross_attn_config", {}).get("num_anchors", ours_args.num_anchors),
        )
    )
    return ours_args


def build_ours_components(checkpoint, args, device):
    logger = PrintLogger()
    ours_args = make_ours_args(checkpoint, args)
    model, _, backbone_spec = load_visualad_model(ours_args, device=device)
    ours_args.features_list = resolve_features_list(ours_args.features_list, backbone_spec.num_layers, logger=logger)
    configure_residual_adapters(model, ours_args, backbone_spec, device, logger=logger)
    load_residual_adapter_state(model, checkpoint.get("adapter_state_dict"), logger=logger)
    model.eval()
    model.to(device)

    model.visual.anomaly_token.data = checkpoint["anomaly_token"].to(device)
    model.visual.normal_token.data = checkpoint["normal_token"].to(device)
    ln_post = getattr(model.visual, "ln_post", None)
    if ln_post is not None and checkpoint.get("ln_post_weight") is not None:
        ln_post.weight.data = checkpoint["ln_post_weight"].to(device)
        ln_post.bias.data = checkpoint["ln_post_bias"].to(device)

    feature_dim = int(model.visual.embed_dim)
    layer_transforms = nn.ModuleDict()
    if "layer_transforms" in checkpoint:
        for layer_name, state_dict in checkpoint["layer_transforms"].items():
            hidden_dim = state_dict["mlp.0.weight"].shape[0]
            layer_transforms[layer_name] = create_feature_transform(
                transform_type="mlp",
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                output_dim=feature_dim,
                dropout=0.0,
            ).to(device)
            layer_transforms[layer_name].load_state_dict(state_dict)
            layer_transforms[layer_name].eval()

    cross_attn = None
    if "cross_attn" in checkpoint:
        from utils.spatial_cross_attention import build_layer_adaptive_cross_attention

        config = checkpoint.get("cross_attn_config", {})
        cross_attn = build_layer_adaptive_cross_attention(
            layers=ours_args.features_list,
            embed_dim=feature_dim,
            num_anchors=config.get("num_anchors", 4),
            dropout=config.get("dropout", 0.1),
            max_patches=config.get("max_patches", 4096),
            res_scale_init=config.get("res_scale_init", 0.01),
        ).to(device)
        cross_attn.load_state_dict(checkpoint["cross_attn"])
        cross_attn.eval()

    refinement_head = None
    if ours_args.use_refined_mask and "refinement_head" in checkpoint:
        config = checkpoint.get("refinement_config", {})
        refinement_head = RefinementHead(
            input_dim=config.get("input_dim", feature_dim),
            num_layers=config.get("num_layers", len(ours_args.features_list)),
            hidden_dim=config.get("hidden_dim", ours_args.refinement_hidden_dim),
            dropout=config.get("dropout", ours_args.refinement_dropout),
        ).to(device)
        refinement_head.load_state_dict(checkpoint["refinement_head"])
        refinement_head.eval()

    preprocess, target_transform = get_transform(ours_args)
    return ours_args, model, layer_transforms, cross_attn, refinement_head, preprocess, target_transform


def forward_ours_maps(images, ours_args, model, layer_transforms, cross_attn, refinement_head):
    vision_output = model.encode_image(images, ours_args.features_list)
    anomaly_features = vision_output["anomaly_features"]
    normal_features = vision_output["normal_features"]
    patch_tokens = vision_output["patch_tokens"]
    patch_start_idx = vision_output["patch_start_idx"]
    patch_features_list = [pt[:, patch_start_idx:, :] for pt in patch_tokens]

    if cross_attn is not None:
        adapted_list = cross_attn(anomaly_features, normal_features, patch_features_list, ours_args.features_list)
        anomaly_features_list = [item["anomaly"] for item in adapted_list]
        normal_features_list = [item["normal"] for item in adapted_list]
    else:
        anomaly_features_list = [anomaly_features] * len(patch_tokens)
        normal_features_list = [normal_features] * len(patch_tokens)

    anomaly_map_list = []
    refinement_features_list = []
    for idx, patch_feature in enumerate(patch_tokens):
        anomaly_feat_norm = F.normalize(anomaly_features_list[idx], dim=1, eps=1e-8)
        normal_feat_norm = F.normalize(normal_features_list[idx], dim=1, eps=1e-8)
        transform_key = f"layer_{ours_args.features_list[idx]}"
        if transform_key in layer_transforms:
            batch, tokens, dim = patch_feature.shape
            patch_feature = layer_transforms[transform_key](patch_feature.reshape(-1, dim)).reshape(batch, tokens, dim)
        patch_only_feature = patch_feature[:, patch_start_idx:, :]
        refinement_features_list.append(patch_only_feature)
        anomaly_map = generate_anomaly_map_from_tokens(
            anomaly_feat_norm,
            normal_feat_norm,
            patch_only_feature,
            ours_args.image_size,
        )
        anomaly_map_list.append(anomaly_map)

    if refinement_head is not None:
        anomaly_logits = refinement_head(refinement_features_list, anomaly_map_list, ours_args.image_size)
    else:
        anomaly_logits = torch.stack(anomaly_map_list).sum(dim=0)

    filtered = []
    for idx in range(anomaly_logits.shape[0]):
        filtered_map = gaussian_filter(anomaly_logits[idx].detach().cpu().numpy(), sigma=args_sigma)
        filtered.append(torch.from_numpy(filtered_map))
    return torch.stack(filtered, dim=0).to(images.device)


def evaluate_ours_curve(checkpoint_path, args, device):
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location=device)
    ours_args, model, layer_transforms, cross_attn, refinement_head, preprocess, target_transform = build_ours_components(
        checkpoint,
        args,
        device,
    )
    dataset = Dataset(
        root=args.data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.dataset_name,
        mode=args.test_mode,
        meta_path=args.meta_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.ours_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    pos_hist = np.zeros(args.num_thresholds, dtype=np.float64)
    neg_hist = np.zeros(args.num_thresholds, dtype=np.float64)
    with torch.no_grad():
        for items in tqdm(loader, desc=METHOD_LABELS["ours"]):
            images = items["img"].to(device)
            masks = items["img_mask"].to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            logits = forward_ours_maps(images, ours_args, model, layer_transforms, cross_attn, refinement_head)
            probs = torch.sigmoid(logits).unsqueeze(1)
            update_pr_histogram(pos_hist, neg_hist, probs, masks, args.num_thresholds, args.score_stride)
    return curve_from_histograms(pos_hist, neg_hist)


def write_curve_csv(output_path, curves):
    with open(output_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["method", "threshold", "recall", "precision"])
        for label, curve in curves.items():
            for threshold, recall, precision in zip(curve["thresholds"], curve["recall"], curve["precision"]):
                writer.writerow([label, f"{threshold:.6f}", f"{recall:.8f}", f"{precision:.8f}"])


def plot_curves(output_dir, curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "DeepLabV3+": "#4C78A8",
        "SegFormer": "#F58518",
        "SETR": "#54A24B",
        "Ours": "#D62728",
    }
    plt.figure(figsize=(6.2, 4.8))
    ax = plt.gca()
    for label, curve in curves.items():
        order = np.argsort(curve["recall"])
        ax.plot(
            curve["recall"][order],
            curve["precision"][order],
            linewidth=2.2 if label != "Ours" else 2.8,
            color=colors.get(label, None),
            label=f"{label} ({curve['pr_auc']:.3f})",
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="PR-AUC", frameon=True, loc="lower left", fontsize=10, title_fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=300)
    plt.savefig(output_dir / "pr_curve.pdf")
    plt.close()


def default_method_paths(args):
    return {
        "deeplabv3plus": repo_path(args.deeplab_checkpoint_path),
        "segformer": repo_path(args.segformer_checkpoint_path),
        "setr": repo_path(args.setr_checkpoint_path),
        "ours": repo_path(args.ours_checkpoint_path),
    }


def main():
    parser = argparse.ArgumentParser("Plot pixel-level PR curves for waterlogging segmentation")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--meta_path", type=str, default=str(DEFAULT_META_PATH))
    parser.add_argument("--dataset_name", type=str, default="uwbench")
    parser.add_argument("--test_mode", type=str, default="test")
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--methods", nargs="*", default=["deeplabv3plus", "segformer", "setr", "ours"],
                        choices=["deeplabv3plus", "segformer", "setr", "ours"])
    parser.add_argument("--deeplab_checkpoint_path", type=str,
                        default="experiments/deeplabv3plus_resnet101/checkpoints/epoch_50.pth")
    parser.add_argument("--segformer_checkpoint_path", type=str,
                        default="experiments/segformer_b2/checkpoints/epoch_50.pth")
    parser.add_argument("--setr_checkpoint_path", type=str,
                        default="experiments/setr_vit_b16/checkpoints/epoch_50.pth")
    parser.add_argument("--ours_checkpoint_path", type=str, default=str(DEFAULT_OURS_CHECKPOINT))
    parser.add_argument("--deeplab_model_cache", type=str, default="baselines/deeplabv3plus/model_cache")
    parser.add_argument("--segformer_model_cache", type=str, default="baselines/segformer/model_cache")
    parser.add_argument("--setr_model_cache", type=str, default="baselines/setr/model_cache")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--ours_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_thresholds", type=int, default=201)
    parser.add_argument("--score_stride", type=int, default=1,
                        help="subsample pixels by this spatial stride before accumulating PR histograms")
    parser.add_argument("--image_size", type=int, default=0, help="0 means use checkpoint config")
    parser.add_argument("--stretch_to_square", action="store_true")
    parser.add_argument("--sigma", type=int, default=4)
    parser.add_argument("--skip_missing", action="store_true", default=True)
    parser.add_argument("--strict", action="store_true", help="fail if a selected checkpoint is missing")
    parser.add_argument("--seed", type=int, default=42)
    parsed = parser.parse_args()

    setup_seed(parsed.seed)
    device = torch.device(parsed.device)
    save_dir = repo_path(parsed.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    parsed.data_path = str(repo_path(parsed.data_path))
    parsed.meta_path = str(repo_path(parsed.meta_path))

    global args_sigma
    args_sigma = int(parsed.sigma)

    checkpoint_paths = default_method_paths(parsed)
    curves = {}
    summary = {}
    for method in parsed.methods:
        checkpoint_path = checkpoint_paths[method]
        if not checkpoint_path.exists():
            message = f"[skip] {METHOD_LABELS[method]} checkpoint not found: {checkpoint_path}"
            if parsed.strict:
                raise FileNotFoundError(message)
            print(message)
            continue

        if method == "ours":
            curve = evaluate_ours_curve(checkpoint_path, parsed, device)
        else:
            curve = evaluate_baseline_curve(method, checkpoint_path, parsed, device)

        label = METHOD_LABELS[method]
        curves[label] = curve
        summary[label] = {
            "checkpoint_path": str(checkpoint_path),
            "pr_auc": curve["pr_auc"],
            "best_f1": curve["best_f1"],
            "best_threshold": curve["best_threshold"],
            "positive_pixels": curve["positive_pixels"],
            "negative_pixels": curve["negative_pixels"],
        }

    if not curves:
        raise RuntimeError("No PR curves were generated. Check checkpoint paths or remove --strict.")

    write_curve_csv(save_dir / "pr_curve.csv", curves)
    with open(save_dir / "pr_summary.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "dataset_name": parsed.dataset_name,
                "data_path": parsed.data_path,
                "meta_path": parsed.meta_path,
                "test_mode": parsed.test_mode,
                "num_thresholds": parsed.num_thresholds,
                "score_stride": parsed.score_stride,
                "legend_metric": "PR-AUC",
                "methods": summary,
            },
            fp,
            indent=2,
            ensure_ascii=False,
        )
    plot_curves(save_dir, curves)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved PR curve outputs to {save_dir}")


if __name__ == "__main__":
    main()
