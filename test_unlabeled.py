import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    def gaussian_filter(array, sigma=0):
        return array

from utils.anomaly_detection import generate_anomaly_map_from_tokens
from utils.backbone_adapters import load_visualad_model, resolve_backbone_name
from utils.backbone_config import resolve_features_list
from utils.experiment_io import save_args_json
from utils.feature_transform import create_feature_transform
from utils.logger import get_logger
from utils.refinement_head import RefinementHead
from utils.residual_adapters import configure_residual_adapters, load_residual_adapter_state
from utils.scoring import DEFAULT_TOPK_RATIO, reduce_anomaly_map
from utils.transforms import get_transform
from utils.visualization import visualize_unlabeled_prediction


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_UNLABELED_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", "datasets", "城市内涝识别数据"))
DEFAULT_SAVE_PATH = os.path.join("experiments", "residual-adapters-refined-mask", "codex-test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def list_images(root):
    image_paths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(dirpath, filename))
    return sorted(image_paths)


def safe_name(root, image_path):
    rel = os.path.relpath(image_path, root)
    base, _ = os.path.splitext(rel)
    return base.replace("\\", "__").replace("/", "__").replace(" ", "_")


def load_components(args, device, logger):
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    args.backbone_type = args.backbone_type or checkpoint.get("backbone_type", "clip")
    args.backbone_name = resolve_backbone_name(
        args.backbone_type,
        args.backbone_name or checkpoint.get("backbone_name"),
        checkpoint.get("backbone", "ViT-L/14@336px"),
    )
    args.backbone = checkpoint.get("backbone", "ViT-L/14@336px")
    args.sam_checkpoint = args.sam_checkpoint or checkpoint.get("sam_checkpoint", "")
    args.image_size = checkpoint.get("image_size", args.image_size)
    args.features_list = checkpoint.get("features_list", args.features_list)
    args.use_residual_adapters = bool(
        getattr(args, "use_residual_adapters", False) or checkpoint.get("use_residual_adapters", False)
    )
    args.adapter_layers = checkpoint.get("adapter_layers", getattr(args, "adapter_layers", "shallow"))
    args.adapter_layer_ids = checkpoint.get("adapter_layer_ids", getattr(args, "adapter_layer_ids", "3,6,9,12"))
    args.adapter_type = checkpoint.get("adapter_type", getattr(args, "adapter_type", "mlp"))
    args.residual_adapter_mode = checkpoint.get("residual_adapter_mode", "")
    args.adapter_ratio = checkpoint.get("adapter_ratio", getattr(args, "adapter_ratio", 0.01))
    args.adapter_bottleneck_ratio = checkpoint.get("adapter_bottleneck_ratio", getattr(args, "adapter_bottleneck_ratio", 0.25))
    args.adapter_dropout = checkpoint.get("adapter_dropout", getattr(args, "adapter_dropout", 0.0))
    args.use_iou_loss = bool(checkpoint.get("use_iou_loss", False))
    args.use_refined_mask = bool(checkpoint.get("use_refined_mask", getattr(args, "use_refined_mask", False)))
    args.refinement_hidden_dim = checkpoint.get("refinement_hidden_dim", getattr(args, "refinement_hidden_dim", 256))
    args.refinement_dropout = checkpoint.get("refinement_dropout", getattr(args, "refinement_dropout", 0.0))

    model, _, backbone_spec = load_visualad_model(args, device=device)
    args.features_list = resolve_features_list(args.features_list, backbone_spec.num_layers, logger=logger)
    configure_residual_adapters(model, args, backbone_spec, device, logger=logger)
    load_residual_adapter_state(model, checkpoint.get("adapter_state_dict"), logger=logger)
    model.eval()
    model.to(device)

    model.visual.anomaly_token.data = checkpoint["anomaly_token"].to(device)
    model.visual.normal_token.data = checkpoint["normal_token"].to(device)
    ln_post = getattr(model.visual, "ln_post", None)
    if ln_post is not None and checkpoint.get("ln_post_weight") is not None:
        ln_post.weight.data = checkpoint["ln_post_weight"].to(device)
        ln_post.bias.data = checkpoint["ln_post_bias"].to(device)

    feature_dim = model.visual.embed_dim
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
            layers=args.features_list,
            embed_dim=feature_dim,
            num_anchors=config.get("num_anchors", 4),
            dropout=config.get("dropout", 0.1),
            max_patches=config.get("max_patches", 4096),
            res_scale_init=config.get("res_scale_init", 0.01),
        ).to(device)
        cross_attn.load_state_dict(checkpoint["cross_attn"])
        cross_attn.eval()

    refinement_head = None
    if args.use_refined_mask and "refinement_head" in checkpoint:
        config = checkpoint.get("refinement_config", {})
        refinement_head = RefinementHead(
            input_dim=config.get("input_dim", feature_dim),
            num_layers=config.get("num_layers", len(args.features_list)),
            hidden_dim=config.get("hidden_dim", args.refinement_hidden_dim),
            dropout=config.get("dropout", args.refinement_dropout),
        ).to(device)
        refinement_head.load_state_dict(checkpoint["refinement_head"])
        refinement_head.eval()

    return model, layer_transforms, cross_attn, refinement_head, backbone_spec


def predict_one(model, layer_transforms, cross_attn, refinement_head, image, args):
    vision_output = model.encode_image(image, args.features_list)
    anomaly_features = vision_output["anomaly_features"]
    normal_features = vision_output["normal_features"]
    patch_tokens = vision_output["patch_tokens"]
    patch_start_idx = vision_output["patch_start_idx"]

    patch_features_list = [pt[:, patch_start_idx:, :] for pt in patch_tokens]
    if cross_attn is not None:
        adapted_list = cross_attn(anomaly_features, normal_features, patch_features_list, args.features_list)
        anomaly_features_list = [a["anomaly"] for a in adapted_list]
        normal_features_list = [a["normal"] for a in adapted_list]
    else:
        anomaly_features_list = [anomaly_features] * len(patch_tokens)
        normal_features_list = [normal_features] * len(patch_tokens)

    anomaly_map_list = []
    refinement_features_list = []
    for idx, patch_feature in enumerate(patch_tokens):
        anomaly_feat_norm = F.normalize(anomaly_features_list[idx], dim=1, eps=1e-8)
        normal_feat_norm = F.normalize(normal_features_list[idx], dim=1, eps=1e-8)

        transform_key = f"layer_{args.features_list[idx]}"
        if transform_key in layer_transforms:
            batch_size, num_patches, feat_dim = patch_feature.shape
            patch_feature = layer_transforms[transform_key](patch_feature.view(-1, feat_dim)).view(
                batch_size, num_patches, feat_dim
            )
        patch_only_feature = patch_feature[:, patch_start_idx:, :]
        refinement_features_list.append(patch_only_feature)
        anomaly_map = generate_anomaly_map_from_tokens(
            anomaly_feat_norm,
            normal_feat_norm,
            patch_only_feature,
            args.image_size,
        )
        anomaly_map_list.append(anomaly_map)

    if refinement_head is not None:
        final_anomaly_map = refinement_head(refinement_features_list, anomaly_map_list, args.image_size).cpu()
    else:
        final_anomaly_map = torch.stack(anomaly_map_list).sum(dim=0).cpu()
    filtered_map = gaussian_filter(final_anomaly_map[0].numpy(), sigma=args.sigma)
    return torch.from_numpy(filtered_map).unsqueeze(0)


def summarize(records):
    scores = np.array([item["score"] for item in records], dtype=np.float32)
    area_ratios = np.array([item["pred_area_ratio"] for item in records], dtype=np.float32)
    return {
        "num_images": int(len(records)),
        "score": {
            "mean": float(scores.mean()) if scores.size else 0.0,
            "std": float(scores.std()) if scores.size else 0.0,
            "min": float(scores.min()) if scores.size else 0.0,
            "max": float(scores.max()) if scores.size else 0.0,
        },
        "pred_area_ratio": {
            "mean": float(area_ratios.mean()) if area_ratios.size else 0.0,
            "std": float(area_ratios.std()) if area_ratios.size else 0.0,
            "min": float(area_ratios.min()) if area_ratios.size else 0.0,
            "max": float(area_ratios.max()) if area_ratios.size else 0.0,
        },
        "note": "No GT masks are available, so Precision/Recall/F1/IoU/AUROC cannot be computed as supervised metrics.",
    }


def main(args):
    setup_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.save_path, exist_ok=True)
    analysis_dir = os.path.join(args.save_path, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    logger = get_logger(args.save_path, filename="test.log")

    image_paths = list_images(args.image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {args.image_dir}")

    model, layer_transforms, cross_attn, refinement_head, backbone_spec = load_components(args, device, logger)
    preprocess, _ = get_transform(args)
    save_args_json(
        args,
        os.path.join(args.save_path, "test_config.json"),
        extra={
            "backbone_embed_dim": backbone_spec.embed_dim,
            "backbone_num_layers": backbone_spec.num_layers,
            "backbone_patch_size": backbone_spec.patch_size,
            "unlabeled": True,
            "num_images": len(image_paths),
            "metrics_note": "No GT masks are available; saved metrics are prediction statistics only.",
        },
    )
    logger.info("Unlabeled testing | images=%d | image_dir=%s", len(image_paths), args.image_dir)
    logger.info("Checkpoint: %s | Save path: %s", args.checkpoint_path, args.save_path)
    logger.info("Backbone: %s:%s | Refined mask: %s | Sigma: %s | eval_threshold: %.3f",
                args.backbone_type, args.backbone_name, args.use_refined_mask, args.sigma, args.eval_threshold)

    records = []
    with torch.no_grad():
        for image_path in tqdm(image_paths):
            pil_image = Image.open(image_path).convert("RGB")
            image = preprocess(pil_image).unsqueeze(0).to(device)
            anomaly_map = predict_one(model, layer_transforms, cross_attn, refinement_head, image, args)
            score = float(reduce_anomaly_map(anomaly_map, mode="topk_mean", topk_ratio=DEFAULT_TOPK_RATIO).item())
            logits = anomaly_map.squeeze().numpy()
            pred_prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
            pred_area_ratio = float((pred_prob >= args.eval_threshold).mean())
            rel_path = os.path.relpath(image_path, args.image_dir)
            out_name = f"score_{score:.3f}_{safe_name(args.image_dir, image_path)}.png"
            visualize_unlabeled_prediction(
                original_image=image.detach().cpu(),
                prediction_map=anomaly_map,
                score=score,
                img_path=rel_path,
                save_path=os.path.join(analysis_dir, out_name),
                eval_threshold=args.eval_threshold,
            )
            records.append({
                "img_path": image_path,
                "relative_path": rel_path,
                "score": score,
                "pred_area_ratio": pred_area_ratio,
                "visualization": os.path.join("analysis", out_name),
            })

    records = sorted(records, key=lambda item: item["score"])
    csv_path = os.path.join(args.save_path, "per_image_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["index", "img_path", "relative_path", "score", "pred_area_ratio", "visualization"])
        writer.writeheader()
        for idx, item in enumerate(records):
            row = dict(item)
            row["index"] = idx
            writer.writerow(row)

    metrics_payload = {
        "dataset_name": args.test_dataset,
        "image_dir": args.image_dir,
        "checkpoint_path": args.checkpoint_path,
        "backbone_type": args.backbone_type,
        "backbone_name": args.backbone_name,
        "use_residual_adapters": args.use_residual_adapters,
        "adapter_type": args.adapter_type,
        "use_refined_mask": args.use_refined_mask,
        "sigma": args.sigma,
        "eval_threshold": args.eval_threshold,
        "metrics": summarize(records),
    }
    with open(os.path.join(args.save_path, "metrics_summary.json"), "w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2, ensure_ascii=False)
    logger.info("Saved unlabeled prediction stats to %s", args.save_path)
    logger.info("No GT masks are available; supervised metrics such as IoU/F1 are not computed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("VisualAD Unlabeled Image Folder Test", add_help=True)
    parser.add_argument("--image_dir", type=str, default=DEFAULT_UNLABELED_ROOT, help="folder containing unlabeled images")
    parser.add_argument("--save_path", type=str, default=DEFAULT_SAVE_PATH, help="path to save unlabeled test results")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to trained model checkpoint")
    parser.add_argument("--test_dataset", type=str, default="codex-test", help="name recorded in outputs")
    parser.add_argument("--backbone_type", type=str, default=None, choices=["clip", "dinov3", "sam"])
    parser.add_argument("--backbone_name", type=str, default=None)
    parser.add_argument("--backbone", type=str, default="ViT-L/14@336px")
    parser.add_argument("--sam_checkpoint", type=str, default="")
    parser.add_argument("--features_list", type=int, nargs="*", default=[6, 12, 18, 24])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--stretch_to_square", action="store_true")
    parser.add_argument("--sigma", type=int, default=4)
    parser.add_argument("--eval_threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--use_residual_adapters", action="store_true")
    parser.add_argument("--adapter_layers", type=str, default="shallow", choices=["shallow", "middle", "all", "custom"])
    parser.add_argument("--adapter_layer_ids", type=str, default="3,6,9,12")
    parser.add_argument("--adapter_type", type=str, default="mlp", choices=["mlp", "conv"])
    parser.add_argument("--adapter_ratio", type=float, default=0.01)
    parser.add_argument("--adapter_bottleneck_ratio", type=float, default=0.25)
    parser.add_argument("--adapter_dropout", type=float, default=0.0)
    parser.add_argument("--use_refined_mask", action="store_true")
    parser.add_argument("--refinement_hidden_dim", type=int, default=256)
    parser.add_argument("--refinement_dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
