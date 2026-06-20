import ProtoWD_lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from dataset import Dataset
from utils.logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
import json
from utils.transforms import get_transform
from utils.metrics import compute_metrics
try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    def gaussian_filter(array, sigma=0):
        return array
from utils.feature_transform import create_feature_transform
from utils.analysis import get_classification_from_segmentation, analyze_classification_distribution
from utils.anomaly_detection import generate_anomaly_map_from_tokens
from utils.backbone_adapters import load_visualad_model, resolve_backbone_name
from utils.backbone_config import resolve_features_list
from utils.checkpoint_io import load_trusted_checkpoint
from utils.experiment_io import save_args_json, save_per_image_results_csv
from utils.path_utils import backbone_tag, default_test_dir, experiment_dir_from_checkpoint, next_results_dir
from utils.refinement_head import RefinementHead
from utils.residual_adapters import configure_residual_adapters, load_residual_adapter_state

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_UWBENCH_ROOT = os.path.abspath(os.path.join(REPO_ROOT, '..', 'UW-Bench', 'training_set'))
DEFAULT_UWBENCH_META = os.path.join(REPO_ROOT, 'data_meta', 'uwbench_meta.json')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def get_segmentation_scores(all_anomaly_maps, all_cls_names, results=None):
    """Compute image scores without requiring all visualization maps to be normalized."""
    try:
        return get_classification_from_segmentation(
            all_anomaly_maps,
            all_cls_names,
            results,
            normalize_maps=False,
        )
    except TypeError as exc:
        if "normalize_maps" not in str(exc):
            raise
        # Compatibility with older utils.analysis.py on remote machines.
        return get_classification_from_segmentation(all_anomaly_maps, all_cls_names, results)


def test(args):
    device = torch.device(args.device)

    # Load checkpoint (required)
    checkpoint = load_trusted_checkpoint(args.checkpoint_path, map_location=device)

    # Use checkpoint values
    args.backbone_type = args.backbone_type or checkpoint.get("backbone_type", "clip")
    args.backbone_name = resolve_backbone_name(
        args.backbone_type,
        args.backbone_name or checkpoint.get("backbone_name"),
        checkpoint.get("backbone", "ViT-L/14@336px"),
    )
    args.backbone = checkpoint.get("backbone", "ViT-L/14@336px")
    args.prototype_token_mode = checkpoint.get(
        "prototype_token_mode",
        "backbone" if args.backbone_type in ("clip", "dinov2", "dinov3") else "external",
    )
    args.sam_checkpoint = args.sam_checkpoint or checkpoint.get("sam_checkpoint", "")
    args.image_size = checkpoint.get("image_size", 518)
    args.features_list = checkpoint.get("features_list", [6, 12, 18, 24])
    args.use_residual_adapters = bool(
        getattr(args, "use_residual_adapters", False) or checkpoint.get("use_residual_adapters", False)
    )
    args.adapter_layers = checkpoint.get("adapter_layers", getattr(args, "adapter_layers", "shallow"))
    args.adapter_layer_ids = checkpoint.get("adapter_layer_ids", getattr(args, "adapter_layer_ids", "3,6,9,12"))
    args.adapter_type = checkpoint.get("adapter_type", getattr(args, "adapter_type", "mlp"))
    args.residual_adapter_mode = checkpoint.get("residual_adapter_mode", getattr(args, "residual_adapter_mode", ""))
    args.adapter_ratio = checkpoint.get("adapter_ratio", getattr(args, "adapter_ratio", 0.01))
    args.adapter_bottleneck_ratio = checkpoint.get(
        "adapter_bottleneck_ratio", getattr(args, "adapter_bottleneck_ratio", 0.25)
    )
    args.adapter_dropout = checkpoint.get("adapter_dropout", getattr(args, "adapter_dropout", 0.0))
    args.use_iou_loss = bool(checkpoint.get("use_iou_loss", getattr(args, "use_iou_loss", False)))
    args.use_refined_mask = bool(checkpoint.get("use_refined_mask", getattr(args, "use_refined_mask", False)))
    args.refinement_hidden_dim = checkpoint.get("refinement_hidden_dim", getattr(args, "refinement_hidden_dim", 256))
    args.refinement_dropout = checkpoint.get("refinement_dropout", getattr(args, "refinement_dropout", 0.0))
    args.num_anchors = int(
        checkpoint.get(
            "num_anchors",
            checkpoint.get("cross_attn_config", {}).get("num_anchors", getattr(args, "num_anchors", 4)),
        )
    )
    tag = backbone_tag(args.backbone_type, args.backbone_name)
    if args.save_path is None:
        if args.use_residual_adapters:
            args.save_path = next_results_dir(experiment_dir_from_checkpoint(args.checkpoint_path))
        else:
            args.save_path = default_test_dir(args.checkpoint_path, tag, args.sigma)

    logger = get_logger(args.save_path, filename='test.log')
    os.makedirs(args.save_path, exist_ok=True)
    args.pre_mask_threshold_source = "sigmoid(filtered_anomaly_map) >= eval_threshold"

    # Load model before transforms so the checkpoint can restore backbone defaults.
    model, _, backbone_spec = load_visualad_model(args, device=device)
    args.features_list = resolve_features_list(args.features_list, backbone_spec.num_layers, logger=logger)
    configure_residual_adapters(model, args, backbone_spec, device, logger=logger)
    load_residual_adapter_state(model, checkpoint.get("adapter_state_dict"), logger=logger)
    save_args_json(
        args,
        os.path.join(args.save_path, 'test_config.json'),
        extra={
            "backbone_embed_dim": backbone_spec.embed_dim,
            "backbone_num_layers": backbone_spec.num_layers,
            "backbone_patch_size": backbone_spec.patch_size,
            "pre_mask_threshold_source": args.pre_mask_threshold_source,
        },
    )
    logger.info(f"Testing: {args.test_dataset} | Backbone: {args.backbone_type}:{args.backbone_name} | "
                f"Image: {args.image_size} | Layers: {args.features_list} | Device: {args.device}")
    logger.info(f"Prototype token mode: {getattr(args, 'prototype_token_mode', 'unknown')}")
    logger.info(f"Checkpoint: {args.checkpoint_path} | Save path: {args.save_path}")
    logger.info(f"Residual adapters: enabled={args.use_residual_adapters} | type={args.adapter_type} | "
                f"mode={getattr(args, 'residual_adapter_mode', '')} | layers={args.adapter_layer_ids} | "
                f"ratio={args.adapter_ratio} | bottleneck_ratio={args.adapter_bottleneck_ratio}")
    logger.info(f"Refined mask: {args.use_refined_mask} | Sigma: {args.sigma} | "
                f"eval_threshold: {args.eval_threshold} | pre_mask_source: {args.pre_mask_threshold_source}")
    logger.info(f"Spatial-aware cross-attention: num_anchors={args.num_anchors}")
    logger.info(f"Metrics mode: {args.metrics_mode} | fast_metrics_stride: {args.fast_metrics_stride} | "
                f"metrics_decimals: {args.metrics_decimals} | analysis_max_samples: {args.analysis_max_samples}")

    preprocess, target_transform = get_transform(args)

    model.eval()
    model.to(device)

    feature_dim = model.visual.embed_dim

    # Load trained tokens
    model.visual.anomaly_token.data = checkpoint["anomaly_token"].to(device)
    model.visual.normal_token.data = checkpoint["normal_token"].to(device)
    ln_post = getattr(model.visual, "ln_post", None)
    if ln_post is not None and checkpoint.get("ln_post_weight") is not None:
        ln_post.weight.data = checkpoint["ln_post_weight"].to(device)
        ln_post.bias.data = checkpoint["ln_post_bias"].to(device)

    # Load feature transforms
    layer_transforms = nn.ModuleDict()
    if "layer_transforms" in checkpoint:
        for layer_name, state_dict in checkpoint["layer_transforms"].items():
            hidden_dim = state_dict['mlp.0.weight'].shape[0]
            layer_transforms[layer_name] = create_feature_transform(
                transform_type="mlp",
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                output_dim=feature_dim,
                dropout=0.0
            ).to(device)
            layer_transforms[layer_name].load_state_dict(state_dict)
            layer_transforms[layer_name].eval()

    # Load cross-attention
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
            res_scale_init=config.get("res_scale_init", 0.01)
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

    # Test dataset
    test_data = Dataset(root=args.test_data_path, transform=preprocess,
                       target_transform=target_transform, dataset_name=args.test_dataset, mode=args.test_mode,
                       meta_path=args.test_meta_path)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    results = {obj: {'gt_sp': [], 'pr_sp': [], 'imgs_masks': [], 'anomaly_maps': []} for obj in obj_list}

    # Data for analysis
    all_original_images = []
    all_anomaly_maps = []
    all_gt_masks = []
    all_cls_names = []
    all_anomaly_labels = []
    all_img_paths = []
    analysis_original_images = []
    analysis_anomaly_maps = []
    analysis_gt_masks = []
    analysis_cls_names = []
    analysis_anomaly_labels = []
    analysis_img_paths = []
    analysis_indices = []
    analysis_max_samples = int(getattr(args, "analysis_max_samples", 0))

    for items in tqdm(test_dataloader):
        image = items['img'].to(device)
        cls_name = items['cls_name'][0]
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

        results[cls_name]['imgs_masks'].append(gt_mask)
        results[cls_name]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            vision_output = model.encode_image(image, args.features_list)
            anomaly_features = vision_output['anomaly_features']
            normal_features = vision_output['normal_features']
            patch_tokens = vision_output['patch_tokens']
            patch_start_idx = vision_output['patch_start_idx']

            # Cross-Attention enhancement
            patch_features_list = [pt[:, patch_start_idx:, :] for pt in patch_tokens]
            if cross_attn is not None:
                adapted_list = cross_attn(anomaly_features, normal_features, patch_features_list, args.features_list)
                anomaly_features_list = [a['anomaly'] for a in adapted_list]
                normal_features_list = [a['normal'] for a in adapted_list]
            else:
                anomaly_features_list = [anomaly_features] * len(patch_tokens)
                normal_features_list = [normal_features] * len(patch_tokens)

            # Generate anomaly maps
            anomaly_map_list = []
            refinement_features_list = []
            for idx, patch_feature in enumerate(patch_tokens):
                anomaly_feat_norm = F.normalize(anomaly_features_list[idx], dim=1, eps=1e-8)
                normal_feat_norm = F.normalize(normal_features_list[idx], dim=1, eps=1e-8)

                transform_key = f'layer_{args.features_list[idx]}'
                if transform_key in layer_transforms:
                    B, N, D = patch_feature.shape
                    patch_feature = layer_transforms[transform_key](patch_feature.view(-1, D)).view(B, N, D)
                patch_only_feature = patch_feature[:, patch_start_idx:, :]
                refinement_features_list.append(patch_only_feature)

                anomaly_map = generate_anomaly_map_from_tokens(
                    anomaly_feat_norm, normal_feat_norm,
                    patch_only_feature,
                    args.image_size
                )
                anomaly_map_list.append(anomaly_map)

            # Fuse and filter
            if refinement_head is not None:
                final_anomaly_map = refinement_head(
                    refinement_features_list,
                    anomaly_map_list,
                    args.image_size,
                ).cpu()
            else:
                final_anomaly_map = torch.stack(anomaly_map_list).sum(dim=0).cpu()
            filtered_map = gaussian_filter(final_anomaly_map[0].numpy(), sigma=args.sigma)
            final_anomaly_map = torch.from_numpy(filtered_map).unsqueeze(0)

            results[cls_name]['anomaly_maps'].append(final_anomaly_map)

            all_anomaly_maps.append(final_anomaly_map)
            all_gt_masks.append(gt_mask)
            all_cls_names.append(cls_name)
            all_anomaly_labels.append(items['anomaly'].item())
            all_img_paths.append(items['img_path'][0])

            if args.enable_analysis and (analysis_max_samples <= 0 or len(analysis_indices) < analysis_max_samples):
                analysis_indices.append(len(all_img_paths) - 1)
                analysis_original_images.append(image.detach().cpu())
                analysis_anomaly_maps.append(final_anomaly_map)
                analysis_gt_masks.append(gt_mask)
                analysis_cls_names.append(cls_name)
                analysis_anomaly_labels.append(items['anomaly'].item())
                analysis_img_paths.append(items['img_path'][0])

    fused_scores = None
    normalized_anomaly_maps = None
    metrics_summary = None

    if not args.skip_metrics:
        # Image-level scores are required for Accuracy and the image-level anomaly metrics.
        if args.metrics_mode in {"segmentation", "all"} or args.enable_analysis:
            fused_scores, normalized_anomaly_maps = get_segmentation_scores(all_anomaly_maps, all_cls_names, results)
        metrics_summary = compute_metrics(
            results,
            obj_list,
            logger,
            eval_threshold=args.eval_threshold,
            metrics_mode=args.metrics_mode,
            sample_threshold=args.sample_threshold,
            fast_metrics_stride=getattr(args, 'fast_metrics_stride', 1),
            metrics_decimals=getattr(args, 'metrics_decimals', 1),
        )
        metrics_payload = {
            "dataset_name": args.test_dataset,
            "test_data_path": args.test_data_path,
            "checkpoint_path": args.checkpoint_path,
            "backbone_type": args.backbone_type,
            "backbone_name": args.backbone_name,
            "use_residual_adapters": args.use_residual_adapters,
            "adapter_type": args.adapter_type,
            "residual_adapter_mode": getattr(args, "residual_adapter_mode", ""),
            "adapter_layer_ids": args.adapter_layer_ids,
            "adapter_ratio": args.adapter_ratio,
            "adapter_bottleneck_ratio": args.adapter_bottleneck_ratio,
            "use_iou_loss": args.use_iou_loss,
            "use_refined_mask": args.use_refined_mask,
            "sigma": args.sigma,
            "pre_mask_threshold_source": args.pre_mask_threshold_source,
            "metrics": metrics_summary,
        }
        metrics_json_path = os.path.join(args.save_path, 'metrics_summary.json')
        with open(metrics_json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_payload, f, indent=2)

    # Analysis (optional)
    if args.enable_analysis:
        from utils.visualization import visualize_anomaly_results
        if fused_scores is None or normalized_anomaly_maps is None:
            fused_scores, normalized_anomaly_maps = get_segmentation_scores(all_anomaly_maps, all_cls_names, results)
        analysis_dir = os.path.join(args.save_path, 'analysis')
        analyze_classification_distribution(fused_scores, all_cls_names, all_anomaly_labels, analysis_dir)
        analysis_scores = [fused_scores[idx] for idx in analysis_indices]
        logger.info(
            "Saving analysis visualizations: %d/%d samples | pre_mask_source=%s",
            len(analysis_indices),
            len(all_img_paths),
            args.pre_mask_threshold_source,
        )
        visualize_anomaly_results(
            analysis_original_images, analysis_anomaly_maps, analysis_gt_masks, analysis_scores,
            analysis_cls_names, analysis_img_paths, analysis_anomaly_labels, args.test_dataset, analysis_dir,
            prediction_maps=analysis_anomaly_maps,
            eval_threshold=args.eval_threshold,
        )

    if fused_scores is None:
        fused_scores, normalized_anomaly_maps = get_segmentation_scores(all_anomaly_maps, all_cls_names, None)
    save_per_image_results_csv(
        os.path.join(args.save_path, 'per_image_results.csv'),
        all_img_paths,
        all_cls_names,
        all_anomaly_labels,
        fused_scores,
    )
    return {
        "metrics": metrics_summary,
        "num_samples": len(all_img_paths),
        "save_path": args.save_path,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser("ProtoWD Test", add_help=True)
    parser.add_argument("--test_data_path", type=str, default=DEFAULT_UWBENCH_ROOT, help="test dataset path")
    parser.add_argument("--test_meta_path", type=str, default=DEFAULT_UWBENCH_META, help="optional meta.json path")
    parser.add_argument("--save_path", type=str, default=None, help='path to save test results')
    parser.add_argument("--test_dataset", type=str, default='uwbench', help="test dataset name")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to trained model checkpoint")
    parser.add_argument("--backbone_type", type=str, default=None, choices=["clip", "dinov2", "dinov3", "sam"],
                        help="optional override for checkpoint backbone family")
    parser.add_argument("--backbone_name", type=str, default=None,
                        help="optional override for checkpoint backbone name")
    parser.add_argument("--sam_checkpoint", type=str, default="",
                        help="optional path to a SAM checkpoint; otherwise model_cache/sam is used")
    parser.add_argument("--test_mode", type=str, default="test", help="split name in meta.json to evaluate")
    parser.add_argument("--sigma", type=int, default=4, help="gaussian filter sigma")
    parser.add_argument("--num_anchors", type=int, default=4,
                        help="metadata fallback for older checkpoints; restored from checkpoint when available")
    parser.add_argument("--eval_threshold", type=float, default=0.5, help="threshold for binary segmentation metrics")
    parser.add_argument("--metrics_mode", type=str, default="all", choices=["segmentation", "all"],
                        help="all: compute segmentation + anomaly-detection style metrics; segmentation: only the main water-segmentation metrics")
    parser.add_argument("--sample_threshold", type=float, default=0.0,
                        help="threshold for image-level water/no-water classification from the raw image score")
    parser.add_argument("--fast_metrics_stride", type=int, default=1,
                        help="subsample flattened pixel arrays by this stride for ranking-based metrics (AP/AUROC/best-F1). 1 keeps exact computation")
    parser.add_argument("--metrics_decimals", type=int, default=1,
                        help="decimal places used in the logged metrics table")
    parser.add_argument("--skip_metrics", action="store_true",
                        help="skip metric-table computation; with --enable_analysis the script still computes image-level scores and normalized maps needed for visualization")
    parser.add_argument("--stretch_to_square", action="store_true",
                        help="stretch images and masks to a square instead of preserving aspect ratio with padding")
    parser.add_argument("--device", type=str, default="cuda:1", help="device to use")
    parser.add_argument("--enable_analysis", action="store_true", help="enable data analysis and visualization")
    parser.add_argument("--analysis_max_samples", type=int, default=0,
                        help="maximum samples to visualize when --enable_analysis is set; 0 saves all samples")
    parser.add_argument("--use_residual_adapters", action="store_true",
                        help="enable residual adapters for older checkpoints that do not record this flag")
    parser.add_argument("--adapter_layers", type=str, default="shallow",
                        choices=["shallow", "middle", "all", "custom"],
                        help="which backbone layers receive residual adapters")
    parser.add_argument("--adapter_layer_ids", type=str, default="3,6,9,12",
                        help="comma-separated adapter layer ids; restored from checkpoint when available")
    parser.add_argument("--adapter_type", type=str, default="mlp", choices=["mlp", "conv", "aaclip"],
                        help="residual adapter type; restored from checkpoint when available")
    parser.add_argument("--adapter_ratio", type=float, default=0.01,
                        help="initial residual adapter scale")
    parser.add_argument("--adapter_bottleneck_ratio", type=float, default=0.25,
                        help="adapter hidden dimension ratio relative to backbone width")
    parser.add_argument("--adapter_dropout", type=float, default=0.0,
                        help="dropout inside residual adapters")
    parser.add_argument("--use_iou_loss", action="store_true",
                        help="metadata flag restored from checkpoint; test does not compute training losses")
    parser.add_argument("--use_refined_mask", action="store_true",
                        help="enable refined-mask head for older checkpoints that do not record this flag")
    parser.add_argument("--refinement_hidden_dim", type=int, default=256,
                        help="hidden channels in the refined-mask head")
    parser.add_argument("--refinement_dropout", type=float, default=0.0,
                        help="dropout in the refined-mask head")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    args = parser.parse_args()
    setup_seed(args.seed)
    test(args)

