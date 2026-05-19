import ProtoWD_lib
import torch
from torch.cuda.amp import GradScaler, autocast
import argparse
import torch.nn.functional as F
from utils.loss import FocalLoss, BinaryDiceLoss, BinaryIoULoss, ContrastiveLoss
from dataset import Dataset
from utils.logger import get_logger
from utils.training_utils import (
    print_training_parameters, validate_training_setup, setup_model_training,
    create_optimizer, setup_feature_transforms, check_for_nan,
    compute_segmentation_loss, validate_gradients, save_checkpoint
)
from tqdm import tqdm
import numpy as np
import os
import random
from utils.transforms import get_transform
from utils.scoring import reduce_anomaly_map, DEFAULT_TOPK_RATIO
from utils.backbone_adapters import load_visualad_model, resolve_backbone_name
from utils.backbone_config import resolve_features_list
from utils.experiment_io import save_args_json
from utils.path_utils import backbone_tag, checkpoint_dir, default_residual_train_dir, default_train_dir
from utils.refinement_head import RefinementHead
from utils.residual_adapters import configure_residual_adapters
torch.use_deterministic_algorithms(True, warn_only=True)

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
    # Additional deterministic settings
    torch.use_deterministic_algorithms(True, warn_only=True)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def generate_anomaly_map_from_tokens(anomaly_features, normal_features, patch_tokens, image_size):
    """
    Generate pixel-level anomaly map using token features with numerical stability
    Args:
        anomaly_features: [B, dim] - anomaly token features
        normal_features: [B, dim] - normal token features  
        patch_tokens: [B, num_patches, dim] - patch token features
        image_size: target image size
    Returns:
        anomaly_map: [B, H, W] - pixel-level anomaly map
    """
    B = anomaly_features.shape[0]
    
    # Normalize all features to prevent numerical instability in cosine similarity
    anomaly_features_norm = F.normalize(anomaly_features, dim=1, eps=1e-8)
    normal_features_norm = F.normalize(normal_features, dim=1, eps=1e-8)
    patch_tokens_norm = F.normalize(patch_tokens, dim=2, eps=1e-8)
    
    # Compute similarity between each patch and anomaly/normal tokens
    anomaly_sim = torch.cosine_similarity(
        patch_tokens_norm, anomaly_features_norm.unsqueeze(1), dim=2
    )  # [B, num_patches]
    
    normal_sim = torch.cosine_similarity(
        patch_tokens_norm, normal_features_norm.unsqueeze(1), dim=2
    )  # [B, num_patches]
    
    # Anomaly score = anomaly_similarity - normal_similarity
    anomaly_score = anomaly_sim - normal_sim  # [B, num_patches]
    
    # Check for NaN in anomaly scores
    if torch.isnan(anomaly_score).any():
        print(f"Warning: NaN detected in anomaly_score")
        anomaly_score = torch.nan_to_num(anomaly_score, nan=0.0)
    
    # Reshape to spatial dimensions
    num_patches = anomaly_score.shape[1]
    patch_size_from_model = int(np.sqrt(num_patches))
    
    anomaly_map = anomaly_score.reshape(B, patch_size_from_model, patch_size_from_model)
    
    # Resize to target image size
    anomaly_map = F.interpolate(
        anomaly_map.unsqueeze(1), 
        size=(image_size, image_size), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(1)
    
    return anomaly_map


def compute_classification_loss_V2(anomaly_maps_list, labels, device):
    """
    Compute classification loss aligned with test.py inference
    Uses segmentation scores directly without normalization (training doesn't need class-wise normalization)
    Args:
        anomaly_maps_list: List of anomaly maps from different layers [[B, H, W], [B, H, W], ...]
        labels: Ground truth labels [B]
        device: Device
    Returns:
        loss: Binary cross-entropy loss
    """
    if not anomaly_maps_list:
        return torch.tensor(0.0, device=device)
    
    # Sum anomaly maps from all layers (same as test.py)
    final_anomaly_maps = torch.stack(anomaly_maps_list).sum(dim=0)  # [B, H, W]
    
    # Reduce anomaly map with Top-K mean to stabilize classification score
    seg_scores = reduce_anomaly_map(
        final_anomaly_maps,
        mode="topk_mean",
        topk_ratio=DEFAULT_TOPK_RATIO
    )  # [B]
    labels_float = labels.float().to(device)
    loss = F.binary_cross_entropy_with_logits(seg_scores, labels_float)
    
    return loss


def train(args):
    args.backbone_name = resolve_backbone_name(args.backbone_type, args.backbone_name, args.backbone)
    tag = backbone_tag(args.backbone_type, args.backbone_name)
    if args.save_path is None:
        if getattr(args, "use_residual_adapters", False):
            args.save_path = default_residual_train_dir(
                args.experiment_root,
                args.train_dataset,
                args.epoch,
                args.backbone_type,
                args.backbone_name,
                experiment_name=args.residual_experiment_name,
            )
        else:
            args.save_path = default_train_dir(args.experiment_root, args.train_dataset, tag, args.image_size, args.epoch)

    logger = get_logger(args.save_path, filename='train.log')
    device = args.device

    # Load and setup model
    try:
        model, _, backbone_spec = load_visualad_model(args, device=device)
    except (RuntimeError, ImportError, FileNotFoundError, ValueError) as exc:
        logger.error(f"Failed to load backbone {args.backbone_type}:{args.backbone_name}: {exc}")
        raise

    args.features_list = resolve_features_list(args.features_list, backbone_spec.num_layers, logger=logger)
    configure_residual_adapters(model, args, backbone_spec, device, logger=logger)
    model.train()
    model.to(device)

    preprocess, target_transform = get_transform(args)

    print_training_parameters(args, logger)
    save_args_json(
        args,
        os.path.join(args.save_path, 'train_config.json'),
        extra={
            "backbone_embed_dim": backbone_spec.embed_dim,
            "backbone_num_layers": backbone_spec.num_layers,
            "backbone_patch_size": backbone_spec.patch_size,
        },
    )

    validate_training_setup(args, model, device, logger)

    # Spatial-Aware Cross-Attention
    from utils.spatial_cross_attention import build_layer_adaptive_cross_attention
    cross_attn = build_layer_adaptive_cross_attention(
        layers=args.features_list,
        embed_dim=model.visual.embed_dim,
        num_anchors=4,
        dropout=0.1,
        max_patches=4096,
        res_scale_init=0.01
    ).to(device)
    cross_attn.train()

    # Load dataset
    train_data = Dataset(root=args.train_data_path, transform=preprocess,
                       target_transform=target_transform, dataset_name=args.train_dataset, mode='train',
                       meta_path=args.train_meta_path)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    # Setup feature transforms and model training
    feature_dim = model.visual.embed_dim
    layer_transforms = setup_feature_transforms(args.features_list, device, feature_dim)
    refinement_head = None
    if args.use_refined_mask:
        refinement_head = RefinementHead(
            input_dim=feature_dim,
            num_layers=len(args.features_list),
            hidden_dim=args.refinement_hidden_dim,
            dropout=args.refinement_dropout,
        ).to(device)
        refinement_head.train()

    setup_model_training(model)

    optimizer = create_optimizer(model, layer_transforms, args, cross_attn=cross_attn, refinement_head=refinement_head)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.learning_rate * 0.1)

    amp_enabled = False
    scaler = GradScaler(enabled=amp_enabled)

    # Initialize losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_iou = BinaryIoULoss() if args.use_iou_loss else None
    loss_token_relation = ContrastiveLoss(temperature=0.1, margin=0.5)
    
    for epoch in tqdm(range(args.epoch)):
        # Keep model in train mode for gradient computation
        model.train()
        if refinement_head is not None:
            refinement_head.train()

        loss_list = []
        image_loss_list = []
        token_relation_loss_list = []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label = items['anomaly']
            # Squeeze only the channel dimension (dim=1), preserve batch dimension
            gt = items['img_mask'].squeeze(1).to(device)  # [B, 1, H, W] -> [B, H, W]
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            def compute_losses():
                vision_output = model.encode_image(image, args.features_list)
                anomaly_features = vision_output['anomaly_features']
                normal_features = vision_output['normal_features']
                patch_tokens = vision_output['patch_tokens']
                patch_start_idx = vision_output['patch_start_idx']

                # Spatial-Aware Cross-Attention enhancement
                patch_features_list = [pt[:, patch_start_idx:, :] for pt in patch_tokens]

                adapted_list = cross_attn(
                    anomaly_features, normal_features,
                    patch_features_list, args.features_list
                )
                anomaly_features_list = [adapted['anomaly'] for adapted in adapted_list]
                normal_features_list = [adapted['normal'] for adapted in adapted_list]

                # Contrastive Loss: Use enhanced features from last layer
                final_anomaly_features = anomaly_features_list[-1]
                final_normal_features = normal_features_list[-1]

                final_anomaly_features_norm = F.normalize(final_anomaly_features, dim=1, eps=1e-8)
                final_normal_features_norm = F.normalize(final_normal_features, dim=1, eps=1e-8)

                if (check_for_nan(final_anomaly_features_norm, "normalized anomaly features", logger, epoch) or
                    check_for_nan(final_normal_features_norm, "normalized normal features", logger, epoch)):
                    return None

                token_relation_val = loss_token_relation(final_anomaly_features_norm, final_normal_features_norm)
                if check_for_nan(token_relation_val, "contrastive_loss", logger, epoch):
                    return None

                # Merged loop: Generate both Segmentation Maps and Classification Maps
                # Avoid duplicate computation, improve training speed
                similarity_map_list = []
                anomaly_maps_list = []
                refinement_features_list = []

                for idx_layer, patch_feature in enumerate(patch_tokens):
                    # Each layer uses its own enhanced features
                    anomaly_feat_norm = F.normalize(anomaly_features_list[idx_layer], dim=1, eps=1e-8)
                    normal_feat_norm = F.normalize(normal_features_list[idx_layer], dim=1, eps=1e-8)

                    current_layer = args.features_list[idx_layer]
                    transform_key = f'layer_{current_layer}'

                    # Apply feature transform
                    if transform_key in layer_transforms:
                        batch_size, num_patches, feat_dim = patch_feature.shape
                        patch_feature_flat = patch_feature.view(-1, feat_dim)
                        transformed_feature = layer_transforms[transform_key](patch_feature_flat)
                        patch_feature = transformed_feature.view(batch_size, num_patches, feat_dim)
                    patch_only_feature = patch_feature[:, patch_start_idx:, :]
                    refinement_features_list.append(patch_only_feature)

                    # Generate anomaly map (compute only once)
                    anomaly_map = generate_anomaly_map_from_tokens(
                        anomaly_feat_norm, normal_feat_norm,
                        patch_only_feature, args.image_size
                    )

                    # For Segmentation Loss
                    anomaly_map_sigmoid = torch.sigmoid(anomaly_map)
                    similarity_map = torch.stack([1 - anomaly_map_sigmoid, anomaly_map_sigmoid], dim=1)
                    similarity_map_list.append(similarity_map)

                    # For Classification Loss (reuse same anomaly_map)
                    anomaly_maps_list.append(anomaly_map)

                image_val = compute_classification_loss_V2(anomaly_maps_list, label, device)
                if check_for_nan(image_val, "image_loss", logger, epoch):
                    return None

                seg_val = torch.tensor(0.0, device=device)
                if similarity_map_list and (anomaly_features.requires_grad or normal_features.requires_grad):
                    seg_val = compute_segmentation_loss(
                        similarity_map_list,
                        gt,
                        loss_focal,
                        loss_dice,
                        loss_iou=loss_iou,
                        use_iou_loss=args.use_iou_loss,
                    )
                    if refinement_head is not None:
                        refined_logits = refinement_head(refinement_features_list, anomaly_maps_list, args.image_size)
                        refined_prob = torch.sigmoid(refined_logits)
                        refined_similarity_map = torch.stack([1 - refined_prob, refined_prob], dim=1)
                        refined_seg_val = compute_segmentation_loss(
                            [refined_similarity_map],
                            gt,
                            loss_focal,
                            loss_dice,
                            use_iou_loss=False,
                        )
                        seg_val = seg_val + refined_seg_val

                loss_components = []
                if image_val.requires_grad:
                    loss_components.append(image_val)
                if token_relation_val.requires_grad:
                    loss_components.append(token_relation_val)
                if seg_val.requires_grad:
                    loss_components.append(seg_val)

                if not loss_components:
                    logger.error("No loss component requires gradients!")
                    return None

                total = sum(loss_components)
                return total, seg_val, image_val, token_relation_val

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                result = compute_losses()

            if result is None:
                optimizer.zero_grad(set_to_none=True)
                continue

            total_loss, seg_loss_val, image_loss_val, token_relation_loss_val = result

            seg_loss_value = float(seg_loss_val.detach().item())
            image_loss_value = float(image_loss_val.detach().item())
            token_relation_loss_value = float(token_relation_loss_val.detach().item())

            if amp_enabled:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
            else:
                total_loss.backward()

            # Validate gradients after backward pass
            if not validate_gradients(model, logger, epoch):
                optimizer.zero_grad(set_to_none=True)
                continue

            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if check_for_nan(model.visual.anomaly_token, "anomaly_token after update", logger, epoch) or \
               check_for_nan(model.visual.normal_token, "normal_token after update", logger, epoch):
                break

            loss_list.append(seg_loss_value)
            image_loss_list.append(image_loss_value)
            token_relation_loss_list.append(token_relation_loss_value)

        scheduler.step()

        # Log training progress
        if (epoch + 1) % args.print_freq == 0:
            logger.info(f'Epoch [{epoch+1}/{args.epoch}] - seg: {np.mean(loss_list):.4f}, '
                       f'cls: {np.mean(image_loss_list):.4f}, contra: {np.mean(token_relation_loss_list):.4f}')

        # Save model checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, layer_transforms, args, epoch + 1,
                          os.path.join(checkpoint_dir(args.save_path), f'epoch_{epoch + 1}.pth'),
                          cross_attn=cross_attn,
                          refinement_head=refinement_head)

    # Save final model
    final_ckp_path = os.path.join(checkpoint_dir(args.save_path), 'final_model.pth')
    save_checkpoint(model, layer_transforms, args, args.epoch, final_ckp_path,
                   cross_attn=cross_attn,
                   refinement_head=refinement_head)

    logger.info(f'Training completed! Model saved to {final_ckp_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("ProtoWD Training", add_help=True)
    parser.add_argument("--train_data_path", type=str, default=DEFAULT_UWBENCH_ROOT, help="train dataset path")
    parser.add_argument("--train_meta_path", type=str, default=DEFAULT_UWBENCH_META, help="optional meta.json path")
    parser.add_argument("--save_path", type=str, default=None, help='path to save results')
    parser.add_argument("--experiment_root", type=str, default='experiments', help='root for auto-named experiments')
    parser.add_argument("--residual_experiment_name", type=str, default='residual-adapters',
                        help="subdirectory name for residual-adapter experiments")
    parser.add_argument("--train_dataset", type=str, default='uwbench', help="train dataset name")
    parser.add_argument("--backbone_type", type=str, default="clip", choices=["clip", "dinov3", "sam"],
                        help="visual backbone family")
    parser.add_argument("--backbone_name", type=str, default=None,
                        help="model identifier inside the selected backbone family")
    parser.add_argument("--backbone", type=str, default="ViT-L/14@336px", 
                        choices=ProtoWD_lib.available_models(), help="CLIP backbone to use")
    parser.add_argument("--sam_checkpoint", type=str, default="",
                        help="optional path to a SAM checkpoint; otherwise model_cache/sam is used")
    parser.add_argument("--feature_config", type=str, default=os.path.join('configs', 'backbone_layers.yaml'),
                        help="YAML file specifying default feature layers per backbone")
    parser.add_argument("--features_list", type=int, nargs="*", default=[6, 12, 18, 24],
                        help="Override feature layers (falls back to YAML config if omitted)")
    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--stretch_to_square", action="store_true",
                        help="stretch images and masks to a square instead of preserving aspect ratio with padding")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--use_residual_adapters", action="store_true",
                        help="enable AA-CLIP style residual adapters in the visual backbone")
    parser.add_argument("--adapter_layers", type=str, default="shallow",
                        choices=["shallow", "middle", "all", "custom"],
                        help="which backbone layers receive residual adapters")
    parser.add_argument("--adapter_layer_ids", type=str, default="3,6,9,12",
                        help="comma-separated adapter layer ids; used directly when --adapter_layers custom")
    parser.add_argument("--adapter_type", type=str, default="mlp", choices=["mlp", "conv"],
                        help="residual adapter type: mlp keeps the original adapter, conv adds a depthwise convolution branch")
    parser.add_argument("--adapter_ratio", type=float, default=0.01,
                        help="initial residual adapter scale")
    parser.add_argument("--adapter_bottleneck_ratio", type=float, default=0.25,
                        help="adapter hidden dimension ratio relative to backbone width")
    parser.add_argument("--adapter_dropout", type=float, default=0.0,
                        help="dropout inside residual adapters")
    parser.add_argument("--use_refined_mask", action="store_true",
                        help="enable lightweight mask reconstruction head after ProtoWD response maps")
    parser.add_argument("--refinement_hidden_dim", type=int, default=256,
                        help="hidden channels in the refined-mask head")
    parser.add_argument("--refinement_dropout", type=float, default=0.0,
                        help="dropout in the refined-mask head")
    parser.set_defaults(use_iou_loss=False)
    parser.add_argument("--use_iou_loss", dest="use_iou_loss", action="store_true",
                        help="enable optional IoU loss in the segmentation loss")
    parser.add_argument("--no_iou_loss", dest="use_iou_loss", action="store_false",
                        help="disable IoU loss for ablation")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--device", type=str, default="cuda:1", help="device to use")

    args = parser.parse_args()
    setup_seed(args.seed)
    device = torch.device(args.device)
    train(args)
