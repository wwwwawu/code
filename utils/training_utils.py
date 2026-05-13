"""
Training utility functions for VisualAD
"""
import torch
import torch.nn as nn
from .feature_transform import create_feature_transform
from .residual_adapters import collect_residual_adapter_state, iter_residual_adapter_parameters


def print_training_parameters(args, logger):
    """Print all training parameters before starting training"""
    backbone_type = getattr(args, "backbone_type", "clip")
    backbone_name = getattr(args, "backbone_name", getattr(args, "backbone", "unknown"))
    logger.info(f"Training: {args.train_dataset} | Backbone: {backbone_type}:{backbone_name} | "
                f"Epochs: {args.epoch} | BS: {args.batch_size} | LR: {args.learning_rate} | "
                f"Image: {args.image_size} | Layers: {args.features_list}")
    logger.info(f"Train data: {getattr(args, 'train_data_path', '')} | "
                f"Meta: {getattr(args, 'train_meta_path', '')} | Device: {getattr(args, 'device', '')}")
    logger.info(f"Save path: {getattr(args, 'save_path', '')} | Save freq: {getattr(args, 'save_freq', '')}")
    if getattr(args, "use_residual_adapters", False):
        logger.info(f"Residual adapters: type={getattr(args, 'adapter_type', 'mlp')} | "
                    f"mode={getattr(args, 'residual_adapter_mode', '')} | "
                    f"layers={getattr(args, 'adapter_layer_ids', [])} | "
                    f"ratio={getattr(args, 'adapter_ratio', 0.01)} | "
                    f"bottleneck_ratio={getattr(args, 'adapter_bottleneck_ratio', 0.25)}")
    if getattr(args, "use_refined_mask", False):
        logger.info(f"Refined mask head: hidden_dim={getattr(args, 'refinement_hidden_dim', 256)} | "
                    f"dropout={getattr(args, 'refinement_dropout', 0.0)} | "
                    "loss=focal+dice")
    loss_terms = ["image_bce", "contrastive", "focal", "dice"]
    if getattr(args, "use_iou_loss", False):
        loss_terms.append("iou")
    logger.info(f"Loss terms: {loss_terms} | use_iou_loss={getattr(args, 'use_iou_loss', False)}")


def validate_training_setup(args, model, device, logger):
    """Validate training setup and requirements"""
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == 'cuda':
        # This check is only a smoke forward. Large backbones such as SAM ViT-L
        # can OOM if we validate with the full training batch before training.
        dummy_batch_size = min(1, int(getattr(args, "batch_size", 1)))
        dummy_input = torch.randn(dummy_batch_size, 3, args.image_size, args.image_size).to(device)
        with torch.no_grad():
            _ = model.encode_image(dummy_input, args.features_list)
        del dummy_input
        torch.cuda.empty_cache()

    import os
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training data path does not exist: {args.train_data_path}")


def setup_model_training(model):
    """Configure model parameters for training"""
    for param in model.parameters():
        param.requires_grad = False

    model.visual.anomaly_token.requires_grad = True
    model.visual.normal_token.requires_grad = True

    ln_post = getattr(model.visual, "ln_post", None)
    if ln_post is not None:
        ln_post.weight.requires_grad = True
        ln_post.bias.requires_grad = True

    for param in iter_residual_adapter_parameters(model):
        param.requires_grad = True


def create_optimizer(model, layer_transforms, args, cross_attn=None, refinement_head=None):
    """Create optimizer with different learning rates for different components"""
    optimizer_params = [
        {'params': [model.visual.anomaly_token], 'lr': args.learning_rate, 'weight_decay': 0.01},
        {'params': [model.visual.normal_token], 'lr': args.learning_rate, 'weight_decay': 0.01},
    ]

    ln_post = getattr(model.visual, "ln_post", None)
    if ln_post is not None:
        optimizer_params.append({
            'params': [ln_post.weight, ln_post.bias],
            'lr': args.learning_rate * 0.1,
            'weight_decay': 0.01
        })

    for transform in layer_transforms.values():
        optimizer_params.append({
            'params': transform.parameters(),
            'lr': args.learning_rate * 0.1,
            'weight_decay': 0.01
        })

    if cross_attn is not None:
        optimizer_params.append({
            'params': cross_attn.parameters(),
            'lr': args.learning_rate * 0.1,
            'weight_decay': 0.01
        })

    if refinement_head is not None:
        optimizer_params.append({
            'params': refinement_head.parameters(),
            'lr': args.learning_rate * 0.1,
            'weight_decay': 0.01
        })

    adapter_params = [param for param in iter_residual_adapter_parameters(model) if param.requires_grad]
    if adapter_params:
        optimizer_params.append({
            'params': adapter_params,
            'lr': args.learning_rate * 0.1,
            'weight_decay': 0.01
        })

    return torch.optim.AdamW(optimizer_params, betas=(0.9, 0.999))


def setup_feature_transforms(features_list, device, feature_dim):
    """Setup feature transformation modules"""
    layer_transforms = nn.ModuleDict()
    for layer_idx in features_list:
        hidden_dim = int(feature_dim * 1.0)
        layer_transforms[f'layer_{layer_idx}'] = create_feature_transform(
            transform_type="mlp", input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=feature_dim, dropout=0.1
        ).to(device)
    return layer_transforms


def check_for_nan(tensor, name, logger, epoch=None):
    """Check tensor for NaN values and log if found"""
    if torch.isnan(tensor).any():
        msg = f"NaN detected in {name}"
        if epoch is not None:
            msg += f" at epoch {epoch+1}"
        logger.error(msg)
        return True
    return False


def compute_segmentation_loss(similarity_map_list, gt, loss_focal, loss_dice, loss_iou=None, use_iou_loss=False):
    """Compute segmentation loss from similarity maps"""
    seg_losses = []
    for similarity_map in similarity_map_list:
        # Hardcoded: beta_focal=1.0, beta_dice=1.0
        seg_losses.append(loss_focal(similarity_map, gt))
        # Only use anomaly channel to avoid double-counting (since channel 0 = 1 - channel 1)
        anomaly_prob = similarity_map[:, 1, :, :]
        seg_losses.append(loss_dice(anomaly_prob, gt))
        if use_iou_loss and loss_iou is not None:
            seg_losses.append(loss_iou(anomaly_prob, gt))
    return sum(seg_losses) if seg_losses else torch.tensor(0.0, device=gt.device, requires_grad=False)


def validate_gradients(model, logger, epoch):
    """Validate and clip gradients"""
    if model.visual.anomaly_token.grad is not None:
        if check_for_nan(model.visual.anomaly_token.grad, "anomaly_token gradient", logger, epoch):
            return False
        torch.nn.utils.clip_grad_norm_([model.visual.anomaly_token], max_norm=1.0)
    
    if model.visual.normal_token.grad is not None:
        if check_for_nan(model.visual.normal_token.grad, "normal_token gradient", logger, epoch):
            return False
        torch.nn.utils.clip_grad_norm_([model.visual.normal_token], max_norm=1.0)
    
    return True


def save_checkpoint(model, layer_transforms, args, epoch, checkpoint_path, cross_attn=None, refinement_head=None):
    """Save model checkpoint"""
    import os

    checkpoint_parent = os.path.dirname(checkpoint_path)
    if checkpoint_parent:
        os.makedirs(checkpoint_parent, exist_ok=True)
    transform_state_dict = {name: t.state_dict() for name, t in layer_transforms.items()}

    ln_post = getattr(model.visual, "ln_post", None)
    checkpoint_data = {
        "anomaly_token": model.visual.anomaly_token.data.clone(),
        "normal_token": model.visual.normal_token.data.clone(),
        "ln_post_weight": ln_post.weight.data.clone() if ln_post else None,
        "ln_post_bias": ln_post.bias.data.clone() if ln_post else None,
        "features_list": args.features_list,
        "image_size": args.image_size,
        "epoch": epoch,
        "backbone": getattr(args, "backbone", "ViT-L/14@336px"),
        "backbone_type": getattr(args, "backbone_type", "clip"),
        "backbone_name": getattr(args, "backbone_name", getattr(args, "backbone", "ViT-L/14@336px")),
        "sam_checkpoint": getattr(args, "sam_checkpoint", ""),
        "layer_transforms": transform_state_dict,
        "transform_type": "mlp",
        "use_residual_adapters": bool(getattr(args, "use_residual_adapters", False)),
        "adapter_type": getattr(args, "adapter_type", "mlp"),
        "residual_adapter_mode": getattr(args, "residual_adapter_mode", ""),
        "adapter_layer_ids": list(getattr(args, "adapter_layer_ids", [])),
        "adapter_layers": getattr(args, "adapter_layers", "shallow"),
        "adapter_ratio": float(getattr(args, "adapter_ratio", 0.01)),
        "adapter_bottleneck_ratio": float(getattr(args, "adapter_bottleneck_ratio", 0.25)),
        "adapter_dropout": float(getattr(args, "adapter_dropout", 0.0)),
        "adapter_state_dict": collect_residual_adapter_state(model),
        "use_iou_loss": bool(getattr(args, "use_iou_loss", False)),
        "use_refined_mask": bool(getattr(args, "use_refined_mask", False)),
        "refinement_hidden_dim": int(getattr(args, "refinement_hidden_dim", 256)),
        "refinement_dropout": float(getattr(args, "refinement_dropout", 0.0)),
    }

    if refinement_head is not None:
        checkpoint_data["refinement_head"] = refinement_head.state_dict()
        checkpoint_data["refinement_config"] = {
            "input_dim": int(getattr(refinement_head, "input_dim", 0)),
            "num_layers": int(getattr(refinement_head, "num_layers", 0)),
            "hidden_dim": int(getattr(refinement_head, "hidden_dim", 0)),
            "dropout": float(getattr(args, "refinement_dropout", 0.0)),
        }

    if cross_attn is not None:
        checkpoint_data["cross_attn"] = cross_attn.state_dict()
        checkpoint_data["cross_attn_config"] = {
            "num_anchors": 4,
            "dropout": 0.1,
            "max_patches": 4096,
            "res_scale_init": 0.01,
        }

    torch.save(checkpoint_data, checkpoint_path)
