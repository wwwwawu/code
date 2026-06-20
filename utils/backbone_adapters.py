"""Backbone adapters for running ProtoWD heads on multiple ViT encoders."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import ProtoWD_lib
from .feature_transform import create_residual_adapter


DEFAULT_BACKBONES = {
    "clip": "ViT-L/14@336px",
    "dinov2": "facebook/dinov2-large",
    "dinov3": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "sam": "vit_l",
}

DINO_V2_HUB_MODELS = {
    "facebook/dinov2-small": "dinov2_vits14",
    "facebook/dinov2-base": "dinov2_vitb14",
    "facebook/dinov2-large": "dinov2_vitl14",
    "facebook/dinov2-giant": "dinov2_vitg14",
    "dinov2_vits14": "dinov2_vits14",
    "dinov2_vitb14": "dinov2_vitb14",
    "dinov2_vitl14": "dinov2_vitl14",
    "dinov2_vitg14": "dinov2_vitg14",
}

DINO_HUB_MODELS = {
    "facebook/dinov3-vits16-pretrain-lvd1689m": "dinov3_vits16",
    "facebook/dinov3-vits16plus-pretrain-lvd1689m": "dinov3_vits16plus",
    "facebook/dinov3-vitb16-pretrain-lvd1689m": "dinov3_vitb16",
    "facebook/dinov3-vitl16-pretrain-lvd1689m": "dinov3_vitl16",
    "facebook/dinov3-vith16plus-pretrain-lvd1689m": "dinov3_vith16plus",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": "dinov3_vit7b16",
}

DINO_WEIGHT_FILES = {
    "facebook/dinov3-vitl16-pretrain-lvd1689m": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "facebook/dinov3-vith16plus-pretrain-lvd1689m": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
}

SAM_CHECKPOINTS = {
    "vit_b": (
        "sam_vit_b_01ec64.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    ),
    "vit_l": (
        "sam_vit_l_0b3195.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    ),
    "vit_h": (
        "sam_vit_h_4b8939.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    ),
}


@dataclass
class BackboneSpec:
    backbone_type: str
    backbone_name: str
    embed_dim: int
    num_layers: int
    patch_size: int
    patch_start_idx: int


class ExternalVisualTokens(nn.Module):
    """Trainable ProtoWD tokens used with frozen external ViT backbones."""

    def __init__(self, embed_dim: int):
        super().__init__()
        scale = embed_dim ** -0.5
        self.embed_dim = embed_dim
        self.anomaly_token = nn.Parameter(scale * torch.randn(embed_dim) * 0.1)
        self.normal_token = nn.Parameter(scale * torch.randn(embed_dim) * 0.1)
        self.ln_post = nn.LayerNorm(embed_dim)


class ExternalBackboneProtoWD(nn.Module):
    """ProtoWD-compatible wrapper for backbones that return patch tokens."""

    def __init__(self, adapter: nn.Module, spec: BackboneSpec):
        super().__init__()
        self.adapter = adapter
        self.visual = ExternalVisualTokens(spec.embed_dim)
        self.backbone_spec = spec
        self.num_layers = spec.num_layers
        self.backbone_type = spec.backbone_type
        self.backbone_name = spec.backbone_name

    def train(self, mode: bool = True):
        super().train(mode)
        self.adapter.eval()
        if hasattr(self.adapter, "set_residual_adapters_train"):
            self.adapter.set_residual_adapters_train(mode)
        return self

    def encode_image(self, image: torch.Tensor, feature_list: Optional[Sequence[int]] = None) -> Dict[str, object]:
        patch_tokens, grid_size = self.adapter(image, feature_list)
        patch_tokens = [self.visual.ln_post(tokens) for tokens in patch_tokens]
        batch_size = image.shape[0]
        anomaly = self.visual.ln_post(self.visual.anomaly_token.unsqueeze(0).expand(batch_size, -1))
        normal = self.visual.ln_post(self.visual.normal_token.unsqueeze(0).expand(batch_size, -1))
        return {
            "anomaly_features": anomaly,
            "normal_features": normal,
            "class_features": None,
            "patch_tokens": patch_tokens,
            "patch_start_idx": 0,
            "grid_size": grid_size,
        }


class DinoV2BackboneProtoWD(nn.Module):
    """ProtoWD wrapper that inserts trainable prototypes into the DINOv2 ViT sequence."""

    def __init__(self, adapter: nn.Module, spec: BackboneSpec):
        super().__init__()
        self.adapter = adapter
        self.visual = ExternalVisualTokens(spec.embed_dim)
        self.backbone_spec = spec
        self.num_layers = spec.num_layers
        self.backbone_type = spec.backbone_type
        self.backbone_name = spec.backbone_name

    def train(self, mode: bool = True):
        super().train(mode)
        self.adapter.eval()
        if hasattr(self.adapter, "set_residual_adapters_train"):
            self.adapter.set_residual_adapters_train(mode)
        return self

    def encode_image(self, image: torch.Tensor, feature_list: Optional[Sequence[int]] = None) -> Dict[str, object]:
        output = self.adapter.forward_with_proto_tokens(
            image=image,
            feature_list=feature_list,
            anomaly_token=self.visual.anomaly_token,
            normal_token=self.visual.normal_token,
        )
        patch_tokens = [self.visual.ln_post(tokens) for tokens in output["patch_tokens"]]
        anomaly = self.visual.ln_post(output["anomaly_features"])
        normal = self.visual.ln_post(output["normal_features"])
        class_features = output.get("class_features")
        if class_features is not None:
            class_features = self.visual.ln_post(class_features)

        return {
            "anomaly_features": anomaly,
            "normal_features": normal,
            "class_features": class_features,
            "patch_tokens": patch_tokens,
            "patch_start_idx": output["patch_start_idx"],
            "grid_size": output["grid_size"],
        }


class DinoV3Adapter(nn.Module):
    def __init__(self, model_name: str, cache_dir: str):
        super().__init__()
        os.makedirs(cache_dir, exist_ok=True)
        self.model = self._load_model(model_name, cache_dir)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.config = getattr(self.model, "config", None)
        self.embed_dim = int(
            getattr(self.model, "embed_dim", 0)
            or getattr(self.model, "num_features", 0)
            or getattr(self.config, "hidden_size", getattr(self.config, "hidden_dim", 0))
        )
        self.num_layers = int(
            getattr(self.model, "n_blocks", 0)
            or len(getattr(self.model, "blocks", []))
            or getattr(self.config, "num_hidden_layers", 0)
        )
        self.patch_size = int(
            getattr(self.model, "patch_size", 0)
            or getattr(getattr(self.model, "patch_embed", None), "patch_size", 16)
            or getattr(self.config, "patch_size", 16)
        )
        if isinstance(self.patch_size, (tuple, list)):
            self.patch_size = int(self.patch_size[0])
        self.num_register_tokens = int(
            getattr(self.model, "n_storage_tokens", 0)
            or getattr(self.config, "num_register_tokens", getattr(getattr(self.model, "embeddings", None), "num_register_tokens", 0))
        )
        self.residual_adapters = nn.ModuleDict()
        self.residual_adapter_handles = []
        self.residual_adapter_mode = "none"

    def _load_model(self, model_name: str, cache_dir: str):
        weight_path = _resolve_dinov3_weight_path(model_name, cache_dir)
        if weight_path:
            hub_name = _resolve_dinov3_hub_name(model_name, weight_path)
            repo_dir = os.environ.get("DINOV3_REPO", os.path.join(cache_dir, "dinov3"))
            if os.path.exists(os.path.join(repo_dir, "dinov3", "hub", "backbones.py")):
                if repo_dir not in sys.path:
                    sys.path.insert(0, repo_dir)
                _patch_dinov3_torch_compat()
                from dinov3.hub import backbones
                return getattr(backbones, hub_name)(weights=weight_path)
            try:
                return torch.hub.load("facebookresearch/dinov3", hub_name, weights=weight_path)
            except Exception as exc:
                raise RuntimeError(
                    "Found local DINOv3 weights, but the DINOv3 model code is not available locally "
                    "and torch.hub could not load it from GitHub. Clone https://github.com/facebookresearch/dinov3 "
                    f"to {repo_dir} or set DINOV3_REPO=/path/to/dinov3, then retry. "
                    f"Local weights: {weight_path}"
                ) from exc

        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise ImportError(
                "DINOv3 requires either a local .pth weight plus the official DINOv3 repo for torch.hub, "
                "or transformers>=4.56 with a compatible PyTorch version."
            ) from exc

        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        config.output_hidden_states = True
        return AutoModel.from_pretrained(
            model_name,
            config=config,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

    def _find_transformer_blocks(self):
        """Find the DINOv3 transformer block list across common HF layouts."""
        candidate_paths = (
            "encoder.layer",
            "encoder.layers",
            "encoder.blocks",
            "layer",
            "layers",
            "blocks",
            "model.encoder.layer",
            "model.encoder.layers",
            "backbone.encoder.layer",
            "backbone.encoder.layers",
        )
        for path in candidate_paths:
            module = self.model
            found = True
            for part in path.split("."):
                if not hasattr(module, part):
                    found = False
                    break
                module = getattr(module, part)
            if found and isinstance(module, (nn.ModuleList, list, tuple)) and len(module) > 0:
                return module
        return None

    def _clear_residual_adapter_hooks(self) -> None:
        for handle in self.residual_adapter_handles:
            handle.remove()
        self.residual_adapter_handles = []

    @staticmethod
    def _apply_adapter_to_block_output(output, adapter):
        if torch.is_tensor(output):
            return adapter(output)
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            return (adapter(output[0]),) + output[1:]
        if hasattr(output, "last_hidden_state") and torch.is_tensor(output.last_hidden_state):
            output.last_hidden_state = adapter(output.last_hidden_state)
            return output
        return output

    def configure_residual_adapters(self, layer_ids, adapter_ratio, bottleneck_ratio, dropout, adapter_type="mlp", device=None) -> None:
        self._clear_residual_adapter_hooks()
        hidden_dim = max(16, int(self.embed_dim * bottleneck_ratio))
        blocks = self._find_transformer_blocks()
        if blocks is None:
            # Fallback keeps DINOv3 runnable if a future HF implementation hides blocks.
            self.residual_adapter_mode = "patch_output"
            patch_start_idx = 0
        else:
            self.residual_adapter_mode = "internal_block"
            patch_start_idx = self._patch_start_idx()

        self.residual_adapters = nn.ModuleDict({
            str(layer_id): create_residual_adapter(
                adapter_type=adapter_type,
                input_dim=self.embed_dim,
                hidden_dim=hidden_dim,
                output_dim=self.embed_dim,
                dropout=dropout,
                init_scale=adapter_ratio,
                patch_start_idx=patch_start_idx,
                sequence_first=False,
            ).to(device)
            for layer_id in layer_ids
        })
        if blocks is None:
            return

        def make_hook(layer_id: int):
            def hook(_module, _inputs, output):
                adapter = self.residual_adapters[str(layer_id)]
                return self._apply_adapter_to_block_output(output, adapter)
            return hook

        for layer_id in layer_ids:
            if 1 <= int(layer_id) <= len(blocks):
                self.residual_adapter_handles.append(
                    blocks[int(layer_id) - 1].register_forward_hook(make_hook(int(layer_id)))
                )

    def set_residual_adapters_train(self, mode: bool) -> None:
        self.residual_adapters.train(mode)

    def _patch_start_idx(self) -> int:
        return 1 + max(0, self.num_register_tokens)

    def _patch_embed_tokens(self, image: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Extract DINOv3 patch tokens and their grid size before prefix tokens are added."""
        if not hasattr(self.model, "patch_embed"):
            raise RuntimeError("DINOv3 backbone-token mode requires a model.patch_embed module.")

        patch_tokens = self.model.patch_embed(image)
        if patch_tokens.dim() == 4:
            if patch_tokens.shape[-1] == self.embed_dim:
                grid_size = (int(patch_tokens.shape[1]), int(patch_tokens.shape[2]))
                patch_tokens = patch_tokens.flatten(1, 2)
            else:
                grid_size = (int(patch_tokens.shape[2]), int(patch_tokens.shape[3]))
                patch_tokens = patch_tokens.flatten(2).transpose(1, 2).contiguous()
        elif patch_tokens.dim() == 3:
            height = max(1, image.shape[-2] // self.patch_size)
            width = max(1, image.shape[-1] // self.patch_size)
            if height * width != patch_tokens.shape[1]:
                side = int(patch_tokens.shape[1] ** 0.5)
                if side * side != patch_tokens.shape[1]:
                    raise RuntimeError(f"DINOv3 patch token count is not square: {patch_tokens.shape[1]}")
                height, width = side, side
            grid_size = (int(height), int(width))
        else:
            raise RuntimeError(f"Unsupported DINOv3 patch_embed output shape: {tuple(patch_tokens.shape)}")
        return patch_tokens, grid_size

    def forward(self, image: torch.Tensor, feature_list: Optional[Sequence[int]]) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
        layers = _resolve_layers(feature_list, self.num_layers)
        if hasattr(self.model, "get_intermediate_layers"):
            zero_based_layers = [layer - 1 for layer in layers]
            patch_tokens = list(
                self.model.get_intermediate_layers(
                    image,
                    n=zero_based_layers,
                    reshape=False,
                    return_class_token=False,
                    norm=True,
                )
            )
            if self.residual_adapter_mode == "patch_output":
                patch_tokens = [
                    self.residual_adapters[str(layer)](tokens)
                    if str(layer) in self.residual_adapters else tokens
                    for layer, tokens in zip(layers, patch_tokens)
                ]
        else:
            try:
                outputs = self.model(
                    pixel_values=image,
                    output_hidden_states=True,
                    return_dict=True,
                    interpolate_pos_encoding=True,
                )
            except TypeError:
                outputs = self.model(pixel_values=image, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states
            patch_start = self._patch_start_idx()
            patch_tokens = []
            for layer in layers:
                tokens = hidden_states[layer][:, patch_start:, :]
                if self.residual_adapter_mode == "patch_output" and str(layer) in self.residual_adapters:
                    tokens = self.residual_adapters[str(layer)](tokens)
                patch_tokens.append(tokens)
        num_patches = patch_tokens[-1].shape[1]
        side = int(num_patches ** 0.5)
        if side * side != num_patches:
            raise RuntimeError(f"DINOv3 patch token count is not square: {num_patches}")
        return patch_tokens, (side, side)


class DinoV2Adapter(DinoV3Adapter):
    """DINOv2 ViT adapter using the same patch-token interface as DINOv3."""

    def __init__(self, model_name: str, cache_dir: str):
        super().__init__(model_name, cache_dir)
        self.prototype_token_mode = "backbone"
        self.num_register_tokens = self._infer_num_register_tokens()

    def _load_model(self, model_name: str, cache_dir: str):
        hub_name = DINO_V2_HUB_MODELS.get(model_name, model_name)
        repo_dir = os.environ.get("DINOV2_REPO", os.path.join(cache_dir, "dinov2"))
        if os.path.exists(os.path.join(repo_dir, "hubconf.py")):
            return torch.hub.load(repo_dir, hub_name, source="local")
        try:
            return torch.hub.load("facebookresearch/dinov2", hub_name)
        except Exception as exc:
            raise RuntimeError(
                "DINOv2 requires either internet access for torch.hub or the official DINOv2 repo "
                f"cloned to {repo_dir}. Clone https://github.com/facebookresearch/dinov2 "
                "or set DINOV2_REPO=/path/to/dinov2, then retry."
            ) from exc

    def _infer_num_register_tokens(self) -> int:
        register_tokens = getattr(self.model, "register_tokens", None)
        if torch.is_tensor(register_tokens):
            return int(register_tokens.shape[1])
        return int(
            getattr(self.model, "num_register_tokens", 0)
            or getattr(self.model, "n_register_tokens", 0)
            or 0
        )

    def _patch_start_idx(self) -> int:
        # DINOv2 backbone-token mode uses [t_w, t_b, cls, register..., patch...].
        return 3 + max(0, self.num_register_tokens)

    def _patch_embed_tokens(self, image: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if not hasattr(self.model, "patch_embed"):
            raise RuntimeError("DINOv2 backbone-token mode requires a model.patch_embed module.")

        patch_tokens = self.model.patch_embed(image)
        if patch_tokens.dim() == 4:
            if patch_tokens.shape[-1] == self.embed_dim:
                grid_size = (int(patch_tokens.shape[1]), int(patch_tokens.shape[2]))
                patch_tokens = patch_tokens.flatten(1, 2)
            else:
                grid_size = (int(patch_tokens.shape[2]), int(patch_tokens.shape[3]))
                patch_tokens = patch_tokens.flatten(2).transpose(1, 2).contiguous()
        elif patch_tokens.dim() == 3:
            height = max(1, image.shape[-2] // self.patch_size)
            width = max(1, image.shape[-1] // self.patch_size)
            if height * width != patch_tokens.shape[1]:
                side = int(patch_tokens.shape[1] ** 0.5)
                if side * side != patch_tokens.shape[1]:
                    raise RuntimeError(f"DINOv2 patch token count is not square: {patch_tokens.shape[1]}")
                height, width = side, side
            grid_size = (int(height), int(width))
        else:
            raise RuntimeError(f"Unsupported DINOv2 patch_embed output shape: {tuple(patch_tokens.shape)}")
        return patch_tokens, grid_size

    def _interpolate_dinov2_pos(self, base_tokens: torch.Tensor, image: torch.Tensor, grid_size: Tuple[int, int]):
        if hasattr(self.model, "interpolate_pos_encoding"):
            try:
                return self.model.interpolate_pos_encoding(base_tokens, image.shape[-1], image.shape[-2])
            except TypeError:
                return self.model.interpolate_pos_encoding(base_tokens, grid_size[1], grid_size[0])

        pos_embed = getattr(self.model, "pos_embed", None)
        if pos_embed is None:
            return torch.zeros(
                1,
                base_tokens.shape[1],
                base_tokens.shape[-1],
                dtype=base_tokens.dtype,
                device=base_tokens.device,
            )

        cls_pos = pos_embed[:, :1, :]
        patch_pos = pos_embed[:, 1:, :]
        if patch_pos.shape[1] != base_tokens.shape[1] - 1:
            src_side = int(patch_pos.shape[1] ** 0.5)
            if src_side * src_side != patch_pos.shape[1]:
                raise RuntimeError(f"Cannot interpolate DINOv2 position embedding with {patch_pos.shape[1]} patches.")
            patch_pos = patch_pos.reshape(1, src_side, src_side, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=grid_size, mode="bicubic", align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)
        return torch.cat([cls_pos, patch_pos], dim=1).to(dtype=base_tokens.dtype, device=base_tokens.device)

    def _register_tokens(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        register_tokens = getattr(self.model, "register_tokens", None)
        if torch.is_tensor(register_tokens):
            return register_tokens.expand(batch_size, -1, -1).to(dtype=dtype, device=device)
        return torch.empty(batch_size, 0, self.embed_dim, dtype=dtype, device=device)

    def _prepare_tokens_with_proto(
        self,
        image: torch.Tensor,
        anomaly_token: torch.Tensor,
        normal_token: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        patch_tokens, grid_size = self._patch_embed_tokens(image)
        batch_size, _, embed_dim = patch_tokens.shape
        dtype = patch_tokens.dtype
        device = patch_tokens.device

        cls_token = getattr(self.model, "cls_token", None)
        if cls_token is None:
            raise RuntimeError("DINOv2 backbone-token mode requires a model.cls_token parameter.")
        class_tokens = cls_token.expand(batch_size, -1, -1).to(dtype=dtype, device=device)

        base_tokens = torch.cat([class_tokens, patch_tokens], dim=1)
        pos_embed = self._interpolate_dinov2_pos(base_tokens, image, grid_size)
        class_pos = pos_embed[:, :1, :].to(dtype=dtype, device=device)
        patch_pos = pos_embed[:, 1:, :].to(dtype=dtype, device=device)

        anomaly = anomaly_token.view(1, 1, embed_dim).expand(batch_size, -1, -1).to(dtype=dtype, device=device)
        normal = normal_token.view(1, 1, embed_dim).expand(batch_size, -1, -1).to(dtype=dtype, device=device)
        register_tokens = self._register_tokens(batch_size, dtype, device)

        tokens = torch.cat(
            [
                anomaly + class_pos,
                normal + class_pos,
                class_tokens + class_pos,
                register_tokens,
                patch_tokens + patch_pos,
            ],
            dim=1,
        )

        pos_drop = getattr(self.model, "pos_drop", None)
        if pos_drop is not None:
            tokens = pos_drop(tokens)
        return tokens, grid_size

    def forward_with_proto_tokens(
        self,
        image: torch.Tensor,
        feature_list: Optional[Sequence[int]],
        anomaly_token: torch.Tensor,
        normal_token: torch.Tensor,
    ) -> Dict[str, object]:
        if not hasattr(self.model, "blocks") or not hasattr(self.model, "norm"):
            raise RuntimeError(
                "DINOv2 backbone-token mode requires a torch.hub/local DINOv2 ViT with blocks and norm. "
                "Use the official facebookresearch/dinov2 model code."
            )

        layers = _resolve_layers(feature_list, self.num_layers)
        tokens, grid_size = self._prepare_tokens_with_proto(image, anomaly_token, normal_token)

        outputs: List[torch.Tensor] = []
        for idx, block in enumerate(self.model.blocks, start=1):
            tokens = block(tokens)
            if idx in layers:
                outputs.append(tokens)

        norm = getattr(self.model, "norm", None)
        if norm is not None:
            outputs = [norm(output) for output in outputs]
            tokens = norm(tokens)

        if len(outputs) != len(layers):
            raise RuntimeError(f"Only captured {len(outputs)} / {len(layers)} DINOv2 layers.")

        patch_start_idx = self._patch_start_idx()
        return {
            "anomaly_features": tokens[:, 0, :],
            "normal_features": tokens[:, 1, :],
            "class_features": tokens[:, 2, :],
            "patch_tokens": outputs,
            "patch_start_idx": patch_start_idx,
            "grid_size": grid_size,
        }


class SAMAdapter(nn.Module):
    def __init__(self, model_name: str, checkpoint_path: str, cache_dir: str):
        super().__init__()
        try:
            from segment_anything import sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "SAM backbones require the official segment-anything package. Install requirements.txt "
                "or run `pip install segment-anything`."
            ) from exc

        checkpoint_path = _resolve_sam_checkpoint(model_name, checkpoint_path, cache_dir)
        self.sam = sam_model_registry[model_name](checkpoint=checkpoint_path)
        self.sam.eval()
        for param in self.sam.parameters():
            param.requires_grad = False

        self.image_encoder = self.sam.image_encoder
        self.embed_dim = int(self.image_encoder.patch_embed.proj.out_channels)
        self.num_layers = len(self.image_encoder.blocks)
        self.patch_size = int(self.image_encoder.patch_embed.proj.kernel_size[0])
        self.image_size = int(self.image_encoder.img_size)
        self.residual_adapters = nn.ModuleDict()

    def configure_residual_adapters(self, layer_ids, adapter_ratio, bottleneck_ratio, dropout, adapter_type="mlp", device=None) -> None:
        hidden_dim = max(16, int(self.embed_dim * bottleneck_ratio))
        self.residual_adapter_mode = "internal_block"
        self.residual_adapters = nn.ModuleDict({
            str(layer_id): create_residual_adapter(
                adapter_type=adapter_type,
                input_dim=self.embed_dim,
                hidden_dim=hidden_dim,
                output_dim=self.embed_dim,
                dropout=dropout,
                init_scale=adapter_ratio,
                patch_start_idx=0,
                sequence_first=False,
            ).to(device)
            for layer_id in layer_ids
        })

    def set_residual_adapters_train(self, mode: bool) -> None:
        self.residual_adapters.train(mode)

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(
            image,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        if image.max().detach() <= 2.0:
            image = image * 255.0
        return self.sam.preprocess(image)

    def forward(self, image: torch.Tensor, feature_list: Optional[Sequence[int]]) -> Tuple[List[torch.Tensor], Tuple[int, int]]:
        layers = _resolve_layers(feature_list, self.num_layers)
        captured: Dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer_id: int):
            def hook(_module, _inputs, output):
                if output.dim() != 4:
                    raise RuntimeError(f"SAM block output must be 4D, got {tuple(output.shape)}")
                adapter = self.residual_adapters[str(layer_id)] if str(layer_id) in self.residual_adapters else None
                if adapter is not None:
                    output = adapter(output)
                if layer_id in layers:
                    captured[layer_id] = output.flatten(1, 2)
                return output
            return hook

        hook_layers = sorted(set(layers) | {int(layer) for layer in self.residual_adapters.keys()})
        for layer in hook_layers:
            handles.append(self.image_encoder.blocks[layer - 1].register_forward_hook(make_hook(layer)))

        try:
            image = self._preprocess(image)
            _ = self.image_encoder(image)
        finally:
            for handle in handles:
                handle.remove()

        patch_tokens = [captured[layer] for layer in layers]
        side = int(patch_tokens[-1].shape[1] ** 0.5)
        return patch_tokens, (side, side)


def _resolve_layers(feature_list: Optional[Sequence[int]], num_layers: int) -> List[int]:
    requested = [int(layer) for layer in (feature_list or []) if int(layer) > 0]
    if not requested:
        requested = [max(1, round(v)) for v in torch.linspace(1, num_layers, steps=min(4, num_layers)).tolist()]
    layers = sorted({min(max(1, layer), num_layers) for layer in requested})
    return layers


def _resolve_dinov3_weight_path(model_name: str, cache_dir: str) -> Optional[str]:
    if model_name and os.path.isfile(model_name):
        return model_name

    filename = DINO_WEIGHT_FILES.get(model_name)
    if filename:
        target = os.path.join(cache_dir, filename)
        if os.path.exists(target):
            return target

    if os.path.isdir(cache_dir):
        pth_files = sorted(
            os.path.join(cache_dir, name)
            for name in os.listdir(cache_dir)
            if name.endswith(".pth")
        )
        if pth_files:
            return pth_files[0]
    return None


def _resolve_dinov3_hub_name(model_name: str, weight_path: str) -> str:
    if model_name in DINO_HUB_MODELS:
        return DINO_HUB_MODELS[model_name]
    filename = os.path.basename(weight_path).lower()
    if "vitl16" in filename:
        return "dinov3_vitl16"
    if "vitb16" in filename:
        return "dinov3_vitb16"
    if "vits16plus" in filename:
        return "dinov3_vits16plus"
    if "vits16" in filename:
        return "dinov3_vits16"
    if "vith16plus" in filename:
        return "dinov3_vith16plus"
    if "vit7b16" in filename:
        return "dinov3_vit7b16"
    raise ValueError(f"Cannot infer DINOv3 hub model name from {model_name} or {weight_path}")


def _patch_dinov3_torch_compat() -> None:
    """Patch tiny compatibility gaps for official DINOv3 code on older torch."""
    if not hasattr(torch, "_dynamo"):
        torch._dynamo = SimpleNamespace(config=SimpleNamespace())
    elif not hasattr(torch._dynamo, "config"):
        torch._dynamo.config = SimpleNamespace()
    if not hasattr(torch._dynamo, "reset_code_caches"):
        torch._dynamo.reset_code_caches = lambda: None

    if not hasattr(torch, "compiler"):
        torch.compiler = SimpleNamespace()
    if not hasattr(torch.compiler, "allow_in_graph"):
        torch.compiler.allow_in_graph = lambda obj: obj
    if not hasattr(torch.compiler, "disable"):
        torch.compiler.disable = lambda obj=None, **_kwargs: obj if obj is not None else (lambda fn: fn)


def _resolve_sam_checkpoint(model_name: str, checkpoint_path: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    if model_name not in SAM_CHECKPOINTS:
        raise ValueError(f"Unsupported SAM backbone {model_name}. Use one of {sorted(SAM_CHECKPOINTS)}")

    filename, url = SAM_CHECKPOINTS[model_name]
    target = os.path.join(cache_dir, filename)
    if os.path.exists(target):
        return target

    try:
        urllib.request.urlretrieve(url, target)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download SAM checkpoint from {url}. Place {filename} in {cache_dir} "
            "or pass --sam_checkpoint."
        ) from exc
    return target


def resolve_backbone_name(backbone_type: str, backbone_name: Optional[str], legacy_backbone: Optional[str]) -> str:
    if backbone_name:
        return backbone_name
    if backbone_type == "clip" and legacy_backbone:
        return legacy_backbone
    return DEFAULT_BACKBONES[backbone_type]


def load_visualad_model(args, device):
    backbone_type = getattr(args, "backbone_type", "clip")
    backbone_name = resolve_backbone_name(
        backbone_type,
        getattr(args, "backbone_name", None),
        getattr(args, "backbone", None),
    )
    args.backbone_type = backbone_type
    args.backbone_name = backbone_name
    args.backbone = backbone_name if backbone_type == "clip" else getattr(args, "backbone", DEFAULT_BACKBONES["clip"])

    if backbone_type == "clip":
        model, preprocess = ProtoWD_lib.load(backbone_name, device=device)
        args.prototype_token_mode = "backbone"
        spec = BackboneSpec(
            backbone_type="clip",
            backbone_name=backbone_name,
            embed_dim=int(model.visual.embed_dim),
            num_layers=int(model.visual.transformer.layers),
            patch_size=int(model.visual.conv1.kernel_size[0]),
            patch_start_idx=3,
        )
        return model, preprocess, spec

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if backbone_type == "dinov2":
        adapter = DinoV2Adapter(backbone_name, os.path.join(repo_root, "model_cache", "dinov2"))
        spec = BackboneSpec(
            backbone_type=backbone_type,
            backbone_name=backbone_name,
            embed_dim=adapter.embed_dim,
            num_layers=adapter.num_layers,
            patch_size=adapter.patch_size,
            patch_start_idx=adapter._patch_start_idx(),
        )
        args.prototype_token_mode = "backbone"
        model = DinoV2BackboneProtoWD(adapter, spec).to(device)
        return model, None, spec
    elif backbone_type == "dinov3":
        adapter = DinoV3Adapter(backbone_name, os.path.join(repo_root, "model_cache", "dinov3"))
        spec = BackboneSpec(
            backbone_type=backbone_type,
            backbone_name=backbone_name,
            embed_dim=adapter.embed_dim,
            num_layers=adapter.num_layers,
            patch_size=adapter.patch_size,
            patch_start_idx=3 + max(0, adapter.num_register_tokens),
        )
        args.prototype_token_mode = "backbone"
        model = DinoV3BackboneProtoWD(adapter, spec).to(device)
        return model, None, spec
    elif backbone_type == "sam":
        adapter = SAMAdapter(
            backbone_name,
            getattr(args, "sam_checkpoint", ""),
            os.path.join(repo_root, "model_cache", "sam"),
        )
    else:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}")

    args.prototype_token_mode = "external"
    spec = BackboneSpec(
        backbone_type=backbone_type,
        backbone_name=backbone_name,
        embed_dim=adapter.embed_dim,
        num_layers=adapter.num_layers,
        patch_size=adapter.patch_size,
        patch_start_idx=0,
    )
    model = ExternalBackboneProtoWD(adapter, spec).to(device)
    return model, None, spec


def get_model_embed_dim(model) -> int:
    return int(getattr(model.visual, "embed_dim"))


def get_model_num_layers(model) -> int:
    if hasattr(model, "num_layers"):
        return int(model.num_layers)
    return int(model.visual.transformer.layers)


class DinoV3BackboneProtoWD(nn.Module):
    """ProtoWD wrapper that inserts learnable prefixes into the DINOv3 ViT sequence."""

    def __init__(self, adapter: DinoV3Adapter, spec: BackboneSpec):
        super().__init__()
        self.adapter = adapter
        self.adapter.prototype_token_mode = "backbone"
        self.visual = ExternalVisualTokens(spec.embed_dim)
        self.backbone_spec = spec
        self.num_layers = spec.num_layers
        self.backbone_type = spec.backbone_type
        self.backbone_name = spec.backbone_name

    def train(self, mode: bool = True):
        super().train(mode)
        self.adapter.eval()
        if hasattr(self.adapter, "set_residual_adapters_train"):
            self.adapter.set_residual_adapters_train(mode)
        return self

    def configure_residual_adapters(self, layer_ids, adapter_ratio, bottleneck_ratio, dropout, adapter_type="mlp", device=None) -> None:
        if not hasattr(self.adapter, "model"):
            return
        blocks = getattr(self.adapter.model, "blocks", None)
        if blocks is None:
            if hasattr(self.adapter, "configure_residual_adapters"):
                self.adapter.configure_residual_adapters(layer_ids, adapter_ratio, bottleneck_ratio, dropout, adapter_type=adapter_type, device=device)
            return
        hidden_dim = max(16, int(self.visual.embed_dim * bottleneck_ratio))
        self.adapter._clear_residual_adapter_hooks()
        self.adapter.residual_adapter_mode = "internal_block"
        self.adapter.residual_adapters = nn.ModuleDict({
            str(layer_id): create_residual_adapter(
                adapter_type=adapter_type,
                input_dim=self.visual.embed_dim,
                hidden_dim=hidden_dim,
                output_dim=self.visual.embed_dim,
                dropout=dropout,
                init_scale=adapter_ratio,
                patch_start_idx=self._patch_start_idx(),
                sequence_first=False,
            ).to(device)
            for layer_id in layer_ids
        })

        def make_hook(layer_id: int):
            def hook(_module, _inputs, output):
                adapter_mod = self.adapter.residual_adapters[str(layer_id)]
                return self.adapter._apply_adapter_to_block_output(output, adapter_mod)
            return hook

        for layer_id in layer_ids:
            if 1 <= int(layer_id) <= len(blocks):
                self.adapter.residual_adapter_handles.append(
                    blocks[int(layer_id) - 1].register_forward_hook(make_hook(int(layer_id)))
                )

    def set_residual_adapters_train(self, mode: bool) -> None:
        if hasattr(self.adapter, "set_residual_adapters_train"):
            self.adapter.set_residual_adapters_train(mode)

    def _patch_start_idx(self) -> int:
        return 3 + max(0, getattr(self.adapter, "num_register_tokens", 0))

    def encode_image(self, image: torch.Tensor, feature_list: Optional[Sequence[int]] = None) -> Dict[str, object]:
        model = self.adapter.model
        if not hasattr(model, "blocks") or not hasattr(model, "norm"):
            raise RuntimeError(
                "DINOv3 backbone-token mode requires a torch.hub/local DINOv3 ViT with blocks and norm. "
                "Use the official facebookresearch/dinov3 model code."
            )
        layers = _resolve_layers(feature_list, self.num_layers)
        patch_tokens, grid_size = self.adapter._patch_embed_tokens(image)
        batch_size, _, embed_dim = patch_tokens.shape
        dtype = patch_tokens.dtype
        device = patch_tokens.device

        cls_token = getattr(model, "cls_token", None)
        if cls_token is None:
            raise RuntimeError("DINOv3 backbone-token mode requires a model.cls_token parameter.")
        class_tokens = cls_token.expand(batch_size, -1, -1).to(dtype=dtype, device=device)

        storage_tokens = getattr(model, "storage_tokens", None)
        if torch.is_tensor(storage_tokens):
            storage_tokens = storage_tokens.expand(batch_size, -1, -1).to(dtype=dtype, device=device)
        else:
            storage_tokens = torch.empty(batch_size, 0, embed_dim, dtype=dtype, device=device)

        anomaly = self.visual.anomaly_token.view(1, 1, embed_dim).expand(batch_size, -1, -1).to(dtype=dtype, device=device)
        normal = self.visual.normal_token.view(1, 1, embed_dim).expand(batch_size, -1, -1).to(dtype=dtype, device=device)
        tokens = torch.cat([anomaly, normal, class_tokens, storage_tokens, patch_tokens], dim=1)

        rope = None
        if hasattr(model, "rope_embed") and model.rope_embed is not None:
            rope = model.rope_embed(H=grid_size[0], W=grid_size[1])

        outputs: List[torch.Tensor] = []
        for idx, block in enumerate(model.blocks, start=1):
            tokens = block(tokens, rope)
            if idx in layers:
                outputs.append(tokens)

        tokens = model.norm(tokens)
        outputs = [model.norm(output) for output in outputs]
        if len(outputs) != len(layers):
            raise RuntimeError(f"Only captured {len(outputs)} / {len(layers)} DINOv3 layers.")

        patch_tokens = [self.visual.ln_post(output) for output in outputs]
        anomaly_features = self.visual.ln_post(tokens[:, 0, :])
        normal_features = self.visual.ln_post(tokens[:, 1, :])
        class_features = self.visual.ln_post(tokens[:, 2, :])
        return {
            "anomaly_features": anomaly_features,
            "normal_features": normal_features,
            "class_features": class_features,
            "patch_tokens": patch_tokens,
            "patch_start_idx": self._patch_start_idx(),
            "grid_size": grid_size,
        }


