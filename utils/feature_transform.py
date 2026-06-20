"""
Feature transformation module
Provides simple linear layers, MLP layers, residual MLP, and adapter residual layers for feature transformation
"""

import torch
import torch.nn as nn


class SimpleLinearTransform(nn.Module):
    """Simple linear transformation layer for quick testing"""
    def __init__(self, input_dim, output_dim=None, dropout=0.0):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.transform = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.transform(x)


class FeatureTransformMLP(nn.Module):
    """Complete MLP module for more complex feature transformations"""
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        if output_dim is None:
            output_dim = input_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


class ResidualMLPTransform(nn.Module):
    """Standard MLP with residual branch to both enhance expressiveness and preserve CLIP original features"""

    def __init__(self, input_dim, hidden_dim=None, output_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(16, input_dim // 2)
        if output_dim is None:
            output_dim = input_dim

        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

        self.residual_proj = None
        if output_dim != input_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x if self.residual_proj is None else self.residual_proj(x)
        x = self.norm(x)
        x = self.mlp(x)
        return residual + x


class ResidualAdapterTransform(nn.Module):
    """LoRA-style residual adapter for fine-tuning features while preserving CLIP semantics"""

    def __init__(self, input_dim, hidden_dim=None, output_dim=None, dropout=0.1, init_scale=1e-3):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        if output_dim != input_dim:
            raise ValueError("ResidualAdapterTransform requires output_dim to match input_dim for residual addition")
        if hidden_dim is None:
            hidden_dim = max(16, input_dim // 4)

        self.pre_norm = nn.LayerNorm(input_dim)
        self.down_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dim, input_dim, bias=False)

        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.res_scale = nn.Parameter(torch.ones(1) * init_scale)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)
        for module in self.context_gate:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        residual = x
        x = self.pre_norm(x)
        adapter = self.down_proj(x)
        gate = self.context_gate(adapter)
        adapter = self.act(adapter)
        adapter = adapter * gate
        adapter = self.up_proj(adapter)
        adapter = self.dropout(adapter)
        return residual + self.res_scale * adapter


class AAClipResidualAdapterTransform(nn.Module):
    """AA-CLIP style adapter: linear feature redirection with residual feature mixing."""

    def __init__(
        self,
        input_dim,
        hidden_dim=None,
        output_dim=None,
        dropout=0.0,
        init_scale=0.1,
        negative_slope=0.01,
        eps=1e-6,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        if output_dim != input_dim:
            raise ValueError("AAClipResidualAdapterTransform requires output_dim to match input_dim")

        self.adapter = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=False),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.adapt_weight = float(init_scale)
        self.eps = float(eps)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        adapted = self.adapter(x)
        adapted_norm = adapted.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        residual_norm = x.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        adapted = adapted * residual_norm / adapted_norm
        return self.adapt_weight * adapted + (1.0 - self.adapt_weight) * x


class ConvResidualAdapterTransform(nn.Module):
    """Residual adapter with an MLP branch and a lightweight spatial convolution branch."""

    def __init__(
        self,
        input_dim,
        hidden_dim=None,
        output_dim=None,
        dropout=0.1,
        init_scale=1e-3,
        patch_start_idx=0,
        sequence_first=False,
        kernel_size=3,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        if output_dim != input_dim:
            raise ValueError("ConvResidualAdapterTransform requires output_dim to match input_dim for residual addition")
        if hidden_dim is None:
            hidden_dim = max(16, input_dim // 4)

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.patch_start_idx = int(patch_start_idx)
        self.sequence_first = bool(sequence_first)

        self.pre_norm = nn.LayerNorm(input_dim)

        self.down_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dim, input_dim, bias=False)
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Sigmoid()
        )

        self.conv_down_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        padding = kernel_size // 2
        self.depthwise_conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_dim,
            bias=False,
        )
        self.conv_act = nn.GELU()
        self.conv_up_proj = nn.Linear(hidden_dim, input_dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.res_scale = nn.Parameter(torch.ones(1) * init_scale)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.xavier_uniform_(self.conv_down_proj.weight)
        nn.init.kaiming_uniform_(self.depthwise_conv.weight, a=0.0, nonlinearity="linear")
        nn.init.zeros_(self.conv_up_proj.weight)
        for module in self.context_gate:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _square_side(num_tokens):
        side = int(num_tokens ** 0.5)
        if side * side != num_tokens:
            raise RuntimeError(f"ConvResidualAdapterTransform expects square patch tokens, got {num_tokens}")
        return side

    def _conv_branch_bhwc(self, x):
        batch, height, width, _ = x.shape
        hidden = self.conv_down_proj(x)
        hidden = hidden.permute(0, 3, 1, 2).contiguous()
        hidden = self.depthwise_conv(hidden)
        hidden = self.conv_act(hidden)
        hidden = hidden.permute(0, 2, 3, 1).contiguous()
        return self.conv_up_proj(hidden).view(batch, height, width, self.input_dim)

    def _conv_branch_tokens(self, x):
        if self.sequence_first:
            prefix = x[:self.patch_start_idx]
            patch_tokens = x[self.patch_start_idx:]
            patch_tokens = patch_tokens.permute(1, 0, 2).contiguous()
        else:
            prefix = x[:, :self.patch_start_idx]
            patch_tokens = x[:, self.patch_start_idx:, :]

        batch, num_tokens, _ = patch_tokens.shape
        side = self._square_side(num_tokens)
        hidden = self.conv_down_proj(patch_tokens)
        hidden = hidden.transpose(1, 2).reshape(batch, self.hidden_dim, side, side)
        hidden = self.depthwise_conv(hidden)
        hidden = self.conv_act(hidden)
        hidden = hidden.reshape(batch, self.hidden_dim, num_tokens).transpose(1, 2).contiguous()
        patch_out = self.conv_up_proj(hidden)

        if self.sequence_first:
            patch_out = patch_out.permute(1, 0, 2).contiguous()
            if prefix.numel() == 0:
                return patch_out
            return torch.cat([torch.zeros_like(prefix), patch_out], dim=0)

        if prefix.numel() == 0:
            return patch_out
        return torch.cat([torch.zeros_like(prefix), patch_out], dim=1)

    def forward(self, x):
        residual = x
        x = self.pre_norm(x)

        mlp_hidden = self.down_proj(x)
        gate = self.context_gate(mlp_hidden)
        mlp_hidden = self.act(mlp_hidden) * gate
        mlp_branch = self.up_proj(mlp_hidden)

        if x.dim() == 4:
            conv_branch = self._conv_branch_bhwc(x)
        elif x.dim() == 3:
            conv_branch = self._conv_branch_tokens(x)
        else:
            conv_branch = torch.zeros_like(mlp_branch)

        adapter = self.dropout(mlp_branch + conv_branch)
        return residual + self.res_scale * adapter


class LeakyReLUTransform(nn.Module):
    """Linear+LeakyReLU+Linear+LeakyReLU transformation layer"""
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, dropout=0.1, negative_slope=0.01):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        if output_dim is None:
            output_dim = input_dim

        self.transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.transform(x)


def create_feature_transform(transform_type="linear", input_dim=1024, hidden_dim=None, output_dim=None, dropout=0.1, negative_slope=0.01):
    """
    Factory function to create feature transformation modules

    Args:
        transform_type: "linear", "mlp", "mlp_residual", "adapter" or "leakyrelu"
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension (used by MLP/Adapter/LeakyReLU)
        output_dim: Output feature dimension
        dropout: Dropout rate
        negative_slope: Negative slope for LeakyReLU (only used by LeakyReLU)

    Returns:
        Feature transformation module
    """
    if transform_type == "linear":
        return SimpleLinearTransform(input_dim, output_dim, dropout)
    elif transform_type == "mlp":
        return FeatureTransformMLP(input_dim, hidden_dim, output_dim, dropout)
    elif transform_type == "mlp_residual":
        return ResidualMLPTransform(input_dim, hidden_dim, output_dim, dropout)
    elif transform_type == "adapter":
        return ResidualAdapterTransform(input_dim, hidden_dim, output_dim, dropout)
    elif transform_type == "leakyrelu":
        return LeakyReLUTransform(input_dim, hidden_dim, output_dim, dropout, negative_slope)
    else:
        raise ValueError(f"Unsupported transform_type: {transform_type}. Use 'linear', 'mlp', 'mlp_residual', 'adapter', or 'leakyrelu'")


def create_residual_adapter(
    adapter_type="mlp",
    input_dim=1024,
    hidden_dim=None,
    output_dim=None,
    dropout=0.1,
    init_scale=1e-3,
    patch_start_idx=0,
    sequence_first=False,
):
    """Create a residual adapter used inside visual backbones."""
    adapter_type = (adapter_type or "mlp").lower()
    if adapter_type == "mlp":
        return ResidualAdapterTransform(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            init_scale=init_scale,
        )
    if adapter_type == "conv":
        return ConvResidualAdapterTransform(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            init_scale=init_scale,
            patch_start_idx=patch_start_idx,
            sequence_first=sequence_first,
        )
    if adapter_type == "aaclip":
        return AAClipResidualAdapterTransform(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            init_scale=init_scale,
        )
    raise ValueError(f"Unsupported adapter_type: {adapter_type}. Use 'mlp', 'conv', or 'aaclip'")


# For backward compatibility, keep original class names
LinearTransform = SimpleLinearTransform
MLPTransform = FeatureTransformMLP
