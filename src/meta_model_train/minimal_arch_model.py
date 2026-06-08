from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCH_TOKEN_SPECS = {
    "M": {"name": "mlp", "count_key": "num_mlp"},
    "A": {"name": "attention", "count_key": "num_attention"},
}
ARCH_TOKENS = tuple(ARCH_TOKEN_SPECS.keys())


def canonicalize_architecture_code(code: str) -> str:
    compact = code.replace("-", "").replace(",", "").replace(" ", "").upper()
    if not compact:
        raise ValueError("architecture_code must be non-empty")
    return "-".join(compact)


def parse_architecture_code(code: str) -> list[str]:
    compact = code.replace("-", "").replace(",", "").replace(" ", "").upper()
    tokens = list(compact)
    if not tokens:
        raise ValueError("architecture_code must be non-empty")
    bad = [tok for tok in tokens if tok not in ARCH_TOKENS]
    if bad:
        raise ValueError(f"architecture_code contains invalid tokens: {bad}; allowed tokens are {ARCH_TOKENS}")
    return tokens


def validate_architecture_tokens(tokens: Iterable[str], expected_length: int | None = None) -> list[str]:
    toks = list(tokens)
    if not toks:
        raise ValueError("architecture must contain at least one token")
    if expected_length is not None and len(toks) != expected_length:
        raise ValueError(f"architecture must have exactly {expected_length} tokens, got {len(toks)}")
    return toks


def is_legal_architecture_tokens(tokens: Iterable[str], expected_length: int | None = None) -> bool:
    try:
        validate_architecture_tokens(tokens, expected_length=expected_length)
    except ValueError:
        return False
    return True


def enumerate_legal_architecture_codes(expected_length: int = 8) -> list[str]:
    legal_codes: list[str] = []
    for tokens in product(ARCH_TOKENS, repeat=expected_length):
        if is_legal_architecture_tokens(tokens, expected_length=expected_length):
            legal_codes.append("-".join(tokens))
    return legal_codes


def architecture_token_counts(code: str) -> dict[str, int]:
    tokens = parse_architecture_code(code)
    return {spec["count_key"]: sum(tok == token for tok in tokens) for token, spec in ARCH_TOKEN_SPECS.items()}


def _periodic_delta_indices(length: int, device: torch.device) -> torch.Tensor:
    qpos = torch.arange(length, device=device)[:, None]
    kpos = torch.arange(length, device=device)[None, :]
    return (kpos - qpos) % length


class PeriodicRelativeBias1D(nn.Module):
    def __init__(self, length: int, n_heads: int):
        super().__init__()
        self.length = int(length)
        self.n_heads = int(n_heads)
        self.table = nn.Parameter(torch.zeros(self.length, self.n_heads))

    def forward(self) -> torch.Tensor:
        idx = _periodic_delta_indices(self.length, self.table.device)
        bias = self.table[idx.long()]
        return bias.permute(2, 0, 1).unsqueeze(0).contiguous()


def zero_init_linear(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class ResidualMLPOp(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, bias=True)
        self.fc2 = nn.Linear(dim, dim, bias=True)
        zero_init_linear(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.relu(self.fc1(x)))


class ResidualAxialAttentionOp(nn.Module):
    def __init__(self, dim: int, height: int, width: int, num_heads: int = 1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.height = height
        self.width = width
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_row = nn.Linear(dim, 3 * dim, bias=True)
        self.out_row = nn.Linear(dim, dim, bias=True)
        self.qkv_col = nn.Linear(dim, 3 * dim, bias=True)
        self.out_col = nn.Linear(dim, dim, bias=True)
        zero_init_linear(self.out_row)
        zero_init_linear(self.out_col)

        self.row_bias = PeriodicRelativeBias1D(width, num_heads)
        self.col_bias = PeriodicRelativeBias1D(height, num_heads)

    def _apply_attention(self, x: torch.Tensor, qkv_layer: nn.Linear, out_layer: nn.Linear, bias: torch.Tensor) -> torch.Tensor:
        batch_like, seq_len, dim = x.shape
        qkv = qkv_layer(x).reshape(batch_like, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        out = out.permute(0, 2, 1, 3).reshape(batch_like, seq_len, dim)
        return out_layer(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, dim = x.shape
        if height != self.height or width != self.width or dim != self.dim:
            raise ValueError(f"expected input (B,{self.height},{self.width},{self.dim}), got {tuple(x.shape)}")

        row_in = x.reshape(batch * height, width, dim)
        row_out = self._apply_attention(row_in, self.qkv_row, self.out_row, self.row_bias())
        row_out = row_out.reshape(batch, height, width, dim)

        col_in = x.permute(0, 2, 1, 3).reshape(batch * width, height, dim)
        col_out = self._apply_attention(col_in, self.qkv_col, self.out_col, self.col_bias())
        col_out = col_out.reshape(batch, width, height, dim).permute(0, 2, 1, 3).contiguous()

        return x + 0.5 * (row_out + col_out)


def patchify_2d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, channels, height, width = x.shape
    if channels != 1:
        raise ValueError(f"expected 1 input channel, got {channels}")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"height={height} and width={width} must be divisible by patch_size={patch_size}")
    hp = height // patch_size
    wp = width // patch_size
    x = x.reshape(batch, channels, hp, patch_size, wp, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.reshape(batch, hp, wp, channels * patch_size * patch_size)


def unpatchify_2d(tokens: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, hp, wp, patch_dim = tokens.shape
    expected_patch_dim = patch_size * patch_size
    if patch_dim != expected_patch_dim:
        raise ValueError(f"expected patch_dim={expected_patch_dim}, got {patch_dim}")
    x = tokens.reshape(batch, hp, wp, 1, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.reshape(batch, 1, hp * patch_size, wp * patch_size)


@dataclass(frozen=True)
class MinimalArchConfig:
    image_size: int = 16
    patch_size: int = 2
    hidden_dim: int = 8
    num_heads: int = 1
    architecture_code: str = "M-A-M-A-M-A-M-A"


class MinimalArchModel(nn.Module):
    def __init__(self, cfg: MinimalArchConfig):
        super().__init__()
        if cfg.image_size % cfg.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.cfg = cfg
        self.patch_dim = cfg.patch_size * cfg.patch_size
        self.hp = cfg.image_size // cfg.patch_size
        self.wp = cfg.image_size // cfg.patch_size

        tokens = validate_architecture_tokens(parse_architecture_code(cfg.architecture_code))
        self.architecture_tokens = tokens
        self.architecture_code = canonicalize_architecture_code(cfg.architecture_code)
        self.depth = len(tokens)

        self.input_proj = nn.Linear(self.patch_dim, cfg.hidden_dim, bias=True)
        self.output_proj = nn.Linear(cfg.hidden_dim, self.patch_dim, bias=True)

        ops: list[nn.Module] = []
        for tok in tokens:
            if tok == "M":
                ops.append(ResidualMLPOp(cfg.hidden_dim))
            elif tok == "A":
                ops.append(ResidualAxialAttentionOp(cfg.hidden_dim, self.hp, self.wp, num_heads=cfg.num_heads))
            else:
                raise AssertionError(f"unreachable token: {tok}")
        self.ops = nn.ModuleList(ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected input rank 4, got shape {tuple(x.shape)}")
        patches = patchify_2d(x, self.cfg.patch_size)
        hidden = self.input_proj(patches)
        for op in self.ops:
            hidden = op(hidden)
        pred_patches = self.output_proj(hidden)
        return unpatchify_2d(pred_patches, self.cfg.patch_size)
