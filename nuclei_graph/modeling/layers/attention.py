"""Source: Nuclei Foundational Model repository."""

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
)

from nuclei_graph.modeling.layers.rope import RoPE


flex_attention = torch.compile(flex_attention, dynamic=True)


class RotarySparseAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, rotate_v: bool = False) -> None:
        """Initialize the attention module.

        Args:
            dim: Model dimension.
            num_heads: Number of attention heads.
            rotate_v: Also apply RoPE to V, not just Q/K. Used for the
                blank-token position/attention-only ablation.
        """
        super().__init__()

        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.rotate_v = rotate_v

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.rope = RoPE(self.head_dim)

    def forward(self, x: Tensor, pos: Tensor, block_mask: BlockMask) -> Tensor:
        q, k, v = rearrange(
            self.qkv(x), "b n (three h d) -> three b h n d", three=3, d=self.head_dim
        )

        q = self.rope(q, pos)
        k = self.rope(k, pos)
        if self.rotate_v:
            v = self.rope(v, pos)

        x_out = flex_attention(q, k, v, block_mask=block_mask)

        if isinstance(x_out, tuple):
            x_out = x_out[0]
        x = rearrange(x_out, "b h n d -> b n (h d)")

        return self.wo(x)


class RelativePositionValueAttention(nn.Module):
    """Vector attention with relative-position info routed into V.

    Following Point Transformer's vector self-attention formulation.
    Not usable with flex_attention's fused kernel. Uses an explicit
    gather over precomputed k-NN neighbor indices instead.
    """

    def __init__(self, dim: int, num_heads: int, mlp_hidden: int = 64) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.rope = RoPE(self.head_dim)

        self.pos_mlp = nn.Sequential(
            nn.Linear(2, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, self.head_dim),
        )
        self.gate = nn.Linear(self.head_dim, self.head_dim, bias=False)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        neighbor_idx: Tensor,
        neighbor_mask: Tensor,
    ) -> Tensor:
        b, n, _ = x.shape
        assert b == 1, "RelativePositionValueAttention assumes packed b=1 sequences"

        q, k, v = rearrange(
            self.qkv(x), "b n (three h d) -> three b h n d", three=3, d=self.head_dim
        )
        q = self.rope(q, pos)
        k = self.rope(k, pos)

        safe_idx = neighbor_idx.clamp(min=0)
        k_neighbors = k[:, :, safe_idx, :]
        v_neighbors = v[:, :, safe_idx, :]

        pos0 = pos[0]
        rel_pos = pos0[safe_idx] - pos0.unsqueeze(1)

        delta = self.gate(self.pos_mlp(rel_pos))
        delta = repeat(delta, "n k d -> 1 h n k d", h=self.num_heads)
        v_ij = v_neighbors + delta

        scores = torch.einsum("bhnd,bhnkd->bhnk", q, k_neighbors) / (self.head_dim**0.5)
        scores = scores.masked_fill(~neighbor_mask.view(1, 1, n, -1), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # guard: fully-masked (padded) rows

        out = torch.einsum("bhnk,bhnkd->bhnd", attn, v_ij)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.wo(out)
