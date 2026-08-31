"""Source: Nuclei Foundational Model repository."""

import torch
from einops import rearrange
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
