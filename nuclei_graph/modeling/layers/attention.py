"""Source: Nuclei Foundational Model repository."""

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
)


flex_attention = torch.compile(flex_attention, dynamic=True)


class RotarySparseAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()

        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        self.num_heads = num_heads

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, pos: Tensor, block_mask: BlockMask) -> Tensor:
        q, k, v = rearrange(
            self.qkv(x), "b n (three h d) -> three b h n d", three=3, d=self.head_dim
        )

        x_out = flex_attention(q, k, v, block_mask=block_mask)

        if isinstance(x_out, tuple):
            x_out = x_out[0]
        x = rearrange(x_out, "b h n d -> b n (h d)")

        return self.wo(x)
